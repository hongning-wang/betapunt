import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
import distrax
import chex
from flax.linen.initializers import constant, orthogonal, kaiming_normal
from typing import Sequence, NamedTuple, Tuple
from flax.training.train_state import TrainState

# Import the user's game environment
from game import TossHoldemEnv, State, NUM_CARDS, PHASE_PREFLOP, PHASE_DISCARD_OOP, PHASE_DISCARD_DLR, PHASE_FLOP, PHASE_TURN, PHASE_RIVER, STACK_SIZE

# --- 1. CONFIGURATION ---
CONFIG = {
    "LR": 3e-4,
    "NUM_ENVS": 128,          #
    "NUM_STEPS": 128,         #
    "TOTAL_TIMESTEPS": 1e7,   # Adjusted for demo
    "UPDATE_EPOCHS": 4,
    "NUM_MINIBATCHES": 4,
    "GAMMA": 0.999,           #
    "GAE_LAMBDA": 0.95,       #
    "CLIP_EPS": 0.2,
    "ENT_COEF": 0.01,
    "MAX_GRAD_NORM": 0.5,
    "TRINAL_DELTA1": 3.0,     #
    "DEBUG": False,
    "ENV_NAME": "TossHoldem",
}

# --- 2. OBSERVATION & ENCODING LOGIC ---

@chex.dataclass(frozen=True)
class EnvState:
    game_state: State
    action_history: chex.Array  # (24, 2) - Stores (player, action) pairs for history
    history_ptr: int            # Pointer to current history step
    time: int

def get_card_tensor(state: State, player_id: int):
    """
    Constructs the 6-channel card tensor.
    Channels: 0: Hole, 1: Flop, 2: Turn, 3: River, 4: All Public, 5: All Cards
    Shape: (6, 4, 13)
    """
    # 1. Identify cards
    hole_cards = jnp.where(player_id == 0, state.p0_hole, state.p1_hole)
    # Filter out -1 (empty) cards
    board = state.board
    
    # Create empty tensor
    tensor = jnp.zeros((6, 52), dtype=jnp.float32)

    # Helper to scatter 1s
    def fill_channel(channel_idx, cards):
        # Only valid cards (>=0)
        valid_mask = cards >= 0
        idx = cards * valid_mask # Avoid index error, masked out later
        row_vals = jnp.where(valid_mask, 1.0, 0.0)
        return tensor.at[channel_idx, idx].max(row_vals) # Use max to set 1s

    # Channel 0: Hole Cards
    tensor = fill_channel(0, hole_cards)
    
    # Board definitions for Toss Hold'em (Board size 6)
    # Indices: 0-1 (Flop), 2-3 (Discarded->Board), 4 (Turn), 5 (River)
    flop_cards = board[0:2] 
    discarded_cards = board[2:4] # Treated as public/flop extension
    turn_card = board[4:5]
    river_card = board[5:6]
    
    # Channel 1: Flop (including discards for simplicity in this variant)
    tensor = fill_channel(1, jnp.concatenate([flop_cards, discarded_cards]))
    
    # Channel 2: Turn
    tensor = fill_channel(2, turn_card)
    
    # Channel 3: River
    tensor = fill_channel(3, river_card)
    
    # Channel 4: All Public
    tensor = fill_channel(4, board)
    
    # Channel 5: All Known (Hole + Public)
    tensor = fill_channel(5, jnp.concatenate([hole_cards, board]))

    # Reshape (6, 52) -> (6, 4, 13)
    # 0-12: Suit 0, 13-25: Suit 1, etc.
    return tensor.reshape((6, 4, 13))

def get_action_tensor(env_state: EnvState, state: State, legal_mask: chex.Array, observer_id: int):
    """
    Constructs the 24-channel action tensor.
    Shape: (24, 4, 13)
    Dims: 0: Self Act, 1: Opp Act, 2: Sum, 3: Legal
    """
    # Create grid (Time, Feature_Rows, Actions)
    tensor = jnp.zeros((24, 4, 13), dtype=jnp.float32)
    
    # Fill history
    # action_history contains (player_idx, action_idx)
    
    def fill_step(i, t):
        p, a = env_state.action_history[i]
        valid = i < env_state.history_ptr
        
        # Relativize rows: 0 if action was by Observer (Self), 1 if Opponent
        is_self = p == observer_id
        row_idx = jnp.where(is_self, 0, 1)
        
        # Set 1 at (i, row_idx, a) if valid
        t = t.at[i, row_idx, a].set(jnp.where(valid, 1.0, 0.0))
        
        # Row 2: Sum (Union of actions at this step)
        t = t.at[i, 2, a].set(jnp.where(valid, 1.0, 0.0))
        
        # Row 3: Legal Actions 
        is_current = i == env_state.history_ptr
        t = t.at[i, 3, :].set(jnp.where(is_current, legal_mask.astype(jnp.float32), 0.0))
        
        return t

    tensor = jax.lax.fori_loop(0, 24, fill_step, tensor)
    return tensor

class AlphaHoldemWrapper:
    def __init__(self):
        self.env = TossHoldemEnv()

    def reset(self, key):
        state = self.env.reset(key)
        
        env_state = EnvState(
            game_state=state,
            action_history=jnp.zeros((24, 2), dtype=jnp.int32),
            history_ptr=0,
            time=0
        )
        return self._get_obs(env_state, state), env_state

    def step(self, key, env_state, action):
        state = env_state.game_state
        current_p = state.current_player
        
        # Step the environment
        next_state = self.env.step(state, key, action)
        
        # Update history
        new_hist = env_state.action_history.at[env_state.history_ptr].set(
            jnp.array([current_p, action], dtype=jnp.int32)
        )
        new_ptr = jnp.minimum(env_state.history_ptr + 1, 23)
        
        next_env_state = EnvState(
            game_state=next_state,
            action_history=new_hist,
            history_ptr=new_ptr,
            time=env_state.time + 1
        )
        
        obs = self._get_obs(next_env_state, next_state)
        
        # --- NAIVE SELF-PLAY REWARD LOGIC ---
        # Reward must be relative to the player who just acted (current_p).
        # We calculate the total profit/loss for the episode (Stack - Initial 400).
        # We use STACK_SIZE (400) as the baseline.
        
        p0_profit = next_state.p0_stack - STACK_SIZE
        p1_profit = next_state.p1_stack - STACK_SIZE
        
        # If I am P0, I want p0_profit. If I am P1, I want p1_profit.
        # This ensures P1 learns to beat P0, not help P0.
        step_reward = jnp.where(current_p == 0, p0_profit, p1_profit)
        
        # Reward is only non-zero at the end of the game
        reward = jnp.where(next_state.done, step_reward, 0.0)
        
        return obs, next_env_state, reward, next_state.done, {}

    def _get_obs(self, env_state, state):
        # Current player ID acts as the observer
        pid = state.current_player
        
        # 1. Card Features (6, 4, 13) - Already handles P0/P1 flipping internally
        card_feat = get_card_tensor(state, pid)
        
        # 2. Action Features (24, 4, 13)
        legal = self.env.get_legal_actions(state)
        # Pass pid to ensure history rows are relative (0=Me, 1=Opp)
        act_feat = get_action_tensor(env_state, state, legal, observer_id=pid)
        
        return (card_feat, act_feat, legal)

    def action_space_n(self):
        return 13

# --- 3. NETWORK ARCHITECTURE (AlphaHoldem) ---

class ResBlock(nn.Module):
    features: int
    
    @nn.compact
    def __call__(self, x):
        residual = x
        x = nn.Conv(self.features, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.Conv(self.features, kernel_size=(3, 3), padding="SAME")(x)
        return nn.relu(x + residual)

class AlphaHoldemNetwork(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, inputs):
        card_x, act_x, mask = inputs
        
        # --- Card Branch ---
        # Input: (Batch, 6, 4, 13) -> NHWC is standard in Flax Conv usually, 
        # but we need to check data layout. We'll transpose to (B, 4, 13, 6) for Conv.
        card_x = jnp.transpose(card_x, (0, 2, 3, 1)) 
        
        # Conv1
        card_h = nn.Conv(features=64, kernel_size=(3, 3), padding="SAME")(card_x) #
        card_h = nn.relu(card_h)
        # ResBlocks (Paper doesn't specify count, assuming 2-3 standard for light nets)
        card_h = ResBlock(64)(card_h)
        card_h = ResBlock(64)(card_h)
        card_h = ResBlock(64)(card_h)
        card_flat = card_h.reshape((card_h.shape[0], -1))

        # --- Action Branch ---
        # Input: (Batch, 24, 4, 13) -> Transpose to (B, 24, 13, 4)
        act_x = jnp.transpose(act_x, (0, 1, 3, 2))
        
        act_h = nn.Conv(features=64, kernel_size=(3, 3), padding="SAME")(act_x)
        act_h = nn.relu(act_h)
        act_h = ResBlock(64)(act_h)
        act_h = ResBlock(64)(act_h)
        act_h = ResBlock(64)(act_h)
        act_flat = act_h.reshape((act_h.shape[0], -1))

        # --- Fusion ---
        # Concatenate flattened features (Though paper proposes fusion *network*, often FC)
        fused = jnp.concatenate([card_flat, act_flat], axis=-1)
        
        # Fully Connected Layers (6.8M params range)
        x = nn.Dense(1024)(fused)
        x = nn.relu(x)
        x = nn.Dense(512)(x)
        x = nn.relu(x)

        # Heads
        actor_logits = nn.Dense(self.action_dim)(x)
        
        # Apply Mask: Set logits of illegal actions to -1e9
        min_logit = -1e9
        actor_logits = jnp.where(mask, actor_logits, min_logit)
        
        pi = distrax.Categorical(logits=actor_logits)
        
        critic = nn.Dense(1)(x)
        
        return pi, jnp.squeeze(critic, axis=-1)

# --- 4. TRINAL-CLIP PPO LOSS ---

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    # Inputs for network
    obs_card: jnp.ndarray
    obs_act: jnp.ndarray
    obs_mask: jnp.ndarray
    # Extra for Trinal Clip
    chips_player: jnp.ndarray
    chips_opponent: jnp.ndarray
    player_id: jnp.ndarray
    
def make_train(config):
    env_wrapper = AlphaHoldemWrapper()
    
    # Calculate schedules
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]

    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac

    def train(rng):
        # Init Env
        rng, rng_reset = jax.random.split(rng)
        reset_rngs = jax.random.split(rng_reset, config["NUM_ENVS"])
        
        # Vectorized Reset
        obsv, env_states = jax.vmap(env_wrapper.reset)(reset_rngs)
        
        # Init Network
        network = AlphaHoldemNetwork(action_dim=env_wrapper.action_space_n())
        
        # Dummy init pass to get shapes
        rng, rng_init = jax.random.split(rng)
        # obs structure is ((B,6,4,13), (B,24,4,13), (B,13))
        dummy_obs = (
            jnp.zeros((1, 6, 4, 13)),
            jnp.zeros((1, 24, 4, 13)),
            jnp.ones((1, 13), dtype=bool)
        )
        network_params = network.init(rng_init, dummy_obs)
        
        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(learning_rate=linear_schedule, eps=1e-5)
        )
        
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # --- WRAPPER FOR THE UPDATE STEP ---
        def _update_step(runner_state, unused):
            
            # 1. ROLLOUT LOOP
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, rng = runner_state
                (card_obs, act_obs, mask_obs) = last_obs

                # Get game state and current player
                gs = env_state.game_state
                pid = gs.current_player  # <--- Capture current player ID
                
                # Get Chips info for Trinal Clip
                chips_p0 = STACK_SIZE - gs.p0_stack
                chips_p1 = STACK_SIZE - gs.p1_stack
                
                is_p0 = gs.current_player == 0
                chips_player = jnp.where(is_p0, chips_p0, chips_p1)
                chips_opponent = jnp.where(is_p0, chips_p1, chips_p0)

                # Select Action
                rng, rng_act = jax.random.split(rng)
                pi, value = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=rng_act)
                log_prob = pi.log_prob(action)

                # Step Env
                rng, rng_step = jax.random.split(rng)
                rng_step = jax.random.split(rng_step, config["NUM_ENVS"])
                
                step_fn = jax.vmap(env_wrapper.step, in_axes=(0, 0, 0))
                next_obs, next_env_state, reward, done, _ = step_fn(rng_step, env_state, action)
                
                transition = Transition(
                    done, action, value, reward, log_prob, 
                    card_obs, act_obs, mask_obs,
                    chips_player, chips_opponent,
                    pid  # <--- Store player_id
                )
                
                return (train_state, next_env_state, next_obs, rng), transition

            # Execute Rollout
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # 2. GAE CALCULATION
            _, next_env_state, last_obs, _ = runner_state
            _, last_val = network.apply(train_state.params, last_obs)
            last_pid = next_env_state.game_state.current_player

            def _calculate_gae(traj_batch, last_val, last_pid):
                def _get_advantages(carry, transition):
                    gae, next_value, next_pid = carry
                    done, value, reward = transition.done, transition.value, transition.reward
                    curr_pid = transition.player_id
                    
                    # Zero-Sum Value Flip
                    is_switch = next_pid != curr_pid
                    correct_next_val = jnp.where(is_switch, -next_value, next_value)
                    
                    delta = reward + config["GAMMA"] * correct_next_val * (1 - done) - value
                    gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    
                    return (gae, value, curr_pid), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val, last_pid),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val, last_pid)

            # 3. UPDATE PHASE
            def _update_epoch(update_state, unused):
                def _update_minibatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info
                    
                    def _loss_fn(params, traj_batch, gae, targets):
    # 1. Forward Pass
    #    We allow batch_obs to be a tuple for the Flax model
                        batch_obs = (traj_batch.obs_card, traj_batch.obs_act, traj_batch.obs_mask)
                        pi, value = network.apply(params, batch_obs)
    
    #    Flatten value to match target shape (B,)
                        value = value.squeeze()
    
    # 2. Calculate Probability Ratio
    #    ratio = exp(log_prob_new - log_prob_old)
                        log_prob = pi.log_prob(traj_batch.action)
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
    
    # 3. Trinal-Clip PPO Actor Loss (The "Hard Clamp")
    #    This formula unifies the logic for Positive and Negative Advantages:
    #    - Positive Advantage: Behaves like PPO (clips at 1+eps).
    #    - Negative Advantage: 
    #         a) Stops penalties if ratio drops below 1-eps (via r_clip).
    #         b) Caps penalties if ratio spikes above delta1 (via r_tcp).
    
                        clip_eps = config["CLIP_EPS"]         # e.g., 0.2
                        delta1 = config["TRINAL_DELTA1"]      # e.g., 3.0
    
    # Step A: Standard PPO inner clip [0.8, 1.2]
                        r_clip = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    
    # Step B: Trinal Logic
    #    - jnp.maximum(ratio, r_clip): Enforces lower bound (1-eps).
    #                                  If ratio < 0.8, force it to 0.8.
    #    - jnp.minimum(..., delta1):   Enforces upper bound (delta1).
    #                                  If ratio > 3.0, force it to 3.0.
                        r_tcp = jnp.minimum(jnp.maximum(ratio, r_clip), delta1)
    
                        loss_actor = -(r_tcp * gae).mean()
    
    # 4. Dynamic Value Loss Clipping (Trinal Delta 2 & 3)
    #    The paper clips the value target based on chips currently in play.
                        delta2 = traj_batch.chips_player     # Stack invested by hero
                        delta3 = traj_batch.chips_opponent   # Stack invested by opponent
    
    #    Clip targets to range [-Chips_Hero, +Chips_Opponent]
                        targets_clipped = jnp.clip(targets, -delta2, delta3)
    
                        loss_critic = jnp.square(targets_clipped - value).mean()
    
    # 5. Entropy & Total Loss
                        entropy = pi.entropy().mean()
    
                        total_loss = loss_actor + 0.5 * loss_critic - config["ENT_COEF"] * entropy
    
                        return total_loss, (loss_actor, loss_critic, entropy)
                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    (loss, (l_a, l_c, ent)), grads = grad_fn(
                        train_state.params, 
                        traj_batch, 
                        advantages, 
                        targets
                    )
                    
                    # Apply Updates
                    new_train_state = train_state.apply_gradients(grads=grads)
                    
                    # Return new state and metrics
                    return new_train_state, (loss, l_a, l_c, ent)
                # Batching logic
                batch_size = config["NUM_ENVS"] * config["NUM_STEPS"]
                permutation = jax.random.permutation(update_state[1], batch_size)
                
                def flatten_traj(t):
                    return jnp.reshape(t, (batch_size, *t.shape[2:]))
                
                traj_flat = jax.tree_map(flatten_traj, traj_batch)
                adv_flat = advantages.reshape((batch_size,))
                tgt_flat = targets.reshape((batch_size,))
                
                shuffled_traj = jax.tree_map(lambda x: x[permutation], traj_flat)
                shuffled_adv = adv_flat[permutation]
                shuffled_tgt = tgt_flat[permutation]
                
                minibatch_size = batch_size // config["NUM_MINIBATCHES"]
                
                def scan_batch(train_state, i):
                    idx_start = i * minibatch_size
                    idx_end = idx_start + minibatch_size
                    slice_fn = lambda x: x[idx_start:idx_end]
                    batch = (jax.tree_map(slice_fn, shuffled_traj), 
                             slice_fn(shuffled_adv), 
                             slice_fn(shuffled_tgt))
                    return _update_minibatch(train_state, batch)

                train_state, metrics = jax.lax.scan(
                    scan_batch, update_state[0], jnp.arange(config["NUM_MINIBATCHES"])
                )
                return (train_state, update_state[1]), metrics

            # Run updates
            rng = runner_state[3]
            rng, _rng = jax.random.split(rng)
            (train_state, _), metrics = jax.lax.scan(
                _update_epoch, (runner_state[0], _rng), None, config["UPDATE_EPOCHS"]
            )
            
            # Update runner state with new train_state and rng
            new_runner_state = (train_state, next_env_state, last_obs, rng)
            return new_runner_state, metrics

        # --- MAIN TRAIN LOOP ---
        runner_state = (train_state, env_states, obsv, rng)
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metrics}

    return train

if __name__ == "__main__":
    train_fn = make_train(CONFIG)
    rng = jax.random.PRNGKey(42)
    print("Starting training...")
    out = jax.jit(train_fn)(rng)
    print("Training complete.")