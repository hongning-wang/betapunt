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
    "NUM_ENVS": 128,          # [cite: 326]
    "NUM_STEPS": 128,         # [cite: 326]
    "TOTAL_TIMESTEPS": 1e7,   # Adjusted for demo
    "UPDATE_EPOCHS": 4,
    "NUM_MINIBATCHES": 4,
    "GAMMA": 0.999,           # [cite: 324]
    "GAE_LAMBDA": 0.95,       # [cite: 324]
    "CLIP_EPS": 0.2,
    "ENT_COEF": 0.01,
    "MAX_GRAD_NORM": 0.5,
    "TRINAL_DELTA1": 3.0,     # [cite: 323]
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
    Constructs the 6-channel card tensor[cite: 371].
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

def get_action_tensor(env_state: EnvState, state: State, legal_mask: chex.Array):
    """
    Constructs the 24-channel action tensor[cite: 272, 273].
    Shape: (24, 4, 12)
    Dims: 0: P0 Act, 1: P1 Act, 2: Sum, 3: Legal
    """
    # Create grid (Time, Feature_Rows, Actions)
    tensor = jnp.zeros((24, 4, 12), dtype=jnp.float32)
    
    # Fill history
    # action_history contains (player_idx, action_idx)
    # We iterate through the fixed history buffer
    
    def fill_step(i, t):
        p, a = env_state.action_history[i]
        valid = i < env_state.history_ptr
        
        # Row 0 (P0) or 1 (P1)
        row_idx = p 
        
        # Set 1 at (i, row_idx, a) if valid
        t = t.at[i, row_idx, a].set(jnp.where(valid, 1.0, 0.0))
        
        # Row 2: Sum (Union of actions at this step)
        t = t.at[i, 2, a].set(jnp.where(valid, 1.0, 0.0))
        
        # Row 3: Legal Actions (Paper says "legal actions", usually at *that* step)
        # Since we don't store historical legal masks, we map current legal mask to the *current* step
        # or just leave it for the current step. 
        # AlphaHoldem implies encoding historical context. 
        # For simplicity: We only write Current Legal Mask to the *current* timestep pointer.
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
        # Step the environment
        next_state = self.env.step(state, key, action)
        
        # Update history
        # Store (current_player, action)
        current_p = state.current_player
        new_hist = env_state.action_history.at[env_state.history_ptr].set(
            jnp.array([current_p, action], dtype=jnp.int32)
        )
        new_ptr = jnp.minimum(env_state.history_ptr + 1, 23)
        
        # Check reset if done (auto-reset logic often handled in training loop, 
        # but here we just return the transition state)
        # Note: If done, the next observation should technically be from a reset env
        # but standard PPO vectorization handles resets usually.
        # Here we just pass the terminal state info.
        
        next_env_state = EnvState(
            game_state=next_state,
            action_history=new_hist,
            history_ptr=new_ptr,
            time=env_state.time + 1
        )
        
        obs = self._get_obs(next_env_state, next_state)
        
        # Reward: In poker, reward is usually only at terminal state (pot change)
        # We calculate reward as change in stack for P0 (since we train P0)
        # Or just use the 'winner' field at the end.
        reward = 0.0
        
        # If done, calculate reward
        # Win: +Pot/2, Lose: -Pot/2 (Zero sum) or simply Delta Stack.
        # Delta Stack is robust.
        reward = jnp.where(
            next_state.done,
            (next_state.p0_stack - state.p0_stack), # Change in stack
            0.0
        )
        
        return obs, next_env_state, reward, next_state.done, {}

    def _get_obs(self, env_state, state):
        # Current player ID
        pid = state.current_player
        
        # 1. Card Features (6, 4, 13)
        card_feat = get_card_tensor(state, pid)
        
        # 2. Action Features (24, 4, 12)
        legal = self.env.get_legal_actions(state)
        act_feat = get_action_tensor(env_state, state, legal)
        
        return (card_feat, act_feat, legal)

    def action_space_n(self):
        return 12

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
        card_h = nn.Conv(features=64, kernel_size=(3, 3), padding="SAME")(card_x) # [cite: 318]
        card_h = nn.relu(card_h)
        # ResBlocks (Paper doesn't specify count, assuming 2-3 standard for light nets)
        card_h = ResBlock(64)(card_h)
        card_h = ResBlock(64)(card_h)
        card_h = ResBlock(64)(card_h)
        card_flat = card_h.reshape((card_h.shape[0], -1))

        # --- Action Branch ---
        # Input: (Batch, 24, 4, 12) -> Transpose to (B, 24, 12, 4)
        act_x = jnp.transpose(act_x, (0, 1, 3, 2))
        
        act_h = nn.Conv(features=64, kernel_size=(3, 3), padding="SAME")(act_x)
        act_h = nn.relu(act_h)
        act_h = ResBlock(64)(act_h)
        act_h = ResBlock(64)(act_h)
        act_h = ResBlock(64)(act_h)
        act_flat = act_h.reshape((act_h.shape[0], -1))

        # --- Fusion ---
        # Concatenate flattened features [cite: 265] (Though paper proposes fusion *network*, often FC)
        fused = jnp.concatenate([card_flat, act_flat], axis=-1)
        
        # Fully Connected Layers [cite: 318] (6.8M params range)
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
        # obs structure is ((B,6,4,13), (B,24,4,12), (B,12))
        dummy_obs = (
            jnp.zeros((1, 6, 4, 13)),
            jnp.zeros((1, 24, 4, 12)),
            jnp.ones((1, 12), dtype=bool)
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

        # --- ROLLOUT LOOP ---
        def _update_step(runner_state, unused):
            train_state, env_state, last_obs, rng = runner_state
            
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, rng = runner_state
                (card_obs, act_obs, mask_obs) = last_obs

                # Get Chips info for Trinal Clip before stepping [cite: 323]
                # We need chips in pot for current player and opponent
                # State is inside env_state.game_state
                gs = env_state.game_state
                # Calculate chips involved (Stack Size - Current Stack)
                # Note: This is an approximation. Ideally we track exact contribution.
                chips_p0 = STACK_SIZE - gs.p0_stack
                chips_p1 = STACK_SIZE - gs.p1_stack
                
                # Determine who is 'player' (the one acting) and 'opponent'
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
                    chips_player, chips_opponent
                )
                
                return (train_state, next_env_state, next_obs, rng), transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # --- GAE CALCULATION ---
            _, _, last_obs, _ = runner_state
            _, last_val = network.apply(train_state.params, last_obs)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = transition.done, transition.value, transition.reward
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # --- UPDATE PHASE (Trinal-Clip) ---
            def _update_epoch(update_state, unused):
                def _update_minibatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info
                    
                    def _loss_fn(params, traj_batch, gae, targets):
                        # Re-run network
                        batch_obs = (traj_batch.obs_card, traj_batch.obs_act, traj_batch.obs_mask)
                        pi, value = network.apply(params, batch_obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        # Ratios
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        
                        # [cite: 401] Trinal-Clip Policy Loss
                        # L_tcp = E[clip(r, clip(r, 1-e, 1+e), delta1) * A]
                        # Note: The paper formulation: clip(ratio, clip(ratio, 1-eps, 1+eps), delta1)
                        # This logic seems to imply a double clip. 
                        # Standard PPO: min(ratio*A, clip(ratio, 1-e, 1+e)*A)
                        # Trinal Clip Logic from paper eq (3):
                        # If A < 0, we use delta1 as upper bound for the ratio.
                        
                        clip_eps = config["CLIP_EPS"]
                        delta1 = config["TRINAL_DELTA1"]
                        
                        # Standard clipped ratio
                        clipped_ratio_std = jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps)
                        
                        # Trinal logic: When Adv < 0, allow ratio to go up to delta1
                        # The paper equation is slightly ambiguous, but typically means:
                        # upper_bound = delta1 if A < 0 else (1+eps) 
                        # Implementation interpretation:
                        
                        # Calculate Surrogates
                        surr1 = ratio * gae
                        
                        # Custom Clipping for Trinal
                        # "delta1 is set to 3... for policy loss when A_t < 0" [cite: 399]
                        upper_bound = jnp.where(gae < 0, delta1, 1 + clip_eps)
                        lower_bound = 1 - clip_eps
                        
                        clipped_ratio_trinal = jnp.clip(ratio, lower_bound, upper_bound)
                        surr2 = clipped_ratio_trinal * gae
                        
                        loss_actor = -jnp.minimum(surr1, surr2).mean()

                        # [cite: 406] Trinal-Clip Value Loss
                        # delta2 = chips_player, delta3 = chips_opponent
                        # L_tcv = (clip(R_t, -delta2, delta3) - V(s))^2
                        delta2 = traj_batch.chips_player
                        delta3 = traj_batch.chips_opponent
                        
                        # Clip the TARGET returns, not the value function output
                        # Paper: clip(R_gamma, -delta2, delta3)
                        targets_clipped = jnp.clip(targets, -delta2, delta3)
                        
                        loss_critic = jnp.square(targets_clipped - value).mean()
                        
                        entropy = pi.entropy().mean()
                        
                        total_loss = loss_actor + 0.5 * loss_critic - config["ENT_COEF"] * entropy
                        return total_loss, (loss_actor, loss_critic, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    (loss, (l_a, l_c, ent)), grads = grad_fn(train_state.params, traj_batch, advantages, targets)
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, (loss, l_a, l_c, ent)

                # Batching logic
                batch_size = config["NUM_ENVS"] * config["NUM_STEPS"]
                permutation = jax.random.permutation(update_state[1], batch_size)
                
                # Flatten batch
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
                    scan_batch, train_state, jnp.arange(config["NUM_MINIBATCHES"])
                )
                return (train_state, rng), metrics

            # Run updates
            rng, _rng = jax.random.split(rng)
            (train_state, _), metrics = jax.lax.scan(
                _update_epoch, (train_state, _rng), None, config["UPDATE_EPOCHS"]
            )
            
            return runner_state, metrics

        # Main Loop
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