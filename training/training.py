import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
import distrax
import chex
from flax.training.train_state import TrainState
from typing import NamedTuple

# Import the user's game environment
# Ensure your 'game.py' is in the same directory and has these members
from game import TossHoldemEnv, State, STACK_SIZE

# --- 1. CONFIGURATION ---
CONFIG = {
    "LR": 3e-4,
    "NUM_ENVS": 128,
    "NUM_STEPS": 128,
    "TOTAL_TIMESTEPS": 1e7,
    "UPDATE_EPOCHS": 4,
    "NUM_MINIBATCHES": 4,
    "GAMMA": 0.999,
    "GAE_LAMBDA": 0.95,
    "CLIP_EPS": 0.2,
    "ENT_COEF": 0.01,
    "MAX_GRAD_NORM": 0.5,
    "TRINAL_DELTA1": 3.0,     # Trinal-Clip Parameter
    "DEBUG": False,
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
    Shape: (24, 4, 12)
    Dims: 0: Self Act, 1: Opp Act, 2: Sum, 3: Legal
    """
    # Create grid (Time, Feature_Rows, Actions)
    tensor = jnp.zeros((24, 4, 12), dtype=jnp.float32)
    
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
        # 1. Step the physics
        state = env_state.game_state
        current_p = state.current_player
        next_state = self.env.step(state, key, action)
        
        # 2. Update History (Observation Logic)
        new_hist = env_state.action_history.at[env_state.history_ptr].set(
            jnp.array([current_p, action], dtype=jnp.int32)
        )
        new_ptr = jnp.minimum(env_state.history_ptr + 1, 23)
        
        # 3. Compute Reward (Episodic & Zero-Sum)
        # We only emit reward if the game is done.
        # The reward is the profit relative to the player who just acted (current_p).
        
        p0_profit = next_state.p0_stack - STACK_SIZE
        # In zero-sum head-to-head, P1 profit is usually -P0 profit (minus rake if applicable)
        # But we use the explicit stack values to be safe.
        p1_profit = next_state.p1_stack - STACK_SIZE
        
        step_profit = jnp.where(current_p == 0, p0_profit, p1_profit)
        reward = jnp.where(next_state.done, step_profit, 0.0)
        
        # 4. Prepare Next State (Intermediate)
        next_env_state = EnvState(
            game_state=next_state,
            action_history=new_hist,
            history_ptr=new_ptr,
            time=env_state.time + 1
        )
        
        # Note: We return next_state (potentially terminal) here. 
        # The Auto-Reset logic happens in the vmapped roll-out loop.
        obs = self._get_obs(next_env_state, next_state)
        
        return obs, next_env_state, reward, next_state.done

    def _get_obs(self, env_state, state):
        pid = state.current_player
        card_feat = get_card_tensor(state, pid)
        legal = self.env.get_legal_actions(state)
        act_feat = get_action_tensor(env_state, state, legal, observer_id=pid)
        return (card_feat, act_feat, legal)

    def action_space_n(self):
        return 12

# --- 3. NETWORK ARCHITECTURE ---

class ResBlock(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x):
        residual = x
        x = nn.Conv(features=self.features, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.Conv(features=self.features, kernel_size=(3, 3), padding="SAME")(x)
        return nn.relu(x + residual)

class AlphaHoldemNetwork(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, inputs):
        # Unpack inputs:
        # card_obs: (Batch, 6, 4, 13)
        # act_obs:  (Batch, 24, 4, 12)
        # mask:     (Batch, 12)
        card_obs, act_obs, mask = inputs

        # --- 1. CARD BRANCH ---
        # Input shape is (B, C, H, W) = (B, 6, 4, 13)
        # Flax Conv expects (B, H, W, C).
        # We transpose to (B, 4, 13, 6)
        card_x = jnp.transpose(card_obs, (0, 2, 3, 1))

        # Initial Conv
        card_h = nn.Conv(features=64, kernel_size=(3, 3), padding="SAME")(card_x)
        card_h = nn.relu(card_h)

        # Residual Tower (3 Blocks)
        card_h = ResBlock(64)(card_h)
        card_h = ResBlock(64)(card_h)
        card_h = ResBlock(64)(card_h)

        # Flatten
        card_flat = card_h.reshape((card_h.shape[0], -1))

        # --- 2. ACTION BRANCH ---
        # Input shape is (B, Time, Rows, Cols) = (B, 24, 4, 12)
        # We treat this as an image where Time=Height, Action=Width, Rows=Channels.
        # Transpose to (B, 24, 12, 4)
        act_x = jnp.transpose(act_obs, (0, 1, 3, 2))

        # Initial Conv
        act_h = nn.Conv(features=64, kernel_size=(3, 3), padding="SAME")(act_x)
        act_h = nn.relu(act_h)

        # Residual Tower (3 Blocks)
        act_h = ResBlock(64)(act_h)
        act_h = ResBlock(64)(act_h)
        act_h = ResBlock(64)(act_h)

        # Flatten
        act_flat = act_h.reshape((act_h.shape[0], -1))

        # --- 3. FUSION & DENSE LAYERS ---
        # Concatenate the flattened representations
        fused = jnp.concatenate([card_flat, act_flat], axis=-1)

        # Dense Block 1
        x = nn.Dense(features=1024)(fused)
        x = nn.relu(x)

        # Dense Block 2
        x = nn.Dense(features=512)(x)
        x = nn.relu(x)

        # --- 4. OUTPUT HEADS ---
        
        # ACTOR HEAD
        # Output logits for each action
        actor_logits = nn.Dense(features=self.action_dim)(x)

        # MASKING:
        # We set the logits of illegal actions to a very large negative number
        # so the probabilities become 0 after softmax.
        min_logit = -1e9
        actor_logits = jnp.where(mask, actor_logits, min_logit)

        # Create distribution
        pi = distrax.Categorical(logits=actor_logits)

        # CRITIC HEAD
        # Output scalar value
        critic_val = nn.Dense(features=1)(x)
        critic_val = jnp.squeeze(critic_val, axis=-1)

        return pi, critic_val

# --- 4. TRAIN LOGIC ---

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs_card: jnp.ndarray
    obs_act: jnp.ndarray
    obs_mask: jnp.ndarray
    chips_player: jnp.ndarray
    chips_opponent: jnp.ndarray
    player_id: jnp.ndarray

def make_train(config):
    env_wrapper = AlphaHoldemWrapper()
    config["NUM_UPDATES"] = int(config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"])

    def train(rng):
        # Init Env
        rng, rng_reset = jax.random.split(rng)
        reset_rngs = jax.random.split(rng_reset, config["NUM_ENVS"])
        obsv, env_states = jax.vmap(env_wrapper.reset)(reset_rngs)
        
        # Init Network
        network = AlphaHoldemNetwork(action_dim=env_wrapper.action_space_n())
        rng, rng_init = jax.random.split(rng)
        dummy_obs = (jnp.zeros((1, 6, 4, 13)), jnp.zeros((1, 24, 4, 12)), jnp.ones((1, 12), dtype=bool))
        network_params = network.init(rng_init, dummy_obs)
        
        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(learning_rate=config["LR"], eps=1e-5)
        )
        
        train_state = TrainState.create(apply_fn=network.apply, params=network_params, tx=tx)

        # --- ENV STEP WITH AUTO-RESET ---
        def step_env_auto_reset(rng, env_state, action):
            # 1. Step the environment
            obs, next_env_state, reward, done = env_wrapper.step(rng, env_state, action)
            
            # 2. Auto-Reset if Done
            # We must generate a fresh reset state if done, to allow the scan loop to continue.
            rng_reset = jax.random.fold_in(rng, 0) # Simple deterministic fold for reset inside step
            reset_obs, reset_state = env_wrapper.reset(rng_reset)
            
            # Select between continued state or reset state
            final_env_state = jax.tree_map(
                lambda x, y: jnp.where(done, x, y), 
                reset_state, next_env_state
            )
            
            final_obs = jax.tree_map(
                lambda x, y: jnp.where(done, x, y), # Broadcast done for obs structure
                reset_obs, obs
            )
            
            # Note: We return the reward from the *terminal* step, but the observation from the *fresh* step.
            return final_obs, final_env_state, reward, done

        def _update_step(runner_state, unused):
            
            # --- 1. ROLLOUT PHASE ---
            def _env_rollout(runner_state, unused):
                train_state, env_state, last_obs, rng = runner_state
                (card_obs, act_obs, mask_obs) = last_obs

                # Metadata for Trinal Clip / GAE
                gs = env_state.game_state
                pid = gs.current_player
                is_p0 = pid == 0
                chips_player = jnp.where(is_p0, STACK_SIZE - gs.p0_stack, STACK_SIZE - gs.p1_stack)
                chips_opponent = jnp.where(is_p0, STACK_SIZE - gs.p1_stack, STACK_SIZE - gs.p0_stack)

                # Action Selection
                rng, rng_act = jax.random.split(rng)
                pi, value = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=rng_act)
                log_prob = pi.log_prob(action)

                # Step with Auto-Reset (Vectorized)
                rng, rng_step = jax.random.split(rng)
                rng_step = jax.random.split(rng_step, config["NUM_ENVS"])
                
                step_fn = jax.vmap(step_env_auto_reset, in_axes=(0, 0, 0))
                next_obs, next_env_state, reward, done = step_fn(rng_step, env_state, action)

                transition = Transition(
                    done, action, value, reward, log_prob, 
                    card_obs, act_obs, mask_obs,
                    chips_player, chips_opponent, pid
                )
                
                # Update runner state for next step
                return (train_state, next_env_state, next_obs, rng), transition

            # Run the rollout scan
            # runner_state here is: (train_state, env_state, current_obs, rng)
            runner_state, traj_batch = jax.lax.scan(
                _env_rollout, runner_state, None, config["NUM_STEPS"]
            )

            # --- 2. GAE CALCULATION ---
            # Unpack the runner state from the end of the rollout
            # This gives us the state/obs for the *start* of the next rollout (bootstrap value)
            train_state, next_env_state, next_obs, rng = runner_state
            
            # Calculate Bootstrap Value (Next Value)
            _, last_val = network.apply(train_state.params, next_obs)
            last_pid = next_env_state.game_state.current_player 

            def _calculate_gae(traj_batch, last_val, last_pid):
                def _get_advantages(carry, transition):
                    gae, next_value, next_pid = carry
                    done, value, reward = transition.done, transition.value, transition.reward
                    curr_pid = transition.player_id
                    
                    # Zero-Sum Value Flip logic
                    # If the player changed between steps, the value function perspective flips.
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

            # --- 3. PPO UPDATE EPOCHS ---
            def _update_epoch(update_state, unused):
                def _update_minibatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info
                    
                    def _loss_fn(params, traj_batch, gae, targets):
                        batch_obs = (traj_batch.obs_card, traj_batch.obs_act, traj_batch.obs_mask)
                        pi, value = network.apply(params, batch_obs)
                        log_prob = pi.log_prob(traj_batch.action)
                        
                        # --- CORRECTED TRINAL-CLIP ACTOR LOSS ---
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        epsilon = config["CLIP_EPS"]
                        delta1 = config["TRINAL_DELTA1"]
                        
                        surr1 = ratio * gae
                        
                        # Trinal Clip Logic
                        lower_bound = 1.0 - epsilon
                        upper_bound = jnp.where(gae < 0, delta1, 1.0 + epsilon)
                        
                        ratio_clipped = jnp.clip(ratio, lower_bound, upper_bound)
                        surr2 = ratio_clipped * gae
                        
                        loss_actor = -jnp.minimum(surr1, surr2).mean()
                        
                        # --- CRITIC LOSS ---
                        delta2 = traj_batch.chips_player
                        delta3 = traj_batch.chips_opponent
                        targets_clipped = jnp.clip(targets, -delta2, delta3)
                        loss_critic = jnp.square(targets_clipped - value).mean()
                        
                        entropy = pi.entropy().mean()
                        total_loss = loss_actor + 0.5 * loss_critic - config["ENT_COEF"] * entropy
                        return total_loss, (loss_actor, loss_critic, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    (loss, (l_a, l_c, ent)), grads = grad_fn(train_state.params, traj_batch, advantages, targets)
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, (loss, l_a, l_c, ent)

                # Batch Handling (Flatten & Shuffle)
                batch_size = config["NUM_ENVS"] * config["NUM_STEPS"]
                permutation = jax.random.permutation(update_state[1], batch_size)
                
                traj_flat = jax.tree_map(lambda x: x.reshape((batch_size, *x.shape[2:])), traj_batch)
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
                    batch = (jax.tree_map(slice_fn, shuffled_traj), slice_fn(shuffled_adv), slice_fn(shuffled_tgt))
                    return _update_minibatch(train_state, batch)

                train_state, metrics = jax.lax.scan(
                    scan_batch, update_state[0], jnp.arange(config["NUM_MINIBATCHES"])
                )
                return (train_state, update_state[1]), metrics

            rng, _rng = jax.random.split(rng)
            (train_state, _), metrics = jax.lax.scan(
                _update_epoch, (train_state, _rng), None, config["UPDATE_EPOCHS"]
            )
            
            # Re-pack the runner state for the next outer loop iteration
            # We must pass 'next_obs' so the next iteration starts from where we left off
            return (train_state, next_env_state, next_obs, rng), metrics

        runner_state = (train_state, env_states, obsv, rng)
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metrics}

    return train

if __name__ == "__main__":
    print("Initializing Training...")
    train_fn = make_train(CONFIG)
    rng = jax.random.PRNGKey(42)
    output = train_fn(rng)
    print("Training Complete.")