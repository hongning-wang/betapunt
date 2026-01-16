# training.py  (FULL TRINAL-CLIP PPO, 13-action Toss-or-Hold’em)


import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import distrax
import chex

from flax.training.train_state import TrainState
from typing import NamedTuple

from game import TossHoldemEnv, State, STACK_SIZE

# -------------------------
# 1) CONFIG
# -------------------------
CONFIG = {
    "LR": 3e-4,
    "NUM_ENVS": 128,
    "NUM_STEPS": 128,
    "TOTAL_TIMESTEPS": 10_000_000,
    "UPDATE_EPOCHS": 4,
    "NUM_MINIBATCHES": 4,
    "GAMMA": 0.999,
    "GAE_LAMBDA": 0.95,
    "CLIP_EPS": 0.2,
    "ENT_COEF": 0.01,
    "MAX_GRAD_NORM": 0.5,
    "TRINAL_DELTA1": 3.0,
    "ADV_NORM": True,
    "DEBUG": False,
}

ACTION_DIM = 13
HIST_T = 26

# -------------------------
# 2) OBS ENCODING
# -------------------------
@chex.dataclass(frozen=True)
class EnvState:
    game_state: State
    aaction_tensor: chex.Array   # (HIST_T, 4, ACTION_DIM): action history tensor
    history_ptr: int
    time: int

def _scatter_ones_1d(base: jnp.ndarray, idx: jnp.ndarray) -> jnp.ndarray:
    # base: (52,), idx: (K,) int, may contain invalid -1
    valid = idx >= 0
    safe_idx = jnp.where(valid, idx, 0)
    vals = jnp.where(valid, 1.0, 0.0).astype(base.dtype)
    return base.at[safe_idx].max(vals)

def get_card_tensor(state: State, player_id: int) -> jnp.ndarray:
    """
    8-channel (4x13) card tensor.
    Channels:
      0 active hole cards (masked)
      1 flop
      2 hero discard
      3 villain discard
      4 turn
      5 river
      6 all public
      7 all known (active hole + public)
    Shape: (8, 4, 13)
    """
    hole = jnp.where(player_id == 0, state.p0_hole, state.p1_hole)
    active_mask = jnp.where(player_id == 0, state.p0_active_mask, state.p1_active_mask)
    hole_masked = jnp.where(active_mask, hole, -1)
    
    board = state.board  # length 6, -1 for unknown

    flop = board[0:2]
    oop_discard = board[2:3]
    dlr_discard = board[3:4]
    turn = board[4:5]
    river = board[5:6]
    
    # Determine hero/villain discards based on position
    is_dealer = jnp.where(player_id == 0, state.is_dealer_p0, ~state.is_dealer_p0)
    hero_discard = jnp.where(is_dealer, dlr_discard, oop_discard)
    villain_discard = jnp.where(is_dealer, oop_discard, dlr_discard)

    t = jnp.zeros((8, 52), dtype=jnp.float32)

    t = t.at[0].set(_scatter_ones_1d(t[0], hole))
    t = t.at[1].set(_scatter_ones_1d(t[1], flop))
    t = t.at[2].set(_scatter_ones_1d(t[2], hero_discard))
    t = t.at[3].set(_scatter_ones_1d(t[3], villain_discard))
    t = t.at[4].set(_scatter_ones_1d(t[4], turn))
    t = t.at[5].set(_scatter_ones_1d(t[5], river))
    t = t.at[6].set(_scatter_ones_1d(t[6], board))
    t = t.at[7].set(_scatter_ones_1d(t[7], jnp.concatenate([hole_masked, board])))

    return t.reshape((8, 4, 13))

def get_action_tensor(env_state: EnvState, observer_id: int) -> jnp.ndarray:
    """
    Returns action tensor with rows reordered for observer perspective.
    Row 0: hero (observer's actions)
    Row 1: villain (opponent's actions)
    Row 2: pot size (before action)
    Row 3: legal
    """
    t = env_state.action_tensor
    row_order = jnp.where(observer_id == 0,
                          jnp.array([0, 1, 2, 3]),
                          jnp.array([1, 0, 2, 3]))
    return t[:, row_order, :]

class AlphaHoldemWrapper:
    def __init__(self):
        self.env = TossHoldemEnv()

    def reset(self, key):
        gs = self.env.reset(key)
        
        # Get initial legal mask
        initial_legal = self.env.get_legal_actions(gs)
        
        # Initialize action tensor with first legal mask
        action_tensor = jnp.zeros((HIST_T, 4, ACTION_DIM), dtype=jnp.float32)
        action_tensor = action_tensor.at[0, 3, :].set(initial_legal.astype(jnp.float32))
        
        env_state = EnvState(
            game_state=gs,
            action_tensor=action_tensor,
            history_ptr=0,
            time=0,
        )
        obs = self._get_obs(env_state, gs)
        return obs, env_state
    
    def step(self, key, env_state: EnvState, action: jnp.ndarray):
        gs = env_state.game_state
        acting_pid = gs.current_player
        ptr = env_state.history_ptr

        # Update action tensor at current slot
        new_action_tensor = env_state.action_tensor
        new_action_tensor = new_action_tensor.at[ptr, acting_pid, action].set(1.0)
        new_action_tensor = new_action_tensor.at[ptr, 2, action].set(gs.pot)

        # Take action
        next_gs = self.env.step(gs, key, action)
        
        # Update history
        new_ptr = jnp.minimum(ptr + 1, HIST_T - 1)

        # Set legal mask for next decision point
        next_legal = self.env.get_legal_actions(next_gs)
        new_action_tensor = new_action_tensor.at[new_ptr, 3, :].set(next_legal.astype(jnp.float32))

        next_env_state = EnvState(
            game_state=next_gs,
            action_tensor=new_action_tensor,
            history_ptr=new_ptr,
            time=env_state.time + 1,
        )

        # Reward convention: terminal reward in P0 coordinates (profit)
        reward_p0 = next_gs.p0_stack - STACK_SIZE
        reward_p0 = jnp.where(next_gs.done, reward_p0, 0.0)

        obs = self._get_obs(next_env_state, next_gs)
        done = next_gs.done
        return obs, next_env_state, reward_p0, done

    def _get_obs(self, env_state: EnvState, gs: State):
        pid = gs.current_player
        card_feat = get_card_tensor(gs, pid)
        legal = self.env.get_legal_actions(gs)
        act_feat = get_action_tensor(env_state, observer_id=pid)
        return (card_feat, act_feat, legal)

    def action_space_n(self):
        return ACTION_DIM

# -------------------------
# 3) NETWORK
# -------------------------
class ResBlock(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x):
        r = x
        x = nn.Conv(self.features, (3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.Conv(self.features, (3, 3), padding="SAME")(x)
        return nn.relu(x + r)

class AlphaHoldemNetwork(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, inputs):
        card_obs, act_obs, mask = inputs

        # Card branch: (B,6,4,13) -> (B,4,13,6)
        card_x = jnp.transpose(card_obs, (0, 2, 3, 1))
        card_h = nn.relu(nn.Conv(64, (3, 3), padding="SAME")(card_x))
        card_h = ResBlock(64)(card_h)
        card_h = ResBlock(64)(card_h)
        card_h = ResBlock(64)(card_h)
        card_flat = card_h.reshape((card_h.shape[0], -1))

        # Action branch: (B,24,4,13) -> (B,24,13,4)
        act_x = jnp.transpose(act_obs, (0, 1, 3, 2))
        act_h = nn.relu(nn.Conv(64, (3, 3), padding="SAME")(act_x))
        act_h = ResBlock(64)(act_h)
        act_h = ResBlock(64)(act_h)
        act_h = ResBlock(64)(act_h)
        act_flat = act_h.reshape((act_h.shape[0], -1))

        fused = jnp.concatenate([card_flat, act_flat], axis=-1)
        x = nn.relu(nn.Dense(1024)(fused))
        x = nn.relu(nn.Dense(512)(x))

        logits = nn.Dense(self.action_dim)(x)
        logits = jnp.where(mask, logits, -1e9)
        pi = distrax.Categorical(logits=logits)

        value = jnp.squeeze(nn.Dense(1)(x), axis=-1)
        return pi, value

# -------------------------
# 4) TRAINING
# -------------------------
class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray     # reward in acting player's perspective
    log_prob: jnp.ndarray
    obs_card: jnp.ndarray
    obs_act: jnp.ndarray
    obs_mask: jnp.ndarray
    chips_player: jnp.ndarray
    chips_opponent: jnp.ndarray
    player_id: jnp.ndarray

def make_train(config):
    envw = AlphaHoldemWrapper()
    num_updates = int(config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"])

    def train(rng):
        # Reset vector envs
        rng, rng_reset = jax.random.split(rng)
        reset_keys = jax.random.split(rng_reset, config["NUM_ENVS"])
        obsv, env_states = jax.vmap(envw.reset)(reset_keys)

        # Init network
        net = AlphaHoldemNetwork(action_dim=envw.action_space_n())
        rng, rng_init = jax.random.split(rng)
        dummy_obs = (
            jnp.zeros((1, 6, 4, 13), dtype=jnp.float32),
            jnp.zeros((1, 24, 4, 13), dtype=jnp.float32),
            jnp.ones((1, 13), dtype=bool),
        )
        params = net.init(rng_init, dummy_obs)

        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(config["LR"], eps=1e-5),
        )
        ts = TrainState.create(apply_fn=net.apply, params=params, tx=tx)

        # Step env with auto-reset; IMPORTANT: use split RNG for reset
        def step_env_auto_reset(rng_i, env_state_i, action_i):
            obs, next_env_state, reward_p0, done = envw.step(rng_i, env_state_i, action_i)

            rng_i, rng_reset_i = jax.random.split(rng_i)
            reset_obs, reset_state = envw.reset(rng_reset_i)

            final_env_state = jax.tree_map(lambda r, n: jnp.where(done, r, n), reset_state, next_env_state)
            final_obs = jax.tree_map(lambda r, n: jnp.where(done, r, n), reset_obs, obs)

            return final_obs, final_env_state, reward_p0, done

        def _update_step(runner_state, _):
            # runner_state: (train_state, env_states, last_obs, rng)
            def _rollout_step(runner_state, _):
                ts, env_states, last_obs, rng = runner_state
                card_obs, act_obs, mask_obs = last_obs

                gs = env_states.game_state
                pid = gs.current_player  # acting player at this step

                # chips invested for dynamic value clipping (aligned with acting player's view)
                chips_p0 = STACK_SIZE - gs.p0_stack
                chips_p1 = STACK_SIZE - gs.p1_stack
                is_p0 = pid == 0
                chips_player = jnp.where(is_p0, chips_p0, chips_p1)
                chips_opponent = jnp.where(is_p0, chips_p1, chips_p0)

                # sample action
                rng, rng_act = jax.random.split(rng)
                pi, value = net.apply(ts.params, last_obs)
                action = pi.sample(seed=rng_act)
                log_prob = pi.log_prob(action)

                # step vector envs
                rng, rng_step = jax.random.split(rng)
                step_keys = jax.random.split(rng_step, config["NUM_ENVS"])
                next_obs, next_env_states, reward_p0, done = jax.vmap(step_env_auto_reset, in_axes=(0, 0, 0))(
                    step_keys, env_states, action
                )

                # convert terminal reward (P0 profit) to acting player's perspective
                reward = jnp.where(pid == 0, reward_p0, -reward_p0)

                tr = Transition(
                    done=done,
                    action=action,
                    value=value,
                    reward=reward,
                    log_prob=log_prob,
                    obs_card=card_obs,
                    obs_act=act_obs,
                    obs_mask=mask_obs,
                    chips_player=chips_player,
                    chips_opponent=chips_opponent,
                    player_id=pid,
                )
                return (ts, next_env_states, next_obs, rng), tr

            runner_state, traj = jax.lax.scan(_rollout_step, runner_state, None, config["NUM_STEPS"])

            ts, next_env_states, next_obs, rng = runner_state
            _, last_val = net.apply(ts.params, next_obs)
            last_pid = next_env_states.game_state.current_player

            # --- GAE with zero-sum value flip on player switch ---
            def _gae_scan(carry, tr: Transition):
                gae, next_value, next_pid = carry

                done = tr.done
                value = tr.value
                reward = tr.reward
                curr_pid = tr.player_id

                is_switch = next_pid != curr_pid
                correct_next_val = jnp.where(is_switch, -next_value, next_value)

                delta = reward + config["GAMMA"] * correct_next_val * (1.0 - done) - value
                gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1.0 - done) * gae
                return (gae, value, curr_pid), gae

            (_, _, _), advantages = jax.lax.scan(
                _gae_scan,
                (jnp.zeros_like(last_val), last_val, last_pid),
                traj,
                reverse=True,
                unroll=16,
            )
            targets = advantages + traj.value

            # Optional advantage normalization
            if config.get("ADV_NORM", True):
                adv_flat = advantages.reshape((-1,))
                adv_mean = adv_flat.mean()
                adv_std = adv_flat.std() + 1e-8
                advantages = (advantages - adv_mean) / adv_std

            advantages = jax.lax.stop_gradient(advantages)
            targets = jax.lax.stop_gradient(targets)

            # --- PPO update epochs ---
            def _update_epoch(update_state, _):
                ts, rng = update_state

                batch_size = config["NUM_ENVS"] * config["NUM_STEPS"]
                rng, rng_perm = jax.random.split(rng)
                perm = jax.random.permutation(rng_perm, batch_size)

                traj_flat = jax.tree_map(lambda x: x.reshape((batch_size, *x.shape[2:])), traj)
                adv_b = advantages.reshape((batch_size,))
                tgt_b = targets.reshape((batch_size,))

                traj_shuf = jax.tree_map(lambda x: x[perm], traj_flat)
                adv_shuf = adv_b[perm]
                tgt_shuf = tgt_b[perm]

                mb_size = batch_size // config["NUM_MINIBATCHES"]

                def _mb_update(ts, i):
                    s = i * mb_size
                    e = s + mb_size
                    mb_tr = jax.tree_map(lambda x: x[s:e], traj_shuf)
                    mb_adv = adv_shuf[s:e]
                    mb_tgt = tgt_shuf[s:e]

                    def _loss_fn(params):
                        batch_obs = (mb_tr.obs_card, mb_tr.obs_act, mb_tr.obs_mask)
                        pi, v = net.apply(params, batch_obs)

                        new_logp = pi.log_prob(mb_tr.action)
                        ratio = jnp.exp(new_logp - mb_tr.log_prob)

                        eps = config["CLIP_EPS"]
                        d1 = config["TRINAL_DELTA1"]

                        # FULL TRINAL-CLIP (paper Eq. 3)
                        r_clip = jnp.clip(ratio, 1.0 - eps, 1.0 + eps)
                        r_tcp = jnp.minimum(jnp.maximum(ratio, r_clip), d1)

                        loss_actor = -jnp.mean(r_tcp * mb_adv)

                        # Critic loss with dynamic target clipping
                        d2 = mb_tr.chips_player
                        d3 = mb_tr.chips_opponent
                        tgt_clip = jnp.clip(mb_tgt, -d2, d3)
                        loss_critic = jnp.mean(jnp.square(tgt_clip - v))

                        ent = jnp.mean(pi.entropy())
                        total = loss_actor + 0.5 * loss_critic - config["ENT_COEF"] * ent
                        return total, (loss_actor, loss_critic, ent)

                    (loss, (la, lc, ent)), grads = jax.value_and_grad(_loss_fn, has_aux=True)(ts.params)
                    ts = ts.apply_gradients(grads=grads)
                    return ts, (loss, la, lc, ent)

                ts, metrics = jax.lax.scan(_mb_update, ts, jnp.arange(config["NUM_MINIBATCHES"]))
                return (ts, rng), metrics

            rng, rng_epochs = jax.random.split(rng)
            (ts, _), metrics = jax.lax.scan(
                _update_epoch,
                (ts, rng_epochs),
                None,
                config["UPDATE_EPOCHS"],
            )

            return (ts, next_env_states, next_obs, rng), metrics

        runner_state = (ts, env_states, obsv, rng)
        runner_state, metrics = jax.lax.scan(_update_step, runner_state, None, num_updates)
        return {"runner_state": runner_state, "metrics": metrics}

    return train

# -------------------------
# 5) MAIN
# -------------------------
if __name__ == "__main__":
    print("Initializing Training (FULL Trinal-Clip PPO, 13-action)...")
    train_fn = make_train(CONFIG)
    rng = jax.random.PRNGKey(42)
    out = jax.jit(train_fn)(rng)
    print("Training complete.")
