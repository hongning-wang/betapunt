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
    action_tensor: chex.Array   # (HIST_T, 4, ACTION_DIM): action history tensor
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

        # Card branch: (B,8,4,13) -> (B,4,13,8)
        card_x = jnp.transpose(card_obs, (0, 2, 3, 1))
        card_h = nn.relu(nn.Conv(64, (3, 3), padding="SAME")(card_x))
        card_h = ResBlock(64)(card_h)
        card_h = ResBlock(64)(card_h)
        card_h = ResBlock(64)(card_h)
        card_flat = card_h.reshape((card_h.shape[0], -1))  # (B, 4*13*64)

        # Action branch: (B,26,4,13) -> (B,4,13,26)
        act_x = jnp.transpose(act_obs, (0, 2, 3, 1))
        act_h = nn.relu(nn.Conv(64, (3, 3), padding="SAME")(act_x))
        act_h = ResBlock(64)(act_h)
        act_h = ResBlock(64)(act_h)
        act_h = ResBlock(64)(act_h)
        act_flat = act_h.reshape((act_h.shape[0], -1))  # (B, 4*13*64)

        fused = jnp.concatenate([card_flat, act_flat], axis=-1)
        x = nn.relu(nn.Dense(1024)(fused))
        x = nn.relu(nn.Dense(512)(x))

        logits = nn.Dense(self.action_dim)(x)
        logits = jnp.where(mask, logits, -1e9)
        pi = distrax.Categorical(logits=logits)

        value = jnp.squeeze(nn.Dense(1)(x), axis=-1)
        return pi, value

# -------------------------
# 4) K-BEST SELF-PLAY STRUCTURES
# -------------------------
class Transition(NamedTuple):
    """Transition for main agent only (we only train the main agent)."""
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
    is_main_agent_turn: jnp.ndarray  # Mask: 1 if main agent acted, 0 otherwise
    opponent_idx: jnp.ndarray        # Which opponent was played against
    main_agent_reward: jnp.ndarray   # Terminal reward for main agent (0 if not done)


@chex.dataclass(frozen=True)
class AgentPool:
    """Pool of historical agent parameters with ELO ratings."""
    params_pool: chex.ArrayTree    # Stacked params: tree of (POOL_SIZE, ...)
    elos: chex.Array               # (POOL_SIZE,) ELO ratings
    active_mask: chex.Array        # (POOL_SIZE,) bool - which slots are filled
    num_agents: int                # Number of active agents
    main_agent_elo: float          # Current main agent's ELO rating


@chex.dataclass(frozen=True) 
class MatchState:
    """State for a single environment running a match."""
    env_state: EnvState
    main_is_p0: bool               # Is main agent playing as P0?


def make_train(config):
    envw = AlphaHoldemWrapper()
    num_updates = int(config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"])
    
    def init_agent_pool(params):
        """Initialize pool with just the initial agent."""
        pool_params = jax.tree_map(
            lambda p: jnp.stack([p] * config["POOL_SIZE"], axis=0),
            params
        )
        return AgentPool(
            params_pool=pool_params,
            elos=jnp.full(config["POOL_SIZE"], config["INITIAL_ELO"]),
            active_mask=jnp.array([True] + [False] * (config["POOL_SIZE"] - 1)),
            num_agents=1,
            main_agent_elo=config["INITIAL_ELO"],
        )
    
    def add_to_pool(pool: AgentPool, new_params, new_elo: float) -> AgentPool:
        """Add new agent to pool, replacing weakest if full."""
        num_active = pool.num_agents
        
        def add_at_slot(slot_idx):
            new_pool_params = jax.tree_map(
                lambda pool_p, new_p: pool_p.at[slot_idx].set(new_p),
                pool.params_pool, new_params
            )
            new_elos = pool.elos.at[slot_idx].set(new_elo)
            new_mask = pool.active_mask.at[slot_idx].set(True)
            return AgentPool(
                params_pool=new_pool_params,
                elos=new_elos,
                active_mask=new_mask,
                num_agents=jnp.minimum(num_active + 1, config["POOL_SIZE"]),
                main_agent_elo=pool.main_agent_elo,  # Preserve main agent ELO
            )
        
        def replace_weakest():
            # Find weakest active agent
            masked_elos = jnp.where(pool.active_mask, pool.elos, jnp.inf)
            weakest_idx = jnp.argmin(masked_elos)
            return add_at_slot(weakest_idx)
        
        def add_new():
            return add_at_slot(num_active)
        
        return jax.lax.cond(
            num_active >= config["POOL_SIZE"],
            replace_weakest,
            add_new
        )
    
    def sample_opponent_idx(rng, pool: AgentPool) -> int:
        """Sample opponent from K-best agents in pool."""
        # Get indices of K-best active agents
        masked_elos = jnp.where(pool.active_mask, pool.elos, -jnp.inf)
        
        # Get top K indices
        k = jnp.minimum(config["K_BEST"], pool.num_agents)
        top_k_indices = jnp.argsort(masked_elos)[::-1][:config["K_BEST"]]
        
        # Create sampling mask for top K
        valid_mask = jnp.arange(config["K_BEST"]) < k
        
        # Sample uniformly from valid top-K
        rng, rng_sample = jax.random.split(rng)
        sample_idx = jax.random.randint(rng_sample, (), 0, config["K_BEST"])
        sample_idx = jnp.minimum(sample_idx, k - 1)
        
        return top_k_indices[sample_idx]
    
    def get_opponent_params(pool: AgentPool, idx: int):
        """Extract params for a specific opponent from pool."""
        return jax.tree_map(lambda p: p[idx], pool.params_pool)
    
    def update_elo(winner_elo: float, loser_elo: float, is_draw: bool = False):
        """Compute new ELO ratings after a match."""
        expected_winner = 1.0 / (1.0 + 10.0 ** ((loser_elo - winner_elo) / 400.0))
        expected_loser = 1.0 - expected_winner
        
        k = config["ELO_K"]
        
        # Score: 1 for win, 0.5 for draw, 0 for loss
        winner_score = jnp.where(is_draw, 0.5, 1.0)
        loser_score = jnp.where(is_draw, 0.5, 0.0)
        
        new_winner_elo = winner_elo + k * (winner_score - expected_winner)
        new_loser_elo = loser_elo + k * (loser_score - expected_loser)
        
        return new_winner_elo, new_loser_elo

    def train(rng):
        # Initialize network
        net = AlphaHoldemNetwork(action_dim=envw.action_space_n())
        rng, rng_init = jax.random.split(rng)
        dummy_obs = (
            jnp.zeros((1, 8, 4, 13), dtype=jnp.float32),
            jnp.zeros((1, HIST_T, 4, ACTION_DIM), dtype=jnp.float32),
            jnp.ones((1, ACTION_DIM), dtype=bool),
        )
        params = net.init(rng_init, dummy_obs)

        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(config["LR"], eps=1e-5),
        )
        train_state = TrainState.create(apply_fn=net.apply, params=params, tx=tx)
        
        # Initialize agent pool
        pool = init_agent_pool(params)
        
        # Reset environments and assign opponents
        rng, rng_reset = jax.random.split(rng)
        reset_keys = jax.random.split(rng_reset, config["NUM_ENVS"])
        obsv, env_states = jax.vmap(envw.reset)(reset_keys)
        
        # Randomly assign main agent to P0 or P1 for each env
        rng, rng_assign = jax.random.split(rng)
        main_is_p0 = jax.random.bernoulli(rng_assign, 0.5, (config["NUM_ENVS"],))
        
        # Sample opponent indices for each env
        rng, rng_opp = jax.random.split(rng)
        opp_keys = jax.random.split(rng_opp, config["NUM_ENVS"])
        opponent_indices = jax.vmap(lambda k: sample_opponent_idx(k, pool))(opp_keys)

        def step_env_with_opponent(rng, env_state, main_params, opp_params, main_is_p0_i, opp_idx):
            """
            Step environment where main agent and opponent take turns.
            Returns transition only for main agent's moves.
            """
            gs = env_state.game_state
            current_player = gs.current_player
            
            # Determine if it's main agent's turn
            is_main_turn = jnp.where(main_is_p0_i, current_player == 0, current_player == 1)
            
            # Get observation for current player
            obs = envw._get_obs(env_state, gs)
            card_obs, act_obs, mask = obs
            
            # Batch the obs for network (add batch dim)
            batched_obs = (card_obs[None], act_obs[None], mask[None])
            
            # Get action from appropriate agent
            def main_agent_action():
                pi, value = net.apply(main_params, batched_obs)
                return pi, value
            
            def opponent_action():
                pi, value = net.apply(opp_params, batched_obs)
                return pi, value
            
            pi, value = jax.lax.cond(is_main_turn, main_agent_action, opponent_action)
            
            rng, rng_act = jax.random.split(rng)
            action = pi.sample(seed=rng_act)
            action = action[0]  # Remove batch dim
            log_prob = pi.log_prob(action[None])[0]
            value = value[0]
            
            # Step environment
            rng, rng_step = jax.random.split(rng)
            next_obs, next_env_state, reward_p0, done = envw.step(rng_step, env_state, action)
            
            # Convert reward to main agent's perspective
            reward_main = jnp.where(main_is_p0_i, reward_p0, -reward_p0)
            
            # Terminal reward for ELO update (only non-zero when done)
            terminal_reward_main = jnp.where(done, reward_main, 0.0)
            
            # Chips for value clipping (from main agent's perspective)
            chips_p0 = STACK_SIZE - gs.p0_stack
            chips_p1 = STACK_SIZE - gs.p1_stack
            chips_main = jnp.where(main_is_p0_i, chips_p0, chips_p1)
            chips_opp = jnp.where(main_is_p0_i, chips_p1, chips_p0)
            
            return next_obs, next_env_state, Transition(
                done=done,
                action=action,
                value=value,
                reward=reward_main,
                log_prob=log_prob,
                obs_card=card_obs,
                obs_act=act_obs,
                obs_mask=mask,
                chips_player=chips_main,
                chips_opponent=chips_opp,
                is_main_agent_turn=is_main_turn.astype(jnp.float32),
                opponent_idx=opp_idx,
                main_agent_reward=terminal_reward_main,
            ), done, reward_main, rng

        def step_env_auto_reset(rng, env_state, main_params, opp_params, main_is_p0_i, opp_idx, pool):
            """Step with auto-reset and re-sample opponent on reset."""
            next_obs, next_env_state, transition, done, reward_main, rng = step_env_with_opponent(
                rng, env_state, main_params, opp_params, main_is_p0_i, opp_idx
            )
            
            # On done, reset and re-sample
            rng, rng_reset = jax.random.split(rng)
            reset_obs, reset_env_state = envw.reset(rng_reset)
            
            rng, rng_assign = jax.random.split(rng)
            new_main_is_p0 = jax.random.bernoulli(rng_assign, 0.5)
            
            rng, rng_opp = jax.random.split(rng)
            new_opp_idx = sample_opponent_idx(rng_opp, pool)
            
            final_env_state = jax.tree_map(
                lambda r, n: jnp.where(done, r, n), reset_env_state, next_env_state
            )
            final_obs = jax.tree_map(
                lambda r, n: jnp.where(done, r, n), reset_obs, next_obs
            )
            final_main_is_p0 = jnp.where(done, new_main_is_p0, main_is_p0_i)
            final_opp_idx = jnp.where(done, new_opp_idx, opp_idx)
            
            return final_obs, final_env_state, transition, final_main_is_p0, final_opp_idx, rng

        def _update_step(runner_state, update_idx):
            train_state, env_states, last_obs, main_is_p0, opponent_indices, pool, rng = runner_state
            
            # Get opponent params for each env
            opp_params_per_env = jax.vmap(lambda idx: get_opponent_params(pool, idx))(opponent_indices)
            
            def _rollout_step(carry, _):
                ts, env_states, last_obs, main_is_p0, opp_indices, pool, rng = carry
                
                rng, rng_step = jax.random.split(rng)
                step_keys = jax.random.split(rng_step, config["NUM_ENVS"])
                
                # Get current opponent params
                opp_params = jax.vmap(lambda idx: get_opponent_params(pool, idx))(opp_indices)
                
                # Vectorized step
                def single_env_step(key, env_st, main_p0, opp_idx):
                    opp_p = get_opponent_params(pool, opp_idx)
                    return step_env_auto_reset(
                        key, env_st, ts.params, opp_p, main_p0, opp_idx, pool
                    )
                
                results = jax.vmap(single_env_step)(
                    step_keys, env_states, main_is_p0, opp_indices
                )
                next_obs, next_env_states, transitions, new_main_is_p0, new_opp_indices, _ = results
                
                return (ts, next_env_states, next_obs, new_main_is_p0, new_opp_indices, pool, rng), transitions
            
            # Collect rollout
            carry = (train_state, env_states, last_obs, main_is_p0, opponent_indices, pool, rng)
            carry, traj = jax.lax.scan(_rollout_step, carry, None, config["NUM_STEPS"])
            train_state, env_states, last_obs, main_is_p0, opponent_indices, pool, rng = carry
            
            # --- Update ELO ratings based on game outcomes ---
            def update_pool_elos(pool, traj):
                """Update ELO ratings for all completed games in trajectory."""
                # Flatten trajectory: (NUM_STEPS, NUM_ENVS) -> (NUM_STEPS * NUM_ENVS,)
                dones = traj.done.reshape(-1)
                rewards = traj.main_agent_reward.reshape(-1)
                opp_indices = traj.opponent_idx.reshape(-1).astype(jnp.int32)
                
                # For each completed game, update ELO
                def update_single_game(carry, game_data):
                    pool_elos, main_elo = carry
                    done, reward, opp_idx = game_data
                    
                    opp_elo = pool_elos[opp_idx]
                    
                    # Calculate expected scores
                    expected_main = 1.0 / (1.0 + 10.0 ** ((opp_elo - main_elo) / 400.0))
                    expected_opp = 1.0 - expected_main
                    
                    # Actual scores based on reward
                    # reward > 0: main won, reward < 0: opp won, reward == 0: draw
                    main_score = jnp.where(reward > 0, 1.0, jnp.where(reward == 0, 0.5, 0.0))
                    opp_score = 1.0 - main_score
                    
                    # New ELOs
                    new_main_elo = main_elo + config["ELO_K"] * (main_score - expected_main)
                    new_opp_elo = opp_elo + config["ELO_K"] * (opp_score - expected_opp)
                    
                    # Only update if game actually ended
                    final_main_elo = jnp.where(done, new_main_elo, main_elo)
                    final_opp_elo = jnp.where(done, new_opp_elo, opp_elo)
                    
                    # Update pool ELOs for this opponent
                    new_pool_elos = pool_elos.at[opp_idx].set(final_opp_elo)
                    
                    return (new_pool_elos, final_main_elo), None
                
                # Stack game data for scanning
                game_data_stack = (dones, rewards, opp_indices)
                
                (new_pool_elos, new_main_elo), _ = jax.lax.scan(
                    update_single_game,
                    (pool.elos, pool.main_agent_elo),
                    game_data_stack,
                )
                
                return pool.replace(elos=new_pool_elos, main_agent_elo=new_main_elo)
            
            pool = update_pool_elos(pool, traj)
            
            # Compute advantages only for main agent turns
            # Get last value for main agent
            card_obs, act_obs, mask = last_obs
            batched_last_obs = (card_obs, act_obs, mask)
            _, last_val = net.apply(train_state.params, batched_last_obs)
            
            # GAE computation with masking for main agent turns
            def _gae_scan(carry, inputs):
                gae, next_value = carry
                tr = inputs
                
                done = tr.done
                value = tr.value
                reward = tr.reward
                is_main = tr.is_main_agent_turn
                
                # Standard GAE
                delta = reward + config["GAMMA"] * next_value * (1.0 - done) - value
                gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1.0 - done) * gae
                
                # Mask: only accumulate for main agent turns
                gae = gae * is_main
                
                return (gae, value), gae
            
            (_, _), advantages = jax.lax.scan(
                _gae_scan,
                (jnp.zeros_like(last_val), last_val),
                traj,
                reverse=True,
                unroll=16,
            )
            targets = advantages + traj.value
            
            # Advantage normalization (only over main agent transitions)
            main_mask = traj.is_main_agent_turn
            if config.get("ADV_NORM", True):
                main_advs = advantages * main_mask
                count = jnp.sum(main_mask) + 1e-8
                adv_mean = jnp.sum(main_advs) / count
                adv_var = jnp.sum(main_mask * jnp.square(advantages - adv_mean)) / count
                adv_std = jnp.sqrt(adv_var) + 1e-8
                advantages = (advantages - adv_mean) / adv_std
            
            advantages = jax.lax.stop_gradient(advantages)
            targets = jax.lax.stop_gradient(targets)
            main_mask = jax.lax.stop_gradient(main_mask)
            
            # PPO update epochs
            def _update_epoch(update_state, _):
                ts, rng = update_state
                
                batch_size = config["NUM_ENVS"] * config["NUM_STEPS"]
                rng, rng_perm = jax.random.split(rng)
                perm = jax.random.permutation(rng_perm, batch_size)
                
                traj_flat = jax.tree_map(lambda x: x.reshape((batch_size, *x.shape[2:])), traj)
                adv_flat = advantages.reshape((batch_size,))
                tgt_flat = targets.reshape((batch_size,))
                mask_flat = main_mask.reshape((batch_size,))
                
                traj_shuf = jax.tree_map(lambda x: x[perm], traj_flat)
                adv_shuf = adv_flat[perm]
                tgt_shuf = tgt_flat[perm]
                mask_shuf = mask_flat[perm]
                
                mb_size = batch_size // config["NUM_MINIBATCHES"]
                
                def _mb_update(ts, i):
                    s = i * mb_size
                    e = s + mb_size
                    mb_tr = jax.tree_map(lambda x: x[s:e], traj_shuf)
                    mb_adv = adv_shuf[s:e]
                    mb_tgt = tgt_shuf[s:e]
                    mb_mask = mask_shuf[s:e]
                    
                    def _loss_fn(params):
                        batch_obs = (mb_tr.obs_card, mb_tr.obs_act, mb_tr.obs_mask)
                        pi, v = net.apply(params, batch_obs)
                        
                        new_logp = pi.log_prob(mb_tr.action)
                        ratio = jnp.exp(new_logp - mb_tr.log_prob)
                        
                        eps = config["CLIP_EPS"]
                        d1 = config["TRINAL_DELTA1"]
                        
                        # Trinal-Clip PPO
                        r_clip = jnp.clip(ratio, 1.0 - eps, 1.0 + eps)
                        r_tcp = jnp.minimum(jnp.maximum(ratio, r_clip), d1)
                        
                        # Masked actor loss (only main agent transitions)
                        actor_terms = r_tcp * mb_adv * mb_mask
                        loss_actor = -jnp.sum(actor_terms) / (jnp.sum(mb_mask) + 1e-8)
                        
                        # Critic loss with dynamic target clipping
                        d2 = mb_tr.chips_player
                        d3 = mb_tr.chips_opponent
                        tgt_clip = jnp.clip(mb_tgt, -d2, d3)
                        critic_terms = jnp.square(tgt_clip - v) * mb_mask
                        loss_critic = jnp.sum(critic_terms) / (jnp.sum(mb_mask) + 1e-8)
                        
                        # Entropy (also masked)
                        ent_terms = pi.entropy() * mb_mask
                        ent = jnp.sum(ent_terms) / (jnp.sum(mb_mask) + 1e-8)
                        
                        total = loss_actor + 0.5 * loss_critic - config["ENT_COEF"] * ent
                        return total, (loss_actor, loss_critic, ent)
                    
                    (loss, aux), grads = jax.value_and_grad(_loss_fn, has_aux=True)(ts.params)
                    ts = ts.apply_gradients(grads=grads)
                    return ts, (loss, aux)
                
                ts, metrics = jax.lax.scan(_mb_update, ts, jnp.arange(config["NUM_MINIBATCHES"]))
                return (ts, rng), metrics
            
            rng, rng_epochs = jax.random.split(rng)
            (train_state, _), epoch_metrics = jax.lax.scan(
                _update_epoch,
                (train_state, rng_epochs),
                None,
                config["UPDATE_EPOCHS"],
            )
            
            # Periodically add current agent to pool with its current ELO
            def add_snapshot(pool):
                # Snapshot the current main agent with its ELO
                return add_to_pool(pool, train_state.params, pool.main_agent_elo)
            
            def keep_pool(pool):
                return pool
            
            should_snapshot = (update_idx + 1) % config["SNAPSHOT_INTERVAL"] == 0
            pool = jax.lax.cond(should_snapshot, add_snapshot, keep_pool, pool)
            
            new_runner_state = (train_state, env_states, last_obs, main_is_p0, opponent_indices, pool, rng)
            return new_runner_state, epoch_metrics
        
        # Initial runner state
        runner_state = (train_state, env_states, obsv, main_is_p0, opponent_indices, pool, rng)
        
        # Training loop
        runner_state, all_metrics = jax.lax.scan(
            _update_step,
            runner_state,
            jnp.arange(num_updates),
        )
        
        return {"runner_state": runner_state, "metrics": all_metrics}
    
    return train


# -------------------------
# 5) MAIN
# -------------------------
if __name__ == "__main__":
    print("Initializing K-Best Self-Play Training...")
    print(f"Pool Size: {CONFIG['POOL_SIZE']}, K-Best: {CONFIG['K_BEST']}")
    train_fn = make_train(CONFIG)
    rng = jax.random.PRNGKey(42)
    out = jax.jit(train_fn)(rng)
    print("Training complete.")
