"""Transformer-based IMPALA V-trace training for Toss-or-Hold'em (JAX/Flax).

This training script implements IMPALA-style training:
  1) Transformer-based card/action encoding
  2) Pure V-trace off-policy correction computed at TRAINING TIME
  3) Short replay window storing complete unrolls for proper V-trace computation
  4) K-Best self-play league training with ELO-based opponent selection

Key architectural choices:
  - Card tensor: (B, 9, 2) - [card_index, type_id] matching model.py CardEncoder
  - Action tensor: (B, 32, 7) - action history matching model.py ActionEncoder
  - Transformer encoders with cross-attention fusion
  - Critic trained on PRIVATE cards (same info as actor)

V-trace implementation:
  - Store complete unrolls (T, E, ...) with behavior policy log probs (logp_mu)
  - At training time: recompute logp_pi, compute log_rhos = logp_pi - logp_mu
  - Run V-trace backward pass to get value targets and advantages
  - Policy loss: -sum(pg_rho * logp_pi * advantages)
  - Importance weights masked to 0 on opponent steps
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import os
import shutil
from typing import Any, Dict, Tuple

import chex
import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
import orbax.checkpoint
from flax.training import orbax_utils

# Import the transformer model
from model import (
    TossHoldemModel,
    NUM_ACTIONS,
    CARD_SEQ_LEN,
    ACTION_SEQ_LEN,
    CARD_INPUT_DIM,
    ACTION_INPUT_DIM,
)

# --- Environment ---
from game import (
    TossHoldemEnv,
    State,
    STACK_SIZE,
)


# =============================================================================
# 1) Configuration
# =============================================================================


@dataclass(frozen=True)
class Config:
    """IMPALA-style V-trace configuration (no PPO).

    This is a research-grade setup using:
    - Pure V-trace for off-policy correction (no PPO clipping)
    - Short replay window for near-on-policy data
    - League training via K-best self-play pool
    """

    # Parallel sampling
    num_envs: int = 2_048 # parallel environments
    rollout_len: int = 32

    # V-trace hyperparameters (IMPALA-style)
    gamma: float = 0.99
    rho_bar: float = 1.0   # Importance weight clip for value target (ρ̂)
    c_bar: float = 1.0     # Trace cutting coefficient (c)
    pg_rho_bar: float = 1.0  # Policy gradient importance weight clip
    vf_coef: float = 0.5

    # Entropy coefficient with schedule
    ent_coef_init: float = 0.02
    ent_coef_final: float = 0.001
    ent_decay_steps: int = 15_000

    # Optimization with schedule
    lr_init: float = 1e-4
    lr_peak: float = 3e-4
    lr_final: float = 1e-5
    warmup_updates: int = 500
    max_grad_norm: float = 0.5  # Tighter for V-trace stability

    # Off-policy replay (short window for IMPALA-style)
    # Note: Actual capacity is max(16, replay_capacity // (rollout_len * num_envs)) unrolls
    # With default settings: max(16, 500k // 262k) = 16 unrolls ≈ 4.2M transitions
    replay_capacity: int = 200_000  # Target transition capacity (see note above)
    num_minibatches: int = 1 # More minibatches for more gradient steps
    update_epochs: int = 1  # Multiple epochs safe with V-trace importance weighting
    warmup_steps: int = 100_000  # Shorter warmup (in transitions)

    # Short replay window (key for V-trace) - only sample from recent data
    recent_window_transitions: int = 2 * num_envs * rollout_len # Sample only from recent N transitions (0 = no limit)

    # League training (K-best self-play)
    pool_max: int = 6
    eval_hands: int = 512
    add_to_pool_every_updates: int = 100
    elo_k: float = 32.0
    elo_tau: float = 2 * eval_hands  # Temperature for EV-soft sigmoid scoring (chip units)

    # Runtime
    seed: int = 0
    total_updates: int = 20_000
    block_size: int = 100  # Number of collect+update steps per scan block (reduces host-device sync)

    # Checkpointing
    checkpoint_dir: str = "/content/drive/MyDrive/checkpoints"
    checkpoint_every: int = 1000
    log_every: int = 2 * block_size


CFG = Config()

# Shared env instance (avoid re-instantiating inside jit)
ENV = TossHoldemEnv()


# =============================================================================
# 2) State Representation (Transformer-compatible)
# =============================================================================


STREET_PREFLOP = 0
STREET_DISCARD = 1
STREET_FLOP = 2
STREET_TURN = 3
STREET_RIVER = 4


@jax.jit
def phase_to_street(phase: chex.Array) -> chex.Array:
    """Map engine phases to 5 street ids used in action history."""
    return jax.lax.switch(
        phase,
        [
            lambda: jnp.int32(STREET_PREFLOP),  # PHASE_PREFLOP
            lambda: jnp.int32(STREET_DISCARD),  # PHASE_DISCARD_OOP
            lambda: jnp.int32(STREET_DISCARD),  # PHASE_DISCARD_DLR
            lambda: jnp.int32(STREET_FLOP),     # PHASE_FLOP
            lambda: jnp.int32(STREET_TURN),     # PHASE_TURN
            lambda: jnp.int32(STREET_RIVER),    # PHASE_RIVER
            lambda: jnp.int32(STREET_RIVER),    # PHASE_SHOWDOWN (unused)
        ],
    )


@jax.jit
def make_action_token(state: State, action: chex.Array, time_id: chex.Array) -> chex.Array:
    """Build a 7-field action token from the current state and action.

    time_id is a monotonically increasing index (global action count).
    """
    is_p0 = state.current_player == 0

    cur_stack = jnp.where(is_p0, state.p0_stack, state.p1_stack).astype(jnp.float32)
    cur_bet = jnp.where(is_p0, state.p0_bet, state.p1_bet).astype(jnp.float32)
    opp_stack = jnp.where(is_p0, state.p1_stack, state.p0_stack).astype(jnp.float32)
    opp_bet = jnp.where(is_p0, state.p1_bet, state.p0_bet).astype(jnp.float32)

    to_call = (opp_bet - cur_bet).astype(jnp.float32)
    # pot_base: pot after calling, matches game.py's raise sizing definition
    pot_base = (state.pot + state.p0_bet + state.p1_bet + to_call).astype(jnp.float32)

    eff_cap = jnp.minimum(cur_stack, opp_stack + to_call)

    street_id = phase_to_street(state.phase).astype(jnp.float32)

    token = jnp.array(
        [
            state.current_player.astype(jnp.float32),
            action.astype(jnp.float32),
            street_id,
            time_id.astype(jnp.float32),
            to_call,
            pot_base,
            eff_cap,
        ],
        dtype=jnp.float32,
    )
    return token



@chex.dataclass(frozen=True)
class AuxState:
    """Extra per-env state to build action history tensors.

    Action tensor shape: (ACTION_SEQ_LEN, ACTION_INPUT_DIM) = (32, 7)
    Each action entry contains (matching model.py ActionEncoder):
        [0] actor_id          int {0,1}
        [1] event_type        int {0..12}
        [2] street_id         int {0..4}
        [3] time_id           int (monotonic action counter)
        [4] to_call           float
        [5] pot_base          float (pot after calling, for raise sizing)
        [6] effective_stack   float
    """
    action_tensor: chex.Array  # (ACTION_SEQ_LEN, ACTION_INPUT_DIM) = (32, 7)
    action_ptr: chex.Array     # Current position in action history


@jax.jit
def aux_reset() -> AuxState:
    return AuxState(
        action_tensor=jnp.zeros((ACTION_SEQ_LEN, ACTION_INPUT_DIM), dtype=jnp.float32),
        action_ptr=jnp.int32(0),
    )


@jax.jit
def aux_record_action(
    aux: AuxState,
    state: State,
    action: chex.Array,
) -> AuxState:
    """Record an action into the action history tensor.

    Uses make_action_token to build the 7-feature action entry matching model.py:
      [0]: actor_id (0 or 1)
      [1]: event_type (0-12)
      [2]: street_id (0-4)
      [3]: time_id (monotonic action counter)
      [4]: to_call
      [5]: pot_base
      [6]: effective_stack
    """
    # Build the 7-feature action token using existing make_action_token
    action_entry = make_action_token(state, action, aux.action_ptr)

    # Write to action tensor (circular buffer behavior if > ACTION_SEQ_LEN actions)
    write_idx = aux.action_ptr % ACTION_SEQ_LEN
    action_tensor = aux.action_tensor.at[write_idx].set(action_entry)

    return aux.replace(
        action_tensor=action_tensor,
        action_ptr=aux.action_ptr + 1,
    )


@jax.jit
def encode_cards(state: State, player_id: chex.Array) -> chex.Array:
    """Encode cards into (9, 2) tensor matching model.py CardEncoder format.

    Output format: (CARD_SEQ_LEN, CARD_INPUT_DIM) = (9, 2)
    Each card entry: [card_index, type_id]
      - card_index: 0-51 valid cards, -1 for undealt/invalid
      - type_id: 0-8 for slot type, -1 for invalid

    Card positions (matching model.py):
      0: heropriv0 (type_id=0)
      1: heropriv1 (type_id=1)
      2: heropriv2 (type_id=2)
      3: flop0 (type_id=3)
      4: flop1 (type_id=4)
      5: herodiscard (type_id=5)
      6: villaindiscard (type_id=6)
      7: turn (type_id=7)
      8: river (type_id=8)
    """
    is_p0 = player_id == 0

    # Get hole cards for the viewing player (hero)
    hero_hole = jnp.where(is_p0, state.p0_hole, state.p1_hole)  # (3,)
    hero_active = jnp.where(is_p0, state.p0_active_mask, state.p1_active_mask)  # (3,)

    # Determine discard indices based on dealer position
    # OOP discards to board[2], dealer discards to board[3]
    hero_is_dealer = jnp.where(is_p0, state.is_dealer_p0, jnp.logical_not(state.is_dealer_p0))
    hero_discard_idx = jnp.where(hero_is_dealer, 3, 2)
    villain_discard_idx = jnp.where(hero_is_dealer, 2, 3)

    # Build all card indices in one array (9 cards)
    # Positions: [hero0, hero1, hero2, flop0, flop1, hero_discard, villain_discard, turn, river]
    card_indices = jnp.array([
        jnp.where(hero_active[0], hero_hole[0], -1),
        jnp.where(hero_active[1], hero_hole[1], -1),
        jnp.where(hero_active[2], hero_hole[2], -1),
        state.board[0],  # flop0
        state.board[1],  # flop1
        state.board[hero_discard_idx],   # hero discard
        state.board[villain_discard_idx], # villain discard
        state.board[4],  # turn
        state.board[5],  # river
    ], dtype=jnp.int32)

    # Type IDs: valid cards get their position index, invalid get -1
    # For hero cards, validity is determined by hero_active
    # For board cards, validity is determined by card_index >= 0
    type_ids_raw = jnp.arange(CARD_SEQ_LEN, dtype=jnp.int32)
    valid_mask = jnp.array([
        hero_active[0],
        hero_active[1],
        hero_active[2],
        card_indices[3] >= 0,  # flop0
        card_indices[4] >= 0,  # flop1
        card_indices[5] >= 0,  # hero discard
        card_indices[6] >= 0,  # villain discard
        card_indices[7] >= 0,  # turn
        card_indices[8] >= 0,  # river
    ])
    type_ids = jnp.where(valid_mask, type_ids_raw, -1)

    # Stack into (9, 2) tensor
    card_tensor = jnp.stack([card_indices, type_ids], axis=-1)

    return card_tensor


@jax.jit
def encode_actions(aux: AuxState, player_id: chex.Array) -> Tuple[chex.Array, chex.Array]:
    """Encode action history into (ACTION_SEQ_LEN, ACTION_INPUT_DIM) tensor with ego-relative player IDs.

    Returns:
        action_tensor: (ACTION_SEQ_LEN, ACTION_INPUT_DIM) = (40, 7) action history
        action_mask: (ACTION_SEQ_LEN,) = (40,) boolean mask for valid actions (pointer-based)
    """
    # Re-center circular buffer so the most recent action is at the end
    # Use gather indices instead of jnp.roll to avoid full memory copy on TPU
    # roll(arr, shift=-ptr) is equivalent to arr[(arange + ptr) % N]
    ptr = aux.action_ptr % ACTION_SEQ_LEN
    gather_idx = (jnp.arange(ACTION_SEQ_LEN) + ptr) % ACTION_SEQ_LEN
    action_tensor = aux.action_tensor[gather_idx]

    # Make player IDs ego-relative (0 = self, 1 = opponent)
    # Original: player_id in action_tensor[..., 0]
    # If we're player 0: keep as-is (0=self, 1=opp)
    # If we're player 1: swap (0->1, 1->0)
    def swap_player_ids(tensor, pid):
        original_pid = tensor[..., 0]
        # Only swap if original_pid is 0 or 1 (valid)
        valid = (original_pid >= 0) & (original_pid <= 1)
        swapped = 1.0 - original_pid
        new_pid = jnp.where(pid == 0, original_pid, jnp.where(valid, swapped, original_pid))
        return tensor.at[..., 0].set(new_pid)

    action_tensor = swap_player_ids(action_tensor, player_id)

    # Create mask: pointer-based - first N actions are valid where N = min(action_ptr, ACTION_SEQ_LEN)
    # After rolling, valid actions are at the END of the tensor (positions ACTION_SEQ_LEN - n_valid onwards)
    n_valid = jnp.minimum(aux.action_ptr, ACTION_SEQ_LEN)
    action_mask = jnp.arange(ACTION_SEQ_LEN) >= (ACTION_SEQ_LEN - n_valid)

    return action_tensor, action_mask


@jax.jit
def encode_obs(
    state: State,
    aux: AuxState,
    player_id: chex.Array,
) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
    """Build card/action tensors + legal mask from state.

    Returns:
        card_tensor: (CARD_SEQ_LEN, CARD_INPUT_DIM) = (9, 2) int32
        action_tensor: (ACTION_SEQ_LEN, ACTION_INPUT_DIM) = (40, 7) float32
        action_mask: (ACTION_SEQ_LEN,) = (40,) bool
        legal_mask: (NUM_ACTIONS,) = (13,) bool
    """
    card_tensor = encode_cards(state, player_id)
    action_tensor, action_mask = encode_actions(aux, player_id)
    legal_mask = ENV.get_legal_actions(state)
    
    return card_tensor, action_tensor, action_mask, legal_mask


# =============================================================================
# 3) Replay Buffer (stores unrolls for proper V-trace computation)
# =============================================================================


@chex.dataclass(frozen=True)
class Unroll:
    """Single unroll (trajectory) for training.

    Stored as (rollout_len, num_envs, ...) to enable proper V-trace computation
    across time steps. adv/ret are NOT stored - computed at training time.
    """
    card: chex.Array        # (T, E, 9, 2) int32 - card tensor
    action_hist: chex.Array # (T, E, 40, 7) float32 - action history
    action_mask: chex.Array # (T, E, 40) bool - action history mask
    legal: chex.Array       # (T, E, 13) bool - legal actions
    action: chex.Array      # (T, E) int32 - action taken
    logp_mu: chex.Array     # (T, E) float32 - log prob under BEHAVIOR policy
    reward: chex.Array      # (T, E) float32
    done: chex.Array        # (T, E) bool
    contrib_self: chex.Array   # (T, E) int32 - chips contributed by self
    contrib_opp: chex.Array    # (T, E) int32 - chips contributed by opponent
    valid: chex.Array       # (T, E) bool - is this a learner turn
    # Bootstrap data for V-trace: observation of state AFTER last transition
    bootstrap_card: chex.Array        # (E, 9, 2) - card tensor for s_T
    bootstrap_action_hist: chex.Array # (E, 40, 7) - action history for s_T
    bootstrap_action_mask: chex.Array # (E, 40) - action mask for s_T
    bootstrap_done: chex.Array        # (E,) - whether last transition was terminal


@chex.dataclass(frozen=True)
class UnrollBuffer:
    """Buffer storing complete unrolls for V-trace computation.

    Each entry is a full trajectory of shape (rollout_len, num_envs, ...).
    This enables proper importance weight computation across time.
    """
    card: chex.Array        # (cap, T, E, 9, 2)
    action_hist: chex.Array # (cap, T, E, 40, 7)
    action_mask: chex.Array # (cap, T, E, 40)
    legal: chex.Array       # (cap, T, E, 13)
    action: chex.Array      # (cap, T, E)
    logp_mu: chex.Array     # (cap, T, E) - behavior policy log prob
    reward: chex.Array      # (cap, T, E)
    done: chex.Array        # (cap, T, E)
    contrib_self: chex.Array # (cap, T, E)
    contrib_opp: chex.Array  # (cap, T, E)
    valid: chex.Array       # (cap, T, E)
    # Bootstrap data for V-trace
    bootstrap_card: chex.Array        # (cap, E, 9, 2)
    bootstrap_action_hist: chex.Array # (cap, E, 40, 7)
    bootstrap_action_mask: chex.Array # (cap, E, 40)
    bootstrap_done: chex.Array        # (cap, E)
    ptr: chex.Array
    size: chex.Array


def unroll_buffer_init(capacity: int, rollout_len: int, num_envs: int) -> UnrollBuffer:
    """Initialize empty unroll buffer."""
    T, E = rollout_len, num_envs
    return UnrollBuffer(
        card=jnp.zeros((capacity, T, E, CARD_SEQ_LEN, CARD_INPUT_DIM), dtype=jnp.int32),
        action_hist=jnp.zeros((capacity, T, E, ACTION_SEQ_LEN, ACTION_INPUT_DIM), dtype=jnp.float32),
        action_mask=jnp.zeros((capacity, T, E, ACTION_SEQ_LEN), dtype=jnp.bool_),
        legal=jnp.zeros((capacity, T, E, NUM_ACTIONS), dtype=jnp.bool_),
        action=jnp.zeros((capacity, T, E), dtype=jnp.int32),
        logp_mu=jnp.zeros((capacity, T, E), dtype=jnp.float32),
        reward=jnp.zeros((capacity, T, E), dtype=jnp.float32),
        done=jnp.zeros((capacity, T, E), dtype=jnp.bool_),
        contrib_self=jnp.zeros((capacity, T, E), dtype=jnp.int32),
        contrib_opp=jnp.zeros((capacity, T, E), dtype=jnp.int32),
        valid=jnp.zeros((capacity, T, E), dtype=jnp.bool_),
        # Bootstrap data
        bootstrap_card=jnp.zeros((capacity, E, CARD_SEQ_LEN, CARD_INPUT_DIM), dtype=jnp.int32),
        bootstrap_action_hist=jnp.zeros((capacity, E, ACTION_SEQ_LEN, ACTION_INPUT_DIM), dtype=jnp.float32),
        bootstrap_action_mask=jnp.zeros((capacity, E, ACTION_SEQ_LEN), dtype=jnp.bool_),
        bootstrap_done=jnp.zeros((capacity, E), dtype=jnp.bool_),
        ptr=jnp.int32(0),
        size=jnp.int32(0),
    )


@jax.jit
def unroll_buffer_add(buf: UnrollBuffer, unroll: Unroll) -> UnrollBuffer:
    """Add a single unroll to the buffer."""
    cap = buf.card.shape[0]
    idx = buf.ptr % cap

    buf2 = buf.replace(
        card=buf.card.at[idx].set(unroll.card),
        action_hist=buf.action_hist.at[idx].set(unroll.action_hist),
        action_mask=buf.action_mask.at[idx].set(unroll.action_mask),
        legal=buf.legal.at[idx].set(unroll.legal),
        action=buf.action.at[idx].set(unroll.action),
        logp_mu=buf.logp_mu.at[idx].set(unroll.logp_mu),
        reward=buf.reward.at[idx].set(unroll.reward),
        done=buf.done.at[idx].set(unroll.done),
        contrib_self=buf.contrib_self.at[idx].set(unroll.contrib_self),
        contrib_opp=buf.contrib_opp.at[idx].set(unroll.contrib_opp),
        valid=buf.valid.at[idx].set(unroll.valid),
        bootstrap_card=buf.bootstrap_card.at[idx].set(unroll.bootstrap_card),
        bootstrap_action_hist=buf.bootstrap_action_hist.at[idx].set(unroll.bootstrap_action_hist),
        bootstrap_action_mask=buf.bootstrap_action_mask.at[idx].set(unroll.bootstrap_action_mask),
        bootstrap_done=buf.bootstrap_done.at[idx].set(unroll.bootstrap_done),
        ptr=(buf.ptr + 1) % cap,
        size=jnp.minimum(buf.size + 1, cap),
    )
    return buf2


@partial(jax.jit, static_argnums=(2,))
def unroll_buffer_sample(
    buf: UnrollBuffer,
    key: chex.PRNGKey,
    batch_size: int,
) -> Tuple[Unroll, chex.PRNGKey]:
    """Sample batch_size unrolls from the buffer using O(1) direct indexing."""
    key, sub = jax.random.split(key)
    cap = buf.card.shape[0]

    # Calculate how many recent unrolls are available for sampling
    transitions_per_unroll = CFG.rollout_len * CFG.num_envs
    recent_window_unrolls = jnp.where(
        CFG.recent_window_transitions > 0,
        jnp.maximum(1, jnp.int32((CFG.recent_window_transitions + transitions_per_unroll - 1) // transitions_per_unroll)),
        cap,
    )
    # The number of valid items we can sample from (e.g., min(200k, 16))
    available = jnp.minimum(buf.size, recent_window_unrolls)

    # OPTIMIZATION: Sample indices directly from range [0, available)
    # This avoids creating a 200,000-element mask every step
    sample_indices = jax.random.randint(sub, (batch_size,), 0, available)

    # Map these "relative ages" back to actual buffer indices
    # index = (current_ptr - 1 - relative_age) % capacity
    idx = (buf.ptr - 1 - sample_indices) % cap

    unroll = Unroll(
        card=buf.card[idx],
        action_hist=buf.action_hist[idx],
        action_mask=buf.action_mask[idx],
        legal=buf.legal[idx],
        action=buf.action[idx],
        logp_mu=buf.logp_mu[idx],
        reward=buf.reward[idx],
        done=buf.done[idx],
        contrib_self=buf.contrib_self[idx],
        contrib_opp=buf.contrib_opp[idx],
        valid=buf.valid[idx],
        bootstrap_card=buf.bootstrap_card[idx],
        bootstrap_action_hist=buf.bootstrap_action_hist[idx],
        bootstrap_action_mask=buf.bootstrap_action_mask[idx],
        bootstrap_done=buf.bootstrap_done[idx],
    )
    return unroll, key


# =============================================================================
# 4) K-Best Self-Play Pool
# =============================================================================


@chex.dataclass(frozen=True)
class AgentPool:
    """Pool of historical agent parameters for self-play with diversity tracking.

    Retains agents based on:
    - Top 4 by ELO (performance)
    - 2 style-diverse agents (behavioral diversity via raise/fold/call rates)
    """
    params: Any              # Stacked params along axis 0
    elo: chex.Array          # (pool_max,) ELO ratings
    add_order: chex.Array    # (pool_max,) monotonic counter when agent was added
    style_stats: chex.Array  # (pool_max, 3) [raise_rate, fold_rate, call_rate]
    size: chex.Array         # Current number of agents
    add_counter: chex.Array  # Scalar: next add_order value


def pool_init(dummy_params: Any) -> AgentPool:
    """Initialize empty agent pool."""
    def _zeros_like(p):
        return jnp.zeros((CFG.pool_max,) + p.shape, dtype=p.dtype)

    params = jax.tree_util.tree_map(_zeros_like, dummy_params)
    elo = jnp.zeros((CFG.pool_max,), dtype=jnp.float32)
    add_order = jnp.full((CFG.pool_max,), -1, dtype=jnp.int32)  # -1 = not yet added
    style_stats = jnp.zeros((CFG.pool_max, 3), dtype=jnp.float32)  # [raise, fold, call]
    return AgentPool(
        params=params,
        elo=elo,
        add_order=add_order,
        style_stats=style_stats,
        size=jnp.int32(0),
        add_counter=jnp.int32(0),
    )


@jax.jit
def pool_add(
    pool: AgentPool,
    new_params: Any,
    initial_elo: chex.Array,
    style_stats: chex.Array,
) -> AgentPool:
    """Add a new agent to the pool with style stats.

    Args:
        pool: Current agent pool
        new_params: Parameters for the new agent
        initial_elo: Initial ELO rating
        style_stats: (3,) array of [raise_rate, fold_rate, call_rate]
    """
    size = pool.size
    can_append = size < CFG.pool_max

    # If pool is full, replace worst agent by ELO
    worst = jnp.argmin(jnp.where(jnp.arange(CFG.pool_max) < size, pool.elo, 1e9))
    write_idx = jnp.where(can_append, size, worst)

    def _write_leaf(pool_leaf, new_leaf):
        return pool_leaf.at[write_idx].set(new_leaf)

    params = jax.tree_util.tree_map(_write_leaf, pool.params, new_params)
    elo = pool.elo.at[write_idx].set(initial_elo)
    add_order = pool.add_order.at[write_idx].set(pool.add_counter)
    new_style_stats = pool.style_stats.at[write_idx].set(style_stats)
    size2 = jnp.where(can_append, size + 1, size)

    return pool.replace(
        params=params,
        elo=elo,
        add_order=add_order,
        style_stats=new_style_stats,
        size=size2,
        add_counter=pool.add_counter + 1,
    )


def elo_expected(ra: chex.Array, rb: chex.Array) -> chex.Array:
    """Expected score for player A against player B."""
    return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))


@jax.jit
def elo_update(
    ra: chex.Array,
    rb: chex.Array,
    score_a: chex.Array,
    k: float,
) -> Tuple[chex.Array, chex.Array]:
    """Update ELO ratings after a match."""
    ea = elo_expected(ra, rb)
    eb = 1.0 - ea
    ra2 = ra + k * (score_a - ea)
    rb2 = rb + k * ((1.0 - score_a) - eb)
    return ra2, rb2


@jax.jit
def pool_select_opponent(
    pool: AgentPool,
    key: chex.PRNGKey,
) -> Tuple[Any, chex.PRNGKey]:
    """Select opponent using diversity-aware sampling.

    Sampling distribution (when pool has enough agents):
    - 50% from top 4 by ELO (hard opponents)
    - 25% from 2 most style-diverse agents
    - 25% from 2 most recently added agents

    This prevents monoculture while maintaining challenge.
    """
    key, k_category, k_pick = jax.random.split(key, 3)
    size = pool.size
    valid_mask = jnp.arange(CFG.pool_max) < size

    # Category selection: 0=top, 1=diverse, 2=recent
    # Probabilities: 50%, 25%, 25%
    category = jax.random.choice(k_category, 3, p=jnp.array([0.50, 0.25, 0.25]))

    # === Category 0: Top by ELO (top 4) ===
    valid_elo = jnp.where(valid_mask, pool.elo, -1e9)
    _, top_idx = jax.lax.top_k(valid_elo, k=4)
    n_top = jnp.minimum(size, jnp.int32(4))
    top_pick = jax.random.randint(k_pick, (), 0, jnp.maximum(n_top, 1))
    idx_top = top_idx[top_pick]

    # === Category 1: Style-diverse (maximize distance from mean style) ===
    # Compute mean style across valid agents
    valid_f = valid_mask.astype(jnp.float32)
    n_valid = jnp.maximum(1.0, jnp.sum(valid_f))
    mean_style = jnp.sum(pool.style_stats * valid_f[:, None], axis=0) / n_valid

    # L2 distance from mean style (style diversity)
    style_dist = jnp.sum((pool.style_stats - mean_style) ** 2, axis=-1)
    style_dist = jnp.where(valid_mask, style_dist, -1e9)

    # Pick from top 2 most diverse
    _, diverse_idx = jax.lax.top_k(style_dist, k=2)
    n_diverse = jnp.minimum(size, jnp.int32(2))
    diverse_pick = jax.random.randint(k_pick, (), 0, jnp.maximum(n_diverse, 1))
    idx_diverse = diverse_idx[diverse_pick]

    # === Category 2: Most recent (top 2 by add_order) ===
    valid_add_order = jnp.where(valid_mask, pool.add_order, -1)
    _, recent_idx = jax.lax.top_k(valid_add_order, k=2)
    n_recent = jnp.minimum(size, jnp.int32(2))
    recent_pick = jax.random.randint(k_pick, (), 0, jnp.maximum(n_recent, 1))
    idx_recent = recent_idx[recent_pick]

    # Select based on category
    idx = jax.lax.switch(
        category,
        [
            lambda: idx_top,
            lambda: idx_diverse,
            lambda: idx_recent,
        ],
    )

    opp_params = jax.tree_util.tree_map(lambda p: p[idx], pool.params)
    return opp_params, key


# =============================================================================
# 5) Rewards
# =============================================================================


@jax.jit
def terminal_rewards(state: State) -> Tuple[chex.Array, chex.Array]:
    """Compute terminal rewards (chip deltas) for both players."""
    pot_total = (state.pot + state.p0_bet + state.p1_bet).astype(jnp.float32)
    contrib0 = (STACK_SIZE - state.p0_stack).astype(jnp.float32)
    contrib1 = (STACK_SIZE - state.p1_stack).astype(jnp.float32)
    
    r0 = jnp.select(
        [state.winner == 0, state.winner == 1, state.winner == 2],
        [pot_total - contrib0, -contrib0, pot_total * 0.5 - contrib0],
        default=0.0,
    )
    r1 = jnp.select(
        [state.winner == 1, state.winner == 0, state.winner == 2],
        [pot_total - contrib1, -contrib1, pot_total * 0.5 - contrib1],
        default=0.0,
    )
    return r0, r1


# =============================================================================
# 6) V-trace Policy Gradient (IMPALA-style, no PPO)
# =============================================================================


@jax.jit
def clip_value_target(
    ret: chex.Array,
    contrib_self: chex.Array,
    contrib_opp: chex.Array,
) -> chex.Array:
    """Clip value target to chip contribution bounds for stability.

    In poker, max loss = chips contributed, max gain = opponent's contribution.
    Both ret and bounds are normalized by STACK_SIZE.
    """
    # Normalize bounds to match normalized returns
    d2 = contrib_self.astype(jnp.float32) / jnp.float32(STACK_SIZE)
    d3 = contrib_opp.astype(jnp.float32) / jnp.float32(STACK_SIZE)
    return jnp.clip(ret, -d2, d3)


def masked_categorical(logits: chex.Array, legal: chex.Array) -> distrax.Categorical:
    """Create categorical distribution with illegal actions masked."""
    # Use -1e9 instead of jnp.finfo(float32).min to avoid numerical issues
    # (finfo.min is a large negative finite number that can cause overflow in some ops)
    neg_inf = -1e9
    masked_logits = jnp.where(legal, logits, neg_inf)
    return distrax.Categorical(logits=masked_logits)


# =============================================================================
# 7) Training State and Forward Pass
# =============================================================================


def make_lr_schedule(total_updates: int) -> optax.Schedule:
    """Create learning rate schedule with warmup and cosine decay."""
    warmup_fn = optax.linear_schedule(
        init_value=CFG.lr_init,
        end_value=CFG.lr_peak,
        transition_steps=CFG.warmup_updates,
    )
    decay_fn = optax.cosine_decay_schedule(
        init_value=CFG.lr_peak,
        decay_steps=max(1, total_updates - CFG.warmup_updates),
        alpha=CFG.lr_final / CFG.lr_peak,  # Final LR as fraction of peak
    )
    return optax.join_schedules(
        schedules=[warmup_fn, decay_fn],
        boundaries=[CFG.warmup_updates],
    )


def get_entropy_coef(update_step: int) -> chex.Array:
    """Get entropy coefficient with linear decay schedule."""
    progress = jnp.minimum(1.0, jnp.float32(update_step) / CFG.ent_decay_steps)
    return CFG.ent_coef_init + progress * (CFG.ent_coef_final - CFG.ent_coef_init)


def make_train_state(rng: chex.PRNGKey, total_updates: int) -> TrainState:
    """Initialize model and optimizer with LR schedule."""
    model = TossHoldemModel()

    # Dummy inputs for initialization
    dummy_card = jnp.zeros((1, CARD_SEQ_LEN, CARD_INPUT_DIM), dtype=jnp.int32)
    dummy_action = jnp.zeros((1, ACTION_SEQ_LEN, ACTION_INPUT_DIM), dtype=jnp.float32)
    dummy_mask = jnp.ones((1, ACTION_SEQ_LEN), dtype=jnp.bool_)

    variables = model.init(rng, dummy_card, dummy_action, dummy_mask)

    # Use LR schedule with warmup and cosine decay
    lr_schedule = make_lr_schedule(total_updates)
    tx = optax.chain(
        optax.clip_by_global_norm(CFG.max_grad_norm),
        optax.adam(lr_schedule),
    )

    return TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=tx,
    )


# Create model instance once at module level to avoid reinstantiation inside JIT
_FORWARD_MODEL = TossHoldemModel()


@jax.jit
def forward_apply(
    params: Any,
    card: chex.Array,
    action_hist: chex.Array,
    action_mask: chex.Array,
) -> Tuple[chex.Array, chex.Array]:
    """Forward pass through the model.

    Uses the module-level _FORWARD_MODEL instance to avoid reinstantiating
    the model inside JIT, which is a Flax anti-pattern.

    Args:
        params: Model parameters
        card: (B, 9, 2) card tensor
        action_hist: (B, 40, 7) action history
        action_mask: (B, 40) action mask

    Returns:
        logits: (B, NUM_ACTIONS) action logits
        value: (B,) value estimates
    """
    logits, value = _FORWARD_MODEL.apply(
        {"params": params},
        card,
        action_hist,
        action_mask,
    )
    return logits, value


@jax.jit
def compute_vtrace_on_unroll(
    params: Any,
    unroll: Unroll,
) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]:
    """Compute V-trace targets and advantages on a single unroll at training time.

    This is the key function for proper IMPALA-style V-trace:
    1. Recompute logp_pi for the entire unroll under current policy
    2. Compute log_rhos = logp_pi - logp_mu (masked to 0 on opponent steps)
    3. Run V-trace to get value targets and advantages

    Note: We run forward pass on ALL transitions (including opponent turns) because
    V-trace needs value estimates at every timestep for TD error propagation.
    The logits on opponent turns are unused but computed together with values.

    Args:
        params: Current policy parameters
        unroll: Single unroll with shape (T, E, ...)

    Returns:
        logp_pi: (T*E,) current policy log probs (flat)
        values: (T*E,) value estimates (flat)
        vs: (T, E) V-trace value targets (stop_gradient applied)
        advantages: (T, E) V-trace advantages (stop_gradient applied)
        pg_rhos: (T, E) clipped importance weights for policy gradient (stop_gradient applied)
        masked_logits: (T*E, NUM_ACTIONS) masked logits for entropy computation
        flat_legal: (T*E, NUM_ACTIONS) legal action mask
    """
    T, E = unroll.card.shape[:2]

    # Flatten for batched forward pass: (T, E, ...) -> (T*E, ...)
    flat_card = unroll.card.reshape(T * E, *unroll.card.shape[2:])
    flat_action_hist = unroll.action_hist.reshape(T * E, *unroll.action_hist.shape[2:])
    flat_action_mask = unroll.action_mask.reshape(T * E, *unroll.action_mask.shape[2:])
    flat_legal = unroll.legal.reshape(T * E, *unroll.legal.shape[2:])
    flat_action = unroll.action.reshape(T * E)

    # Forward pass to get current policy logits and values
    logits, values_flat = forward_apply(
        params, flat_card, flat_action_hist, flat_action_mask
    )

    # Compute current policy log probs
    neg_inf = -1e9
    masked_logits = jnp.where(flat_legal, logits, neg_inf)
    logp_all = jax.nn.log_softmax(masked_logits, axis=-1)
    logp_pi_flat = jnp.take_along_axis(logp_all, flat_action[:, None], axis=-1)[:, 0]

    # Reshape to (T, E) for V-trace computation
    logp_pi_te = logp_pi_flat.reshape(T, E)
    values_te = values_flat.reshape(T, E)

    # Compute importance weights: log_rho = logp_pi - logp_mu
    # On opponent steps (valid=False), logp_mu=0 and we should set log_rho=0 too
    log_rhos = jnp.where(unroll.valid, logp_pi_te - unroll.logp_mu, 0.0)

    # Bootstrap value: V(s_T) - the value of the state AFTER the last transition
    # Use the stored bootstrap observation to compute this properly
    _, bootstrap_values = forward_apply(
        params,
        unroll.bootstrap_card,      # (E, CARD_SEQ_LEN, CARD_INPUT_DIM)
        unroll.bootstrap_action_hist,  # (E, ACTION_SEQ_LEN, ACTION_INPUT_DIM)
        unroll.bootstrap_action_mask,  # (E, ACTION_SEQ_LEN)
    )
    # Zero out bootstrap where the last transition was terminal
    # bootstrap = (1 - done_last) * V(s_T)
    last_value = jnp.where(unroll.bootstrap_done, 0.0, bootstrap_values)  # (E,)

    # Run V-trace per environment with BOTH rho_bar and pg_rho_bar
    def _vtrace_one_env(rewards, vals, dones, last_v, log_rho):
        return compute_vtrace(
            rewards, vals, dones, last_v, log_rho,
            CFG.rho_bar, CFG.c_bar, CFG.pg_rho_bar, CFG.gamma
        )

    vs, advantages, pg_rhos = jax.vmap(_vtrace_one_env, in_axes=(1, 1, 1, 0, 1), out_axes=(1, 1, 1))(
        unroll.reward.astype(jnp.float32),
        values_te,
        unroll.done,
        last_value,
        log_rhos,
    )

    # CRITICAL: Stop gradients through V-trace targets, advantages, and pg_rhos
    # These are targets/weights computed from the current policy, but gradients
    # should NOT flow back through the V-trace recursion
    vs = jax.lax.stop_gradient(vs)
    advantages = jax.lax.stop_gradient(advantages)
    pg_rhos = jax.lax.stop_gradient(pg_rhos)

    # Return flat versions for loss computation, plus masked_logits for entropy
    return logp_pi_flat, values_flat, vs, advantages, pg_rhos, masked_logits, flat_legal


@jax.jit
def vtrace_loss_on_unroll(
    params: Any,
    unroll: Unroll,
    ent_coef: chex.Array,
) -> Tuple[chex.Array, Dict[str, chex.Array]]:
    """Compute V-trace loss on a single unroll.

    This implements proper IMPALA V-trace by computing importance weights
    at training time, not at collection time.

    Args:
        params: Current policy parameters
        unroll: Single unroll with shape (T, E, ...)
        ent_coef: Entropy coefficient

    Returns:
        loss: Scalar loss
        metrics: Dict of training metrics
    """
    # Compute V-trace targets, advantages, and get logits for entropy (single forward pass)
    logp_pi_flat, values_flat, vs, advantages, pg_rhos, masked_logits, flat_legal = compute_vtrace_on_unroll(params, unroll)

    # Flatten V-trace outputs for loss computation
    vs_flat = vs.reshape(-1)
    advantages_flat = advantages.reshape(-1)
    pg_rhos_flat = pg_rhos.reshape(-1)
    valid_flat = unroll.valid.reshape(-1)
    contrib_self_flat = unroll.contrib_self.reshape(-1)
    contrib_opp_flat = unroll.contrib_opp.reshape(-1)

    valid_f = valid_flat.astype(jnp.float32)
    denom = jnp.maximum(1.0, jnp.sum(valid_f))

    # ==========================================================================
    # PURE V-TRACE POLICY GRADIENT (IMPALA-style)
    # ==========================================================================
    # L_policy = -E[ρ_clipped * log π(a|s) * A]
    # Standard V-trace: NO advantage normalization. The importance sampling
    # correction handles distribution mismatch; normalization can interfere
    # with IS weights and cause instability as the data mixture changes.
    policy_loss = -jnp.sum(pg_rhos_flat * logp_pi_flat * advantages_flat * valid_f) / denom

    # Value loss against V-trace targets
    # Train ONLY on learner-to-act states. Opponent-to-act states have noisy
    # targets because opponent's hidden cards affect transitions but aren't
    # in the learner's observation. V-trace recursion still crosses opponent
    # steps (using rho=1), but we don't supervise those value predictions.
    target = clip_value_target(vs_flat, contrib_self_flat, contrib_opp_flat)
    v_loss = (target - values_flat) ** 2
    value_loss = jnp.sum(v_loss * valid_f) / denom

    # Entropy bonus - reuse masked_logits from the single forward pass
    dist = masked_categorical(masked_logits, flat_legal)
    ent_flat = dist.entropy()
    entropy = jnp.sum(ent_flat * valid_f) / denom

    # Total loss
    total = policy_loss + CFG.vf_coef * value_loss - ent_coef * entropy

    # Metrics
    def _masked_mean(x, w):
        d = jnp.maximum(1.0, jnp.sum(w))
        return jnp.sum(x * w) / d

    # Compute rho only on learner steps (where logp_mu is meaningful)
    # On opponent steps, logp_mu=0 which would give misleading rho values
    logp_mu_flat = unroll.logp_mu.reshape(-1)
    rho_exp = jnp.where(valid_flat, jnp.exp(logp_pi_flat - logp_mu_flat), 1.0)
    rho_valid = jnp.where(valid_flat, rho_exp, 0.0)

    metrics = {
        "loss": total,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": entropy,
        "value_mean": _masked_mean(values_flat, valid_f),
        "adv_mean": _masked_mean(advantages_flat, valid_f),
        "ret_mean": _masked_mean(vs_flat, valid_f),
        "valid_frac": jnp.mean(valid_f),
        "rho_mean": _masked_mean(rho_valid, valid_f),
        "rho_max": jnp.max(rho_valid),
    }

    return total, metrics


def train_step_on_unroll(
    state: TrainState,
    unroll: Unroll,
    ent_coef: chex.Array,
) -> Tuple[TrainState, Dict[str, chex.Array]]:
    """Single training step with V-trace computed at training time.

    Args:
        state: Current training state
        unroll: Single unroll with shape (T, E, ...)
        ent_coef: Current entropy coefficient
    """
    def _loss_fn(params, unroll_, ent_c):
        return vtrace_loss_on_unroll(params, unroll_, ent_c)

    (loss, metrics), grads = jax.value_and_grad(_loss_fn, has_aux=True)(
        state.params, unroll, ent_coef
    )
    new_state = state.apply_gradients(grads=grads)
    metrics = dict(metrics)
    metrics["loss"] = loss
    metrics["ent_coef"] = ent_coef

    return new_state, metrics


# =============================================================================
# 8) Rollout Collection
# =============================================================================


@chex.dataclass(frozen=True)
class VecEnvState:
    """Vectorized environment state."""
    env_state: State
    aux: AuxState
    main_is_p0: chex.Array  # If True, main agent plays as player0
    rng: chex.Array         # Per-env PRNG keys


def vec_reset(
    key: chex.PRNGKey,
    n: int,
) -> Tuple[VecEnvState, chex.PRNGKey]:
    """Reset vectorized environments using the global ENV instance."""
    key, k_seat, k_env = jax.random.split(key, 3)
    main_is_p0 = jax.random.bernoulli(k_seat, 0.5, (n,))

    ks = jax.random.split(k_env, n)
    s = jax.vmap(ENV.reset)(ks)
    aux = jax.vmap(lambda _: aux_reset())(jnp.arange(n))

    return VecEnvState(env_state=s, aux=aux, main_is_p0=main_is_p0, rng=ks), key


@partial(jax.jit, static_argnums=(4,))
def policy_act(
    params: Any,
    state: State,
    aux: AuxState,
    key: chex.PRNGKey,
    deterministic: bool = False,
) -> Tuple[chex.Array, chex.Array, chex.Array, chex.PRNGKey]:
    """Select action for the current player.

    Args:
        params: Model parameters
        state: Current game state
        aux: Auxiliary state for action history
        key: Random key
        deterministic: If True, use argmax (for stable ELO evaluation).
                       If False, use Gumbel-argmax sampling (for training).

    Returns:
        action: Selected action
        logp: Log probability of action
        value: Value estimate
        key: Updated random key
    """
    pid = state.current_player.astype(jnp.int32)
    card, action_hist, action_mask, legal = encode_obs(state, aux, pid)

    # Add batch dimension
    card_b = card[None, ...]
    action_hist_b = action_hist[None, ...]
    action_mask_b = action_mask[None, ...]

    logits_b, value_b = forward_apply(params, card_b, action_hist_b, action_mask_b)
    logits = logits_b[0]
    value = value_b[0]

    neg_inf = -1e9
    masked_logits = jnp.where(legal, logits, neg_inf)

    # Deterministic: pure argmax for stable evaluation
    # Stochastic: Gumbel-argmax for exploration during training
    key, key_sample = jax.random.split(key)

    def _deterministic_action():
        return jnp.argmax(masked_logits).astype(jnp.int32)

    def _stochastic_action():
        g = jax.random.gumbel(key_sample, (NUM_ACTIONS,), dtype=jnp.float32)
        return jnp.argmax(masked_logits + g).astype(jnp.int32)

    action = jax.lax.cond(deterministic, _deterministic_action, _stochastic_action)

    # Compute log prob for this action
    logp_all = jax.nn.log_softmax(masked_logits)
    logp = logp_all[action].astype(jnp.float32)

    return action, logp, value.astype(jnp.float32), key


@jax.jit
def env_step_and_optional_reset(
    s: State,
    key: chex.PRNGKey,
    action: chex.Array,
) -> Tuple[State, State, chex.Array, chex.PRNGKey]:
    """Step environment and reset if done."""
    k_step = jax.random.fold_in(key, jnp.uint32(0))
    k_reset = jax.random.fold_in(key, jnp.uint32(1))
    
    s_next = ENV.step(s, k_step, action)
    done = s_next.done
    s_reset = ENV.reset(k_reset)
    s_cont = jax.tree_util.tree_map(
        lambda a, b: jnp.where(done, a, b),
        s_reset,
        s_next,
    )
    
    return s_next, s_cont, done, key


@jax.jit
def reset_aux_where_done(aux: AuxState, done: chex.Array) -> AuxState:
    """Reset auxiliary state for completed episodes."""
    done_expanded = done[:, None, None]
    action_tensor = jnp.where(
        done_expanded,
        jnp.zeros_like(aux.action_tensor),
        aux.action_tensor,
    )
    action_ptr = jnp.where(done, jnp.zeros_like(aux.action_ptr), aux.action_ptr)

    return aux.replace(
        action_tensor=action_tensor,
        action_ptr=action_ptr,
    )


@jax.jit
def compute_vtrace(
    rewards: chex.Array,
    values: chex.Array,
    dones: chex.Array,
    last_value: chex.Array,
    log_rhos: chex.Array,
    rho_bar: float,
    c_bar: float,
    pg_rho_bar: float,
    gamma: float,
) -> Tuple[chex.Array, chex.Array, chex.Array]:
    """Compute V-trace targets and advantages for off-policy learning.

    V-trace (from IMPALA) provides theoretically sound off-policy corrections
    by a importance weights to bound variance. This allows learning
    from stale data collected by different policy versions.

    Reference: Espeholt et al. "IMPALA: Scalable Distributed Deep-RL with
               Importance Weighted Actor-Learner Architectures" (2018)

    Args:
        rewards: (T,) rewards at each timestep
        values: (T,) value estimates V(s_t) at each timestep
        dones: (T,) episode termination flags
        last_value: scalar, bootstrap value V(s_T) for final state
        log_rhos: (T,) log importance weights log(π(a|s) / μ(a|s))
        rho_bar: clip threshold for value target importance weights
        c_bar: clip threshold for trace cutting coefficients
        pg_rho_bar: clip threshold for policy gradient importance weights
        gamma: discount factor

    Returns:
        vs: (T,) V-trace value targets
        advantages: (T,) policy gradient advantages
        pg_rhos: (T,) clipped importance weights for policy gradient
    """
    T = rewards.shape[0]
    not_done = 1.0 - dones.astype(jnp.float32)

    # Compute importance weights and clip them
    rhos = jnp.exp(log_rhos)
    rho_v = jnp.minimum(rho_bar, rhos)         # ρ̂_t for value target
    rho_pg = jnp.minimum(pg_rho_bar, rhos)     # ρ_pg for policy gradient
    cs = jnp.minimum(c_bar, rhos)              # c_t for trace

    # Append bootstrap value: values_extended[T] = last_value
    values_extended = jnp.concatenate([values, last_value[None]])

    # Temporal differences: δ_t = ρ̂_t * (r_t + γ * V(s_{t+1}) - V(s_t))
    # Note: V(s_{t+1}) is values_extended[t+1], V(s_t) is values_extended[t]
    td_targets = rewards + gamma * not_done * values_extended[1:]
    deltas = rho_v * (td_targets - values_extended[:-1])

    # V-trace: v_s = V(s) + Σ_{t=s}^{T-1} γ^{t-s} (Π_{i=s}^{t-1} c_i) δ_t
    # Compute backwards: v_{t} = V(s_t) + δ_t + γ * c_t * (1-done_t) * (v_{t+1} - V(s_{t+1}))
    # where (v_{t+1} - V(s_{t+1})) is the "carry" from the future

    # Reverse for backward scan
    deltas_rev = jnp.flip(deltas, axis=0)
    cs_rev = jnp.flip(cs, axis=0)
    not_done_rev = jnp.flip(not_done, axis=0)

    def vtrace_step(carry, inputs):
        """Backward accumulation of V-trace corrections."""
        delta, c, nd = inputs
        # carry = v_{t+1} - V(s_{t+1}) from the future
        # new_carry = δ_t + γ * c_t * (1-done_t) * carry
        new_carry = delta + gamma * c * nd * carry
        return new_carry, new_carry

    # Initial carry is 0 (no correction beyond the horizon)
    _, corrections_rev = jax.lax.scan(
        vtrace_step, 0.0, (deltas_rev, cs_rev, not_done_rev)
    )
    corrections = jnp.flip(corrections_rev, axis=0)

    # V-trace targets: v_s = V(s) + correction
    vs = values + corrections

    # Advantages for policy gradient:
    # A_t = ρ_pg * (r_t + γ * v_{t+1} - V(s_t))
    # where v_{t+1} is the V-trace target at t+1 (or last_value if t=T-1)
    # Note: use rho_pg (clipped with pg_rho_bar), not rho_v
    vs_extended = jnp.concatenate([vs, last_value[None]])
    vs_next = vs_extended[1:]
    advantages = rho_pg * (rewards + gamma * not_done * vs_next - values)

    return vs, advantages, rho_pg  


@jax.jit
def collect_rollout(
    main_params: Any,
    pool: AgentPool,
    buf: UnrollBuffer,
    vec: VecEnvState,
    key: chex.PRNGKey,
) -> Tuple[UnrollBuffer, VecEnvState, chex.PRNGKey]:
    """Collect rollout_len transitions per environment.

    Stores raw trajectory data as an Unroll. V-trace is computed at training time
    with proper importance weights (log_rhos = logp_pi - logp_mu).

    Key change from previous version:
    - NO V-trace computation here - just store observations and behavior policy log probs
    - Critic uses PRIVATE cards (same as actor) for value estimation
    """
    # Select opponent from pool
    opp_params, key = pool_select_opponent(pool, key)
    opp_params = jax.tree_util.tree_map(
        lambda o, m: jax.lax.cond(pool.size == 0, lambda _: m, lambda _: o, operand=None),
        opp_params,
        main_params,
    )

    env_ids = jnp.arange(CFG.num_envs, dtype=jnp.uint32)

    def step_fn(carry, t):
        vec = carry
        s = vec.env_state
        aux = vec.aux
        main_is_p0 = vec.main_is_p0
        base_keys = vec.rng
        learner_pid = jnp.where(main_is_p0, jnp.int32(0), jnp.int32(1))
        pid_cur = s.current_player.astype(jnp.int32)
        is_learner_turn = pid_cur == learner_pid

        # Encode from current player's perspective (for action selection)
        # Legal mask is player-independent, so we can reuse it
        card_cur_b, action_hist_cur_b, action_mask_cur_b, legal_b = jax.vmap(encode_obs)(
            s, aux, pid_cur
        )

        # For storage: need learner's perspective
        # Avoid computing learner encodings when learner == current player
        def _select_learner_obs(s_i, aux_i, pid_cur_i, learner_pid_i, card_cur_i, action_hist_cur_i, action_mask_cur_i):
            def _use_cur(_):
                return card_cur_i, action_hist_cur_i, action_mask_cur_i

            def _use_learner(_):
                card_l = encode_cards(s_i, learner_pid_i)
                action_hist_l, action_mask_l = encode_actions(aux_i, learner_pid_i)
                return card_l, action_hist_l, action_mask_l

            return jax.lax.cond(pid_cur_i == learner_pid_i, _use_cur, _use_learner, operand=None)

        card_b, action_hist_b, action_mask_b = jax.vmap(_select_learner_obs)(
            s, aux, pid_cur, learner_pid, card_cur_b, action_hist_cur_b, action_mask_cur_b
        )

        # Forward pass for main agent (learner)
        logits_main_b, _ = forward_apply(
            main_params, card_cur_b, action_hist_cur_b, action_mask_cur_b
        )

        # Opponent forward (conditional)
        has_opponent = pool.size > 0

        def compute_opp_logits():
            return forward_apply(opp_params, card_cur_b, action_hist_cur_b, action_mask_cur_b)[0]

        def use_main_logits():
            return logits_main_b

        logits_opp_b = jax.lax.cond(has_opponent, compute_opp_logits, use_main_logits)

        # Select logits based on whose turn it is
        logits = jnp.where(is_learner_turn[:, None], logits_main_b, logits_opp_b)

        # Sample actions
        def _mk_key(k, eid, salt):
            k = jax.random.fold_in(k, eid)
            return jax.random.fold_in(k, t + jnp.uint32(salt))

        k_act = jax.vmap(lambda k, eid: _mk_key(k, eid, 0))(base_keys, env_ids)
        g = jax.vmap(lambda k: jax.random.gumbel(k, (NUM_ACTIONS,), dtype=jnp.float32))(k_act)
        neg_inf = -1e9
        masked_logits = jnp.where(legal_b, logits, neg_inf)
        action = jnp.argmax(masked_logits + g, axis=-1).astype(jnp.int32)

        # Compute behavior policy log prob (logp_mu) - only meaningful on learner turns
        # For non-learner turns, we store 0 and mask later
        logp_all = jax.nn.log_softmax(masked_logits, axis=-1)
        logp = jnp.take_along_axis(logp_all, action[:, None], axis=-1)[:, 0].astype(jnp.float32)
        logp_mu = jnp.where(is_learner_turn, logp, jnp.float32(0.0))

        # Record action
        aux2 = jax.vmap(aux_record_action)(aux, s, action)

        # Step environment
        k_env = jax.vmap(lambda k, eid: _mk_key(k, eid, 131071))(base_keys, env_ids)
        s_next, s_cont, done, _ = jax.vmap(env_step_and_optional_reset)(s, k_env, action)

        # Compute rewards (normalized by STACK_SIZE for stable training)
        # This puts rewards in ~[-1, +1] range so policy and value losses are comparable
        r0, r1 = terminal_rewards(s_next)
        rew_term = jnp.where(main_is_p0, r0, r1)
        rew = jnp.where(done, rew_term, jnp.float32(0.0))
        rew = rew / jnp.float32(STACK_SIZE)  # Normalize rewards

        # Chip contributions for value clipping
        contrib_p0_pre = (STACK_SIZE - s.p0_stack).astype(jnp.int32)
        contrib_p1_pre = (STACK_SIZE - s.p1_stack).astype(jnp.int32)
        contrib_self_pre = jnp.where(main_is_p0, contrib_p0_pre, contrib_p1_pre)
        contrib_opp_pre = jnp.where(main_is_p0, contrib_p1_pre, contrib_p0_pre)

        contrib_p0_post = (STACK_SIZE - s_next.p0_stack).astype(jnp.int32)
        contrib_p1_post = (STACK_SIZE - s_next.p1_stack).astype(jnp.int32)
        contrib_self_post = jnp.where(main_is_p0, contrib_p0_post, contrib_p1_post)
        contrib_opp_post = jnp.where(main_is_p0, contrib_p1_post, contrib_p0_post)

        contrib_self = jnp.where(done, contrib_self_post, contrib_self_pre)
        contrib_opp = jnp.where(done, contrib_opp_post, contrib_opp_pre)

        # Store learner's observation (with private cards)
        tmp = {
            "card": card_b,
            "action_hist": action_hist_b,
            "action_mask": action_mask_b,
            "legal": legal_b,
            "action": action,
            "logp_mu": logp_mu,
            "reward": rew,
            "done": done,
            "contrib_self": contrib_self,
            "contrib_opp": contrib_opp,
            "valid": is_learner_turn,
        }

        # Reset aux for completed episodes
        aux3 = reset_aux_where_done(aux2, done)

        # Re-randomize seating on resets
        k_seat = jax.vmap(lambda k, eid: _mk_key(k, eid, 262143))(base_keys, env_ids)
        new_seat = jax.vmap(lambda k: jax.random.bernoulli(k, 0.5))(k_seat)
        main_is_p0_2 = jnp.where(done, new_seat, main_is_p0)

        vec2 = VecEnvState(env_state=s_cont, aux=aux3, main_is_p0=main_is_p0_2, rng=base_keys)
        return vec2, tmp

    vec2, traj = jax.lax.scan(
        step_fn,
        vec,
        jnp.arange(CFG.rollout_len, dtype=jnp.uint32),
    )

    # Advance RNGs
    key, sub = jax.random.split(key)
    vec2 = vec2.replace(rng=jax.random.split(sub, CFG.num_envs))

    # Capture bootstrap observation: state AFTER the last transition (s_T)
    # This is the state in vec2 before any new transitions
    # We need the observation from the LEARNER's perspective at the end of the unroll
    bootstrap_learner_pid = jnp.where(vec2.main_is_p0, jnp.int32(0), jnp.int32(1))
    bootstrap_card, bootstrap_action_hist, bootstrap_action_mask, _ = jax.vmap(encode_obs)(
        vec2.env_state, vec2.aux, bootstrap_learner_pid
    )
    # Track whether the last transition was terminal (bootstrap should be 0 if so)
    bootstrap_done = traj["done"][-1, :]  # (E,) - done flag at last timestep

    # Store unroll WITHOUT computing V-trace - that happens at training time
    # traj has shape (T, E, ...) which is what we want
    unroll = Unroll(
        card=traj["card"],
        action_hist=traj["action_hist"],
        action_mask=traj["action_mask"],
        legal=traj["legal"],
        action=traj["action"],
        logp_mu=traj["logp_mu"],
        reward=traj["reward"],
        done=traj["done"],
        contrib_self=traj["contrib_self"],
        contrib_opp=traj["contrib_opp"],
        valid=traj["valid"],
        bootstrap_card=bootstrap_card,
        bootstrap_action_hist=bootstrap_action_hist,
        bootstrap_action_mask=bootstrap_action_mask,
        bootstrap_done=bootstrap_done,
    )

    buf = unroll_buffer_add(buf, unroll)
    return buf, vec2, key


# =============================================================================
# 9) Pool Evaluation
# =============================================================================


@jax.jit
def play_one_hand(
    params_a: Any,
    params_b: Any,
    key: chex.PRNGKey,
) -> Tuple[chex.Array, chex.PRNGKey]:
    """Play one hand: player0 uses params_a, player1 uses params_b.

    Uses DETERMINISTIC action selection for stable ELO evaluation.
    This prevents variance in ELO ratings due to sampling randomness.

    Returns:
        r0: Raw chip delta for player 0 (player A). Positive = A won chips.
        key: Updated random key.

    Uses the global ENV instance.
    """
    key, k_reset = jax.random.split(key)
    s = ENV.reset(k_reset)
    aux = aux_reset()

    def cond_fn(carry):
        s, aux, key = carry
        return ~s.done

    def body_fn(carry):
        s, aux, key = carry
        pid = s.current_player.astype(jnp.int32)
        params = jax.lax.cond(pid == 0, lambda _: params_a, lambda _: params_b, operand=None)
        # Use deterministic=True for stable ELO evaluation
        a, _, _, key = policy_act(params, s, aux, key, deterministic=True)
        aux2 = aux_record_action(aux, s, a)
        key, k_step = jax.random.split(key)
        s2 = ENV.step(s, k_step, a)
        return (s2, aux2, key)

    sT, _, key = jax.lax.while_loop(cond_fn, body_fn, (s, aux, key))
    r0, _ = terminal_rewards(sT)
    # Return raw chip delta instead of win/loss score
    return r0, key


@jax.jit
def play_one_hand_with_action_counts(
    params_a: Any,
    params_b: Any,
    key: chex.PRNGKey,
) -> Tuple[chex.Array, chex.Array, chex.PRNGKey]:
    """Play one hand and track action counts for player A (P0).

    Returns:
        r0: Chip delta for player 0
        action_counts: (3,) array [raise_count, fold_count, call_count] for P0
        key: Updated random key
    """
    key, k_reset = jax.random.split(key)
    s = ENV.reset(k_reset)
    aux = aux_reset()

    def cond_fn(carry):
        s, aux, key, counts = carry
        return ~s.done

    def body_fn(carry):
        s, aux, key, counts = carry
        pid = s.current_player.astype(jnp.int32)
        params = jax.lax.cond(pid == 0, lambda _: params_a, lambda _: params_b, operand=None)
        a, _, _, key = policy_act(params, s, aux, key, deterministic=True)
        aux2 = aux_record_action(aux, s, a)
        key, k_step = jax.random.split(key)
        s2 = ENV.step(s, k_step, a)

        # Track actions for player 0 only
        # Action classification: 0=fold, 1-2=check/call, 3+=raise
        is_p0 = pid == 0
        is_raise = (a >= 3).astype(jnp.float32)
        is_fold = (a == 0).astype(jnp.float32)
        is_call = ((a == 1) | (a == 2)).astype(jnp.float32)
        p0_mask = is_p0.astype(jnp.float32)

        new_counts = counts + p0_mask * jnp.array([is_raise, is_fold, is_call])
        return (s2, aux2, key, new_counts)

    init_counts = jnp.zeros(3, dtype=jnp.float32)
    sT, _, key, final_counts = jax.lax.while_loop(
        cond_fn, body_fn, (s, aux, key, init_counts)
    )
    r0, _ = terminal_rewards(sT)
    return r0, final_counts, key


@jax.jit
def compute_style_stats(
    params: Any,
    key: chex.PRNGKey,
    n_hands: int = 64,
) -> Tuple[chex.Array, chex.PRNGKey]:
    """Compute style statistics for an agent by playing hands against itself.

    Returns:
        style_stats: (3,) array [raise_rate, fold_rate, call_rate]
        key: Updated random key
    """
    key, sub = jax.random.split(key)
    keys = jax.random.split(sub, n_hands)

    # Play hands with agent as both P0 and P1 (self-play)
    _, action_counts_all, _ = jax.vmap(
        play_one_hand_with_action_counts, in_axes=(None, None, 0)
    )(params, params, keys)

    # Sum counts across all hands
    total_counts = jnp.sum(action_counts_all, axis=0)  # (3,)
    total_actions = jnp.sum(total_counts)

    # Compute rates (avoid division by zero)
    style_stats = total_counts / jnp.maximum(1.0, total_actions)

    return style_stats, key


@jax.jit
def eval_and_update_pool(
    pool: AgentPool,
    main_state: TrainState,
    main_elo: jnp.ndarray,
    key: chex.PRNGKey,
) -> Tuple[AgentPool, jnp.ndarray, chex.PRNGKey]:
    """Evaluate main agent against pool and update ELOs with diversity-aware retention."""
    pool_max = CFG.pool_max
    eval_hands = CFG.eval_hands
    elo_k = CFG.elo_k
    
    # Generate keys for matches
    key, sub = jax.random.split(key)
    keys_grid = jax.random.split(sub, pool_max * eval_hands).reshape((pool_max, eval_hands, -1))
    
    valid_mask = jnp.arange(pool_max) < pool.size
    valid_f = valid_mask.astype(jnp.float32)
    n_valid = jnp.maximum(1.0, jnp.sum(valid_f))
    
    # Play matches - half as P0, half as P1 to eliminate position bias
    # Use match-based money metric: sum chip deltas, then apply sigmoid
    def play_vs_one_opp(opp_params, keys_n):
        half = eval_hands // 2
        # First half: main as P0 - get raw chip deltas for main
        money_p0, _ = jax.vmap(play_one_hand, in_axes=(None, None, 0))(
            main_state.params, opp_params, keys_n[:half]
        )
        # Second half: main as P1 (opp is P0)
        # play_one_hand returns r0 (opp's reward), so main's reward is -r0
        money_opp_as_p0, _ = jax.vmap(play_one_hand, in_axes=(None, None, 0))(
            opp_params, main_state.params, keys_n[half:]
        )
        money_p1 = -money_opp_as_p0  # Main's money when playing as P1

        # Sum total chip delta over the match
        R_total = jnp.sum(money_p0) + jnp.sum(money_p1)

        # EV-soft score: sigmoid(R_total / tau)
        # This gives a score in (0, 1) that reflects the margin of victory
        score = jax.nn.sigmoid(R_total / CFG.elo_tau)
        return score
    
    scores_pool = jax.vmap(play_vs_one_opp, in_axes=(0, 0))(pool.params, keys_grid)
    scores_pool = jnp.where(valid_mask, scores_pool, 0.0)
    
    # Update ELOs
    ea_pool = elo_expected(main_elo, pool.elo)
    eb_pool = 1.0 - ea_pool
    
    delta_main = elo_k * (scores_pool - ea_pool)
    delta_opp = elo_k * ((1.0 - scores_pool) - eb_pool)
    
    main_change = jnp.sum(delta_main * valid_f) / n_valid
    main_elo_next = main_elo + main_change
    
    pool_elo_next = pool.elo + jnp.where(valid_mask, delta_opp, 0.0)
    pool = pool.replace(elo=pool_elo_next)

    # Compute style stats for main agent before adding to pool
    main_style_stats, key = compute_style_stats(main_state.params, key)

    # Add main to pool with style stats
    valid_elos = jnp.where(valid_mask, pool.elo, -1e9)
    _, topk_idx = jax.lax.top_k(valid_elos, k=4)

    k_eff = jnp.minimum(pool.size, jnp.int32(4))
    topk_mask = (jnp.arange(4) < k_eff).astype(jnp.float32)

    topk_vals = pool.elo[topk_idx]
    denom = jnp.maximum(1.0, jnp.sum(topk_mask))
    avg_top_elo = jnp.sum(topk_vals * topk_mask) / denom

    entry_elo = jnp.where(pool.size > 0, avg_top_elo, 0.0)
    pool = pool_add(pool, main_state.params, entry_elo, main_style_stats)

    # =========================================================================
    # Diversity-aware retention: keep pool_max=6 agents
    # - 4 top by ELO (performance)
    # - 2 style-diverse (behavioral diversity)
    # - 2 most recent (recency)
    # Note: Categories can overlap, final count capped at pool_max
    # =========================================================================
    size_curr = pool.size
    valid_mask_final = jnp.arange(pool_max) < size_curr
    valid_f_final = valid_mask_final.astype(jnp.float32)

    # Initialize keep mask (will mark which agents to keep)
    keep_mask = jnp.zeros(pool_max, dtype=jnp.bool_)

    # === Select top 4 by ELO ===
    valid_elos_final = jnp.where(valid_mask_final, pool.elo, -1e9)
    _, elo_top_idx = jax.lax.top_k(valid_elos_final, k=4)
    n_elo_keep = jnp.minimum(size_curr, jnp.int32(4))
    elo_keep_mask = (jnp.arange(4) < n_elo_keep)
    keep_mask = keep_mask.at[elo_top_idx].set(
        keep_mask[elo_top_idx] | elo_keep_mask
    )

    # === Select 2 style-diverse (furthest from mean style, excluding already kept) ===
    # Compute mean style across valid agents
    n_valid_final = jnp.maximum(1.0, jnp.sum(valid_f_final))
    mean_style = jnp.sum(pool.style_stats * valid_f_final[:, None], axis=0) / n_valid_final

    # L2 distance from mean style
    style_dist = jnp.sum((pool.style_stats - mean_style) ** 2, axis=-1)
    # Exclude already-kept agents and invalid slots
    style_dist_masked = jnp.where(valid_mask_final & ~keep_mask, style_dist, -1e9)
    _, diverse_idx = jax.lax.top_k(style_dist_masked, k=2)
    # Only keep if there are remaining valid agents
    n_remaining = jnp.sum((valid_mask_final & ~keep_mask).astype(jnp.int32))
    n_diverse_keep = jnp.minimum(n_remaining, jnp.int32(2))
    diverse_keep_mask = (jnp.arange(2) < n_diverse_keep)
    keep_mask = keep_mask.at[diverse_idx].set(
        keep_mask[diverse_idx] | diverse_keep_mask
    )

    # === Select 2 most recent (by add_order, excluding already kept) ===
    valid_add_order_final = jnp.where(valid_mask_final & ~keep_mask, pool.add_order, -1)
    _, recent_idx = jax.lax.top_k(valid_add_order_final, k=2)
    n_remaining_for_recent = jnp.sum((valid_mask_final & ~keep_mask).astype(jnp.int32))
    n_recent_keep = jnp.minimum(n_remaining_for_recent, jnp.int32(2))
    recent_keep_mask = (jnp.arange(2) < n_recent_keep)
    keep_mask = keep_mask.at[recent_idx].set(
        keep_mask[recent_idx] | recent_keep_mask
    )

    # Gather indices of agents to keep (up to pool_max)
    keep_indices = jnp.where(keep_mask, jnp.arange(pool_max), pool_max)
    keep_indices_sorted = jnp.sort(keep_indices)
    # Filter out the pool_max sentinel values
    n_keep = jnp.sum(keep_mask.astype(jnp.int32))

    # Gather kept agents into compact form
    def _gather_compact(x):
        # Gather from keep_indices_sorted, masking invalid entries
        gathered = x[keep_indices_sorted % pool_max]
        valid = (jnp.arange(pool_max) < n_keep).reshape((-1,) + (1,) * (x.ndim - 1))
        return jnp.where(valid, gathered, jnp.zeros_like(gathered))

    new_params = jax.tree_util.tree_map(_gather_compact, pool.params)
    new_elo = _gather_compact(pool.elo)
    new_add_order = _gather_compact(pool.add_order)
    new_style_stats = _gather_compact(pool.style_stats)

    pool_final = pool.replace(
        params=new_params,
        elo=new_elo,
        add_order=new_add_order,
        style_stats=new_style_stats,
        size=n_keep,
    )

    return pool_final, main_elo_next, key


# =============================================================================
# 10) Main Training Loop
# =============================================================================


def make_update_step():
    """Create JIT-compiled update function for unroll-based training."""

    @jax.jit
    def update_phase(
        ts: TrainState,
        buf: UnrollBuffer,
        key: chex.PRNGKey,
        ent_coef: chex.Array,
    ) -> Tuple[TrainState, Dict[str, chex.Array], chex.PRNGKey]:
        """Run one update phase on sampled unrolls.

        V-trace is computed at training time with proper importance weights.

        Args:
            ts: Training state
            buf: Unroll buffer
            key: Random key
            ent_coef: Current entropy coefficient
        """

        def _epoch_step(carry, _):
            ts, key = carry

            # Sample unrolls from buffer
            # Each unroll has shape (T, E, ...) = (rollout_len, num_envs, ...)
            unrolls, key = unroll_buffer_sample(buf, key, CFG.num_minibatches)

            # Train on each sampled unroll
            def _unroll_step(ts_local, i):
                # Extract single unroll: (T, E, ...)
                unroll_i = jax.tree_util.tree_map(lambda x: x[i], unrolls)
                new_ts, metrics = train_step_on_unroll(ts_local, unroll_i, ent_coef)
                return new_ts, metrics

            ts, metrics_stack = jax.lax.scan(
                _unroll_step,
                ts,
                jnp.arange(CFG.num_minibatches),
            )

            metrics_avg = jax.tree_util.tree_map(lambda x: jnp.mean(x), metrics_stack)
            return (ts, key), metrics_avg

        (ts_new, key_new), metrics_epochs = jax.lax.scan(
            _epoch_step,
            (ts, key),
            None,
            length=CFG.update_epochs,
        )

        final_metrics = jax.tree_util.tree_map(lambda x: jnp.mean(x), metrics_epochs)

        return ts_new, final_metrics, key_new

    return update_phase


def make_train_block(block_size: int):
    """Create JIT-compiled function that runs block_size collect+update steps in a scan.

    This reduces host-device synchronization overhead by keeping the TPU busy
    for multiple steps without returning control to Python.
    """

    @partial(jax.jit, static_argnums=())
    def train_block(
        ts: TrainState,
        pool: AgentPool,
        buf: UnrollBuffer,
        vec: VecEnvState,
        key: chex.PRNGKey,
        base_step: chex.Array,
    ) -> Tuple[TrainState, UnrollBuffer, VecEnvState, chex.PRNGKey, Dict[str, chex.Array]]:
        """Run block_size collect+update iterations.

        Args:
            ts: Training state
            pool: Agent pool (read-only during block, updated outside)
            buf: Unroll buffer
            vec: Vectorized environment state
            key: Random key
            base_step: Starting update step number (for entropy schedule)

        Returns:
            Updated ts, buf, vec, key, and averaged metrics
        """

        def one_step(carry, step_offset):
            ts_inner, buf_inner, vec_inner, key_inner = carry
            upd = base_step + step_offset

            # 1. Collect rollout
            buf_inner, vec_inner, key_inner = collect_rollout(
                ts_inner.params, pool, buf_inner, vec_inner, key_inner
            )

            # 2. Compute entropy coefficient for this step
            ent_coef = get_entropy_coef(upd)

            # 3. Update policy (inline the update_phase logic)
            def _epoch_step(epoch_carry, _):
                ts_e, key_e = epoch_carry
                unrolls, key_e = unroll_buffer_sample(buf_inner, key_e, CFG.num_minibatches)

                def _unroll_step(ts_u, i):
                    unroll_i = jax.tree_util.tree_map(lambda x: x[i], unrolls)
                    new_ts, metrics = train_step_on_unroll(ts_u, unroll_i, ent_coef)
                    return new_ts, metrics

                ts_e, metrics_stack = jax.lax.scan(
                    _unroll_step, ts_e, jnp.arange(CFG.num_minibatches)
                )
                metrics_avg = jax.tree_util.tree_map(jnp.mean, metrics_stack)
                return (ts_e, key_e), metrics_avg

            (ts_inner, key_inner), metrics_epochs = jax.lax.scan(
                _epoch_step, (ts_inner, key_inner), None, length=CFG.update_epochs
            )
            step_metrics = jax.tree_util.tree_map(jnp.mean, metrics_epochs)

            return (ts_inner, buf_inner, vec_inner, key_inner), step_metrics

        (ts_out, buf_out, vec_out, key_out), metrics_all = jax.lax.scan(
            one_step,
            (ts, buf, vec, key),
            jnp.arange(block_size, dtype=jnp.int32),
        )

        # Average metrics over all steps in the block
        metrics_avg = jax.tree_util.tree_map(jnp.mean, metrics_all)

        return ts_out, buf_out, vec_out, key_out, metrics_avg

    return train_block


def main() -> None:
    """Main training loop."""
    print(f"JAX Devices: {jax.devices()}")
    print(f"Configuration: {CFG}")

    # Initialize
    key = jax.random.PRNGKey(CFG.seed)
    key, k_init = jax.random.split(key)

    # Pass total_updates to make_train_state for LR schedule
    ts = make_train_state(k_init, CFG.total_updates)

    # Initialize unroll buffer
    # Capacity is in number of unrolls, not transitions
    # Each unroll contains rollout_len * num_envs transitions
    # Note: With default settings (500k / 262k = 1), min capacity of 16 applies
    transitions_per_unroll = CFG.rollout_len * CFG.num_envs
    unroll_capacity = 4  # Minimum capacity for stability
    actual_transitions = unroll_capacity * transitions_per_unroll
    buf = unroll_buffer_init(unroll_capacity, CFG.rollout_len, CFG.num_envs)
    print(f"Unroll buffer capacity: {unroll_capacity} unrolls ({actual_transitions:,} transitions)")

    # Count parameters
    num_params = sum(x.size for x in jax.tree_util.tree_leaves(ts.params))
    print(f"Model parameters: {num_params:,}")

    # Initialize checkpointing
    ckpt_dir = os.path.abspath(CFG.checkpoint_dir)
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpter = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(ts)

    # Restore latest checkpoint if available (params only; fresh optimizer state)
    ckpt_steps = []
    for name in os.listdir(ckpt_dir):
        full = os.path.join(ckpt_dir, name)
        if os.path.isdir(full) and name.startswith("step_"):
            try:
                step = int(name.split("_", 1)[1])
                ckpt_steps.append(step)
            except ValueError:
                pass
    if ckpt_steps:
        latest_step = max(ckpt_steps)
        restore_path = os.path.join(ckpt_dir, f"step_{latest_step}")
        restored = ckpter.restore(restore_path)
        restored_params = (
            restored["params"]
            if isinstance(restored, dict) and "params" in restored
            else getattr(restored, "params", restored)
        )
        ts = ts.replace(params=restored_params)
        print(f"Restored checkpoint params from {restore_path}")

    # Initialize pool with initial agent (compute style stats for diversity tracking)
    pool = pool_init(ts.params)
    key, k_style = jax.random.split(key)
    init_style_stats, _ = compute_style_stats(ts.params, k_style)
    pool = pool_add(pool, ts.params, jnp.float32(0.0), init_style_stats)
    main_elo = jnp.float32(0.0)

    # Initialize environment
    key, k_reset = jax.random.split(key)
    vec, key = vec_reset(k_reset, CFG.num_envs)

    # Compile functions
    print("Compiling update functions...")
    do_collect = jax.jit(collect_rollout)
    do_train_block = make_train_block(CFG.block_size)

    # Warmup unroll buffer
    print("Warming up unroll buffer...")
    warm_updates = max(1, CFG.warmup_steps // (CFG.num_envs * CFG.rollout_len))
    for _ in range(warm_updates):
        buf, vec, key = do_collect(ts.params, pool, buf, vec, key)
    print(f"Buffer size after warmup: {buf.size} unrolls")

    # Training loop (block-based to reduce host-device sync overhead)
    print("Starting training...")
    import time
    start_time = time.time()

    # Compute number of blocks
    total_blocks = CFG.total_updates // CFG.block_size
    assert CFG.total_updates % CFG.block_size == 0, (
        f"total_updates ({CFG.total_updates}) must be divisible by block_size ({CFG.block_size})"
    )
    # Ensure pool/log/checkpoint intervals align with block_size
    assert CFG.log_every % CFG.block_size == 0 or CFG.block_size % CFG.log_every == 0, (
        f"log_every ({CFG.log_every}) should align with block_size ({CFG.block_size})"
    )

    for block_idx in range(total_blocks):
        base_step = block_idx * CFG.block_size
        end_step = base_step + CFG.block_size

        # Run block_size steps in a single JIT call (reduces host-device sync by ~block_size×)
        ts, buf, vec, key, metrics = do_train_block(
            ts, pool, buf, vec, key, jnp.int32(base_step)
        )

        # Pool maintenance (check if any step in this block triggers it)
        # Pool update happens after step N where (N+1) % add_to_pool_every_updates == 0
        # i.e., steps 99, 199, 299, ... (0-indexed)
        pool_update_steps_in_block = [
            s for s in range(base_step, end_step)
            if (s + 1) % CFG.add_to_pool_every_updates == 0
        ]
        if pool_update_steps_in_block:
            pool, main_elo, key = eval_and_update_pool(pool, ts, main_elo, key)

        # Logging (at end of each block if block aligns with log_every)
        if end_step % CFG.log_every == 0:
            m = jax.device_get(metrics)

            elapsed = time.time() - start_time
            steps_since_log = CFG.block_size  # We log every block when block_size == log_every
            sps = (steps_since_log * CFG.num_envs * CFG.rollout_len) / elapsed
            start_time = time.time()

            print(
                f"Upd {end_step}/{CFG.total_updates} | "
                f"Buf: {int(buf.size)} | Pool: {int(pool.size)} | "
                f"ELO: {float(main_elo):.1f} | "
                f"SPS: {sps:.0f} | "
                f"Loss: {m['loss']:.4f} | "
                f"Pol: {m['policy_loss']:.4f} | "
                f"Val: {m['value_loss']:.4f} | "
                f"Ent: {m['entropy']:.4f} | "
                f"EntC: {float(m['ent_coef']):.4f} | "
                f"rho: {m['rho_mean']:.2f}"
            )

        # Checkpointing
        if end_step % CFG.checkpoint_every == 0:
            save_path = f"{ckpt_dir}/step_{end_step}"
            if os.path.exists(save_path):
                shutil.rmtree(save_path)
            ckpter.save(save_path, ts, save_args=save_args)
            print(f"Saved checkpoint to {save_path}")

    print("Training complete!")


if __name__ == "__main__":
    main()