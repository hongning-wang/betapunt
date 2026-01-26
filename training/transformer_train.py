"""Transformer-based IMPALA V-trace training for Toss-or-Hold'em (JAX/Flax).

This training script implements IMPALA-style training:
  1) Transformer-based card/action encoding
  2) Pure V-trace off-policy correction computed at TRAINING TIME
  3) Short replay window storing complete unrolls for proper V-trace computation
  4) K-Best self-play league training with ELO-based opponent selection
  5) Opponent history encoding for exploitation (NEW)

Key architectural choices:
  - Card tensor: (B, 9, 2) - [card_index, type_id] matching model.py CardEncoder
  - Action tensor: (B, 32, 8) - action history matching model.py ActionEncoder (includes is_dealer)
  - History tensor: (B, 64) - opponent statistics for exploitation
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
from typing import Any, Dict, Tuple

import chex
import distrax
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
    ACTION_INPUT_DIM,  # Now 8 (includes is_dealer)
    HISTORY_INPUT_DIM,  # 64 opponent history features
)

# Import opponent stats tracking
from opponent_stats import (
    VecOpponentStats,
    vec_opponent_stats_init,
    vec_opponent_stats_to_tensor,
    NUM_FREQ_STATS,
    NUM_AVG_STATS,
    FREQ_IDX_VPIP_DEALER,
    FREQ_IDX_VPIP_BB,
    FREQ_IDX_PREFLOP_RAISE_DEALER,
    FREQ_IDX_PREFLOP_RAISE_BB,
    FREQ_IDX_PREFLOP_FOLD_DEALER,
    FREQ_IDX_PREFLOP_FOLD_BB,
    FREQ_IDX_DISCARD_HIGH_DEALER,
    FREQ_IDX_DISCARD_HIGH_BB,
    FREQ_IDX_FLOP_FOLD_DEALER,
    FREQ_IDX_FLOP_FOLD_BB,
    FREQ_IDX_FLOP_CALL_DEALER,
    FREQ_IDX_FLOP_CALL_BB,
    FREQ_IDX_FLOP_RAISE_DEALER,
    FREQ_IDX_FLOP_RAISE_BB,
    FREQ_IDX_FLOP_BET_DEALER,
    FREQ_IDX_FLOP_BET_BB,
    FREQ_IDX_FLOP_CHECK_RAISE,
    FREQ_IDX_FLOP_CBET_DEALER,
    FREQ_IDX_FLOP_CBET_BB,
    FREQ_IDX_FLOP_DONK_BB,
    FREQ_IDX_TURN_FOLD_DEALER,
    FREQ_IDX_TURN_FOLD_BB,
    FREQ_IDX_TURN_CALL_DEALER,
    FREQ_IDX_TURN_CALL_BB,
    FREQ_IDX_TURN_RAISE_DEALER,
    FREQ_IDX_TURN_RAISE_BB,
    FREQ_IDX_TURN_BET_DEALER,
    FREQ_IDX_TURN_BET_BB,
    FREQ_IDX_TURN_DOUBLE_BARREL_DEALER,
    FREQ_IDX_TURN_DOUBLE_BARREL_BB,
    FREQ_IDX_RIVER_FOLD_DEALER,
    FREQ_IDX_RIVER_FOLD_BB,
    FREQ_IDX_RIVER_CALL_DEALER,
    FREQ_IDX_RIVER_CALL_BB,
    FREQ_IDX_RIVER_RAISE_DEALER,
    FREQ_IDX_RIVER_RAISE_BB,
    FREQ_IDX_RIVER_BET_DEALER,
    FREQ_IDX_RIVER_BET_BB,
    FREQ_IDX_RIVER_TRIPLE_BARREL_DEALER,
    FREQ_IDX_RIVER_TRIPLE_BARREL_BB,
    FREQ_IDX_RIVER_OVERBET_DEALER,
    FREQ_IDX_RIVER_OVERBET_BB,
    FREQ_IDX_SHOWDOWN_WIN,
    FREQ_IDX_ALLIN_SHOWDOWN_WIN,
    FREQ_IDX_ALLIN_DEALER,
    FREQ_IDX_ALLIN_BB,
    FREQ_IDX_ALLIN_CALL_DEALER,
    FREQ_IDX_ALLIN_CALL_BB,
    FREQ_IDX_WENT_SHOWDOWN_DEALER,
    FREQ_IDX_WENT_SHOWDOWN_BB,
    AVG_IDX_OPEN_RAISE_SIZE_DEALER,
    AVG_IDX_OPEN_RAISE_SIZE_BB,
    AVG_IDX_FLOP_BET_SIZE_DEALER,
    AVG_IDX_FLOP_BET_SIZE_BB,
    AVG_IDX_TURN_BET_SIZE_DEALER,
    AVG_IDX_TURN_BET_SIZE_BB,
    AVG_IDX_RIVER_BET_SIZE_DEALER,
    AVG_IDX_RIVER_BET_SIZE_BB,
    AVG_IDX_STREETS_SEEN,
)

# --- Environment ---
from game import (
    TossHoldemEnv,
    State,
    STACK_SIZE,
    PHASE_PREFLOP,
    PHASE_DISCARD_OOP,
    PHASE_DISCARD_DLR,
    PHASE_FLOP,
    PHASE_TURN,
    PHASE_RIVER,
    PHASE_SHOWDOWN,
    ACT_FOLD,
    ACT_CHECK,
    ACT_CALL,
    ACT_RAISE_MIN,
    ACT_RAISE_33,
    ACT_RAISE_50,
    ACT_RAISE_75,
    ACT_RAISE_125,
    ACT_RAISE_200,
    ACT_ALLIN,
    ACT_DISCARD_0,
    ACT_DISCARD_1,
    ACT_DISCARD_2,
    BB,
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
    vf_coef: float = 0.25

    # Entropy coefficient with schedule
    ent_coef_init: float = 0.01
    ent_coef_final: float = 0.001
    ent_decay_steps: int = 40_000

    # Illegal action penalty - penalizes high logits for illegal actions
    illegal_action_coef: float = 0.1

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
    recent_window_transitions: int = 500_000  # Sample only from recent N transitions (0 = no limit)

    # League training (K-best self-play)
    pool_max: int = 8
    k_best: int = 4  # Number of top agents to keep by ELO
    eval_hands: int = 512
    add_to_pool_every_updates: int = 100
    elo_k: float = 32.0
    elo_tau: float = 100.0  # Temperature for sigmoid scoring in chip units (BB scale)

    # Runtime
    seed: int = 0
    total_updates: int = 40_000
    block_size: int = 100  # Number of collect+update steps per scan block (reduces host-device sync)

    # Checkpointing
    checkpoint_dir: str = "./drive/MyDrive/checkpoints"
    checkpoint_every: int = 1000
    log_every: int = 50

    # Resume from checkpoint (weights only, fresh optimizer state)
    resume_weights_from: str = ""  # e.g., "./drive/MyDrive/checkpoints/step_40000"


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
    """Build an 8-field action token from the current state and action.

    time_id is a monotonically increasing index (global action count).
    
    Fields:
        [0]: actor_id (0 or 1)
        [1]: event_type (0-12)
        [2]: street_id (0-4)
        [3]: time_id
        [4]: to_call
        [5]: pot_base
        [6]: effective_stack
        [7]: is_dealer (NEW)
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
    
    # NEW: Determine if current player is dealer
    current_is_dealer = jnp.where(is_p0, state.is_dealer_p0, ~state.is_dealer_p0)
    is_dealer = current_is_dealer.astype(jnp.float32)

    token = jnp.array(
        [
            state.current_player.astype(jnp.float32),
            action.astype(jnp.float32),
            street_id,
            time_id.astype(jnp.float32),
            to_call,
            pot_base,
            eff_cap,
            is_dealer,  # NEW: index 7
        ],
        dtype=jnp.float32,
    )
    return token



@chex.dataclass(frozen=True)
class AuxState:
    """Extra per-env state to build action history tensors.

    Action tensor shape: (ACTION_SEQ_LEN, ACTION_INPUT_DIM) = (32, 8)
    Each action entry contains (matching model.py ActionEncoder):
        [0] actor_id          int {0,1}
        [1] event_type        int {0..12}
        [2] street_id         int {0..4}
        [3] time_id           int (monotonic action counter)
        [4] to_call           float
        [5] pot_base          float (pot after calling, for raise sizing)
        [6] effective_stack   float
        [7] is_dealer         int {0,1} (NEW)
    """
    action_tensor: chex.Array  # (ACTION_SEQ_LEN, ACTION_INPUT_DIM) = (32, 8)
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

    Uses make_action_token to build the 8-feature action entry matching model.py:
      [0]: actor_id (0 or 1)
      [1]: event_type (0-12)
      [2]: street_id (0-4)
      [3]: time_id (monotonic action counter)
      [4]: to_call
      [5]: pot_base
      [6]: effective_stack
      [7]: is_dealer (NEW)
    """
    # Build the 8-feature action token using existing make_action_token
    action_entry = make_action_token(state, action, aux.action_ptr)

    # Write to action tensor (circular buffer behavior if > ACTION_SEQ_LEN actions)
    write_idx = aux.action_ptr % ACTION_SEQ_LEN
    action_tensor = aux.action_tensor.at[write_idx].set(action_entry)

    return aux.replace(
        action_tensor=action_tensor,
        action_ptr=aux.action_ptr + 1,
    )


def _canonize_suits(cards: chex.Array) -> chex.Array:
    """Canonize suits based on order of first appearance.

    The first suit seen becomes 0, second becomes 1, etc.
    This ensures suit-isomorphic hands have identical representations.

    Args:
        cards: (N,) array of card indices (0-51), -1 for invalid

    Returns:
        (N,) array of card indices with canonized suits
    """
    valid = cards >= 0
    ranks = jnp.where(valid, cards % 13, 0).astype(jnp.int32)
    suits = jnp.where(valid, cards // 13, 0).astype(jnp.int32)

    def scan_fn(carry, x):
        suit, is_valid = x
        mapping, next_canonical = carry

        # Check if we've seen this suit before
        current_canonical = mapping[suit]
        is_new = is_valid & (current_canonical < 0)

        # Update mapping if this is a new suit
        new_mapping = jnp.where(is_new, mapping.at[suit].set(next_canonical), mapping)
        new_next = jnp.where(is_new, next_canonical + 1, next_canonical)

        return (new_mapping, new_next), None

    init = (jnp.full(4, -1, dtype=jnp.int32), jnp.int32(0))
    (final_mapping, _), _ = jax.lax.scan(scan_fn, init, (suits, valid))

    # Apply mapping: canonical_suit * 13 + rank
    canonical_suits = jnp.take(final_mapping, suits, axis=0)
    new_cards = jnp.where(valid, canonical_suits * 13 + ranks, -1)
    return new_cards


@jax.jit
def encode_cards(state: State, player_id: chex.Array) -> chex.Array:
    """Encode cards into (9, 2) tensor matching model.py CardEncoder format.

    Output format: (CARD_SEQ_LEN, CARD_INPUT_DIM) = (9, 2)
    Each card entry: [card_index, type_id]
      - card_index: 0-51 valid cards (with canonized suits), -1 for undealt/invalid
      - type_id: 0-5 for slot type, -1 for invalid

    Cards are sorted within groups (hero hand, flop) by rank.
    Suits are canonized based on order of first appearance.

    Card positions and type_ids:
      0-2: hero private cards (sorted by rank, type_id=0)
      3-4: flop cards (sorted by rank, type_id=1)
      5: herodiscard (type_id=2)
      6: villaindiscard (type_id=3)
      7: turn (type_id=4)
      8: river (type_id=5)
    """
    is_p0 = player_id == 0

    # Get hole cards for the viewing player (hero)
    hero_hole = jnp.where(is_p0, state.p0_hole, state.p1_hole)  # (3,)
    hero_active = jnp.where(is_p0, state.p0_active_mask, state.p1_active_mask)  # (3,)

    # Sort hero cards by rank (primary), then suit (secondary)
    hero_rank = hero_hole % 13
    hero_suit = hero_hole // 13
    hero_sort_keys = jnp.where(hero_active, hero_rank * 4 + hero_suit, jnp.int32(999))
    hero_sorted_idx = jnp.argsort(hero_sort_keys)
    hero_sorted = hero_hole[hero_sorted_idx]
    hero_active_sorted = hero_active[hero_sorted_idx]

    # Determine discard indices based on dealer position
    # OOP discards to board[2], dealer discards to board[3]
    hero_is_dealer = jnp.where(is_p0, state.is_dealer_p0, jnp.logical_not(state.is_dealer_p0))
    hero_discard_idx = jnp.where(hero_is_dealer, 3, 2)
    villain_discard_idx = jnp.where(hero_is_dealer, 2, 3)

    # Sort flop cards by rank (primary), then suit (secondary)
    flop_cards = jnp.array([state.board[0], state.board[1]], dtype=jnp.int32)
    flop_valid = flop_cards >= 0
    flop_rank = flop_cards % 13
    flop_suit = flop_cards // 13
    flop_sort_keys = jnp.where(flop_valid, flop_rank * 4 + flop_suit, jnp.int32(999))
    flop_sorted_idx = jnp.argsort(flop_sort_keys)
    flop_sorted = flop_cards[flop_sorted_idx]

    # Build all card indices in order (for suit canonization)
    # Order: hero cards (sorted), flop (sorted), hero_discard, villain_discard, turn, river
    card_indices_raw = jnp.array([
        jnp.where(hero_active_sorted[0], hero_sorted[0], -1),
        jnp.where(hero_active_sorted[1], hero_sorted[1], -1),
        jnp.where(hero_active_sorted[2], hero_sorted[2], -1),
        flop_sorted[0],
        flop_sorted[1],
        state.board[hero_discard_idx],
        state.board[villain_discard_idx],
        state.board[4],  # turn
        state.board[5],  # river
    ], dtype=jnp.int32)

    # Canonize suits based on order of first appearance
    card_indices = _canonize_suits(card_indices_raw)

    # Type IDs: simplified to 6 types
    # 0 = hero private, 1 = flop, 2 = hero discard, 3 = villain discard, 4 = turn, 5 = river
    type_ids_raw = jnp.array([0, 0, 0, 1, 1, 2, 3, 4, 5], dtype=jnp.int32)
    valid_mask = jnp.array([
        hero_active_sorted[0],
        hero_active_sorted[1],
        hero_active_sorted[2],
        card_indices[3] >= 0,
        card_indices[4] >= 0,
        card_indices[5] >= 0,
        card_indices[6] >= 0,
        card_indices[7] >= 0,
        card_indices[8] >= 0,
    ])
    type_ids = jnp.where(valid_mask, type_ids_raw, -1)

    # Stack into (9, 2) tensor
    card_tensor = jnp.stack([card_indices, type_ids], axis=-1)

    return card_tensor


@jax.jit
def encode_actions(aux: AuxState, player_id: chex.Array) -> Tuple[chex.Array, chex.Array]:
    """Encode action history into (ACTION_SEQ_LEN, ACTION_INPUT_DIM) tensor with ego-relative player IDs.

    Returns:
        action_tensor: (ACTION_SEQ_LEN, ACTION_INPUT_DIM) = (32, 8) action history
        action_mask: (ACTION_SEQ_LEN,) = (32,) boolean mask for valid actions (pointer-based)
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
    opp_history: chex.Array = None,
) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]:
    """Build card/action tensors + legal mask + history from state.

    Returns:
        card_tensor: (CARD_SEQ_LEN, CARD_INPUT_DIM) = (9, 2) int32
        action_tensor: (ACTION_SEQ_LEN, ACTION_INPUT_DIM) = (32, 8) float32
        action_mask: (ACTION_SEQ_LEN,) = (32,) bool
        legal_mask: (NUM_ACTIONS,) = (13,) bool
        history_tensor: (HISTORY_INPUT_DIM,) = (64,) float32
    """
    card_tensor = encode_cards(state, player_id)
    action_tensor, action_mask = encode_actions(aux, player_id)
    legal_mask = ENV.get_legal_actions(state)
    
    # Use provided history or default to neutral
    if opp_history is None:
        history_tensor = jnp.full(HISTORY_INPUT_DIM, 0.5, dtype=jnp.float32)
    else:
        history_tensor = opp_history
    
    return card_tensor, action_tensor, action_mask, legal_mask, history_tensor


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
    action_hist: chex.Array # (T, E, 32, 8) float32 - action history
    action_mask: chex.Array # (T, E, 32) bool - action history mask
    history: chex.Array     # (T, E, 64) float32 - opponent history features (NEW)
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
    bootstrap_action_hist: chex.Array # (E, 32, 8) - action history for s_T
    bootstrap_action_mask: chex.Array # (E, 32) - action mask for s_T
    bootstrap_history: chex.Array     # (E, 64) - history tensor for s_T (NEW)
    bootstrap_done: chex.Array        # (E,) - whether last transition was terminal


@chex.dataclass(frozen=True)
class UnrollBuffer:
    """Buffer storing complete unrolls for V-trace computation.

    Each entry is a full trajectory of shape (rollout_len, num_envs, ...).
    This enables proper importance weight computation across time.
    """
    card: chex.Array        # (cap, T, E, 9, 2)
    action_hist: chex.Array # (cap, T, E, 32, 8)
    action_mask: chex.Array # (cap, T, E, 32)
    history: chex.Array     # (cap, T, E, 64) - NEW
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
    bootstrap_action_hist: chex.Array # (cap, E, 32, 8)
    bootstrap_action_mask: chex.Array # (cap, E, 32)
    bootstrap_history: chex.Array     # (cap, E, 64) - NEW
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
        history=jnp.zeros((capacity, T, E, HISTORY_INPUT_DIM), dtype=jnp.float32),  # NEW
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
        bootstrap_history=jnp.zeros((capacity, E, HISTORY_INPUT_DIM), dtype=jnp.float32),  # NEW
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
        history=buf.history.at[idx].set(unroll.history),  # NEW
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
        bootstrap_history=buf.bootstrap_history.at[idx].set(unroll.bootstrap_history),  # NEW
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
    """Sample batch_size unrolls from the buffer.

    Returns unrolls with shape (batch_size, T, E, ...).
    """
    key, sub = jax.random.split(key)
    cap = buf.card.shape[0]

    # Use recent window for sampling (convert transitions to unrolls)
    # Each unroll contains rollout_len * num_envs transitions
    # Use ceiling division to ensure at least 1 unroll when window > 0
    transitions_per_unroll = CFG.rollout_len * CFG.num_envs
    # Ceiling division: (a + b - 1) // b, with minimum of 1
    recent_window_unrolls = jnp.where(
        CFG.recent_window_transitions > 0,
        jnp.maximum(1, jnp.int32((CFG.recent_window_transitions + transitions_per_unroll - 1) // transitions_per_unroll)),
        cap,
    )
    available = jnp.minimum(buf.size, recent_window_unrolls)

    # Sample from available unrolls
    age = (buf.ptr - 1 - jnp.arange(cap, dtype=jnp.int32)) % cap
    eligible = jnp.where(available > 0, age < available, jnp.zeros_like(age, dtype=jnp.bool_))
    elig_w = eligible.astype(jnp.float32)
    elig_sum = jnp.sum(elig_w)

    def _sample():
        p = jnp.where(
            elig_sum > 0,
            elig_w / elig_sum,
            jnp.ones((cap,), dtype=jnp.float32) / jnp.float32(cap)
        )
        return jax.random.choice(sub, cap, shape=(batch_size,), replace=True, p=p)

    idx = _sample()

    unroll = Unroll(
        card=buf.card[idx],
        action_hist=buf.action_hist[idx],
        action_mask=buf.action_mask[idx],
        history=buf.history[idx],  # NEW
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
        bootstrap_history=buf.bootstrap_history[idx],  # NEW
        bootstrap_done=buf.bootstrap_done[idx],
    )
    return unroll, key


# =============================================================================
# 4) K-Best Self-Play Pool
# =============================================================================


@chex.dataclass(frozen=True)
class AgentPool:
    """Pool of historical agent parameters for self-play."""
    params: Any              # Stacked params along axis 0
    elo: chex.Array          # (pool_max,) ELO ratings
    style_stats: chex.Array  # (pool_max, 3) style statistics [raise_rate, fold_rate, call_rate]
    add_order: chex.Array    # (pool_max,) incremental counter for recency tracking
    size: chex.Array         # Current number of agents


def pool_init(dummy_params: Any) -> AgentPool:
    """Initialize empty agent pool."""
    def _zeros_like(p):
        return jnp.zeros((CFG.pool_max,) + p.shape, dtype=p.dtype)

    params = jax.tree_util.tree_map(_zeros_like, dummy_params)
    elo = jnp.zeros((CFG.pool_max,), dtype=jnp.float32)
    style_stats = jnp.zeros((CFG.pool_max, 3), dtype=jnp.float32)
    add_order = jnp.zeros((CFG.pool_max,), dtype=jnp.int32)
    return AgentPool(params=params, elo=elo, style_stats=style_stats, add_order=add_order, size=jnp.int32(0))


@jax.jit
def pool_add(pool: AgentPool, new_params: Any, initial_elo: chex.Array, new_style_stats: chex.Array) -> AgentPool:
    """Add a new agent to the pool."""
    size = pool.size
    can_append = size < CFG.pool_max

    # If pool is full, replace worst agent
    worst = jnp.argmin(jnp.where(jnp.arange(CFG.pool_max) < size, pool.elo, 1e9))
    write_idx = jnp.where(can_append, size, worst)

    def _write_leaf(pool_leaf, new_leaf):
        return pool_leaf.at[write_idx].set(new_leaf)

    params = jax.tree_util.tree_map(_write_leaf, pool.params, new_params)
    elo = pool.elo.at[write_idx].set(initial_elo)
    style_stats = pool.style_stats.at[write_idx].set(new_style_stats)
    # add_order tracks insertion order: max existing + 1
    next_order = jnp.max(pool.add_order) + 1
    add_order = pool.add_order.at[write_idx].set(next_order)
    size2 = jnp.where(can_append, size + 1, size)

    return pool.replace(params=params, elo=elo, style_stats=style_stats, add_order=add_order, size=size2)


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
    - 60% from top 4 by ELO (hard opponents)
    - 10% from 2 most style-diverse agents
    - 30% from 2 most recently added agents

    This prevents monoculture while maintaining challenge.
    """
    key, k_category, k_top, k_diverse, k_recent = jax.random.split(key, 5)
    size = pool.size
    valid_mask = jnp.arange(CFG.pool_max) < size

    # Category selection: 0=top, 1=diverse, 2=recent
    # Probabilities: 60%, 10%, 30%
    category = jax.random.choice(k_category, 3, p=jnp.array([0.60, 0.10, 0.30]))

    # --- Category 0: Top K by ELO ---
    elo_with_invalid = jnp.where(valid_mask, pool.elo, -1e9)
    top_k = jnp.minimum(CFG.k_best, size)

    # Get indices of top K by ELO
    top_indices = jnp.argsort(elo_with_invalid)[::-1][:CFG.k_best]
    top_mask = jnp.zeros(CFG.pool_max, dtype=jnp.bool_).at[top_indices].set(True)
    top_mask = top_mask & valid_mask

    # Sample uniformly from top K
    top_probs = jnp.where(top_mask, 1.0, 0.0)
    top_probs = top_probs / jnp.maximum(1.0, jnp.sum(top_probs))
    idx_top = jax.random.choice(k_top, CFG.pool_max, p=top_probs)

    # --- Category 1: Most style-diverse ---
    # Compute style diversity as variance from mean style
    mean_style = jnp.sum(pool.style_stats * valid_mask[:, None], axis=0) / jnp.maximum(1.0, size.astype(jnp.float32))
    style_dist = jnp.sum((pool.style_stats - mean_style) ** 2, axis=1)
    style_dist = jnp.where(valid_mask, style_dist, -1e9)

    # Get indices of top 2 most diverse
    diverse_indices = jnp.argsort(style_dist)[::-1][:2]
    diverse_mask = jnp.zeros(CFG.pool_max, dtype=jnp.bool_).at[diverse_indices].set(True)
    diverse_mask = diverse_mask & valid_mask

    diverse_probs = jnp.where(diverse_mask, 1.0, 0.0)
    diverse_probs = diverse_probs / jnp.maximum(1.0, jnp.sum(diverse_probs))
    idx_diverse = jax.random.choice(k_diverse, CFG.pool_max, p=diverse_probs)

    # --- Category 2: Most recently added ---
    add_order_valid = jnp.where(valid_mask, pool.add_order, -1)
    recent_indices = jnp.argsort(add_order_valid)[::-1][:2]
    recent_mask = jnp.zeros(CFG.pool_max, dtype=jnp.bool_).at[recent_indices].set(True)
    recent_mask = recent_mask & valid_mask

    recent_probs = jnp.where(recent_mask, 1.0, 0.0)
    recent_probs = recent_probs / jnp.maximum(1.0, jnp.sum(recent_probs))
    idx_recent = jax.random.choice(k_recent, CFG.pool_max, p=recent_probs)

    # Select based on category
    idx = jax.lax.switch(category, [
        lambda: idx_top,
        lambda: idx_diverse,
        lambda: idx_recent,
    ])

    # Extract params at selected index
    selected_params = jax.tree_util.tree_map(lambda p: p[idx], pool.params)

    return selected_params, key


# =============================================================================
# 5) Terminal Rewards
# =============================================================================


@jax.jit
def terminal_rewards(state: State) -> Tuple[chex.Array, chex.Array]:
    """Compute terminal rewards when game is done.

    Returns rewards relative to starting stack (400).
    Positive = won chips, negative = lost chips.
    """
    # Reward = final stack - starting stack
    r0 = (state.p0_stack - STACK_SIZE).astype(jnp.float32)
    r1 = (state.p1_stack - STACK_SIZE).astype(jnp.float32)

    # Only meaningful when done
    r0 = jnp.where(state.done, r0, jnp.float32(0.0))
    r1 = jnp.where(state.done, r1, jnp.float32(0.0))

    return r0, r1


# =============================================================================
# 6) Value Target Clipping
# =============================================================================


@jax.jit
def clip_value_target(
    v_target: chex.Array,
    contrib_self: chex.Array,
    contrib_opp: chex.Array,
) -> chex.Array:
    """Clip value target to feasible range based on chip commitments.

    The value estimate cannot exceed what we can win or lose given
    current chip commitments.

    Args:
        v_target: V-trace value target (normalized by STACK_SIZE)
        contrib_self: Chips we've committed (unnormalized)
        contrib_opp: Chips opponent has committed (unnormalized)

    Returns:
        Clipped value target (normalized)
    """
    # Max win: opponent's contribution (we take their chips)
    # Max loss: our contribution (we lose our chips)
    max_win = contrib_opp.astype(jnp.float32) / STACK_SIZE
    max_loss = -contrib_self.astype(jnp.float32) / STACK_SIZE

    return jnp.clip(v_target, max_loss, max_win)


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

    # Dummy inputs for initialization (updated dimensions)
    dummy_card = jnp.zeros((1, CARD_SEQ_LEN, CARD_INPUT_DIM), dtype=jnp.int32)
    dummy_action = jnp.zeros((1, ACTION_SEQ_LEN, ACTION_INPUT_DIM), dtype=jnp.float32)  # Now 8
    dummy_mask = jnp.ones((1, ACTION_SEQ_LEN), dtype=jnp.bool_)
    dummy_history = jnp.zeros((1, HISTORY_INPUT_DIM), dtype=jnp.float32)  # NEW

    variables = model.init(rng, dummy_card, dummy_action, dummy_mask, dummy_history)

    # Use LR schedule with warmup and cosine decay
    lr_schedule = make_lr_schedule(total_updates)
    tx = optax.chain(
        optax.clip_by_global_norm(CFG.max_grad_norm),
        optax.adam(lr_schedule),
    )

    ts = TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=tx,
    )

    # Optionally load weights from a checkpoint (fresh optimizer state)
    if CFG.resume_weights_from:
        ckpter = orbax.checkpoint.PyTreeCheckpointer()
        restored = ckpter.restore(CFG.resume_weights_from)
        print(f"Loaded weights from {CFG.resume_weights_from}")
        # Replace params only, keep fresh optimizer state
        ts = ts.replace(params=restored["params"])

    return ts


# Create model instance once at module level to avoid reinstantiation inside JIT
_FORWARD_MODEL = TossHoldemModel()


@jax.jit
def forward_apply(
    params: Any,
    card: chex.Array,
    action_hist: chex.Array,
    action_mask: chex.Array,
    history: chex.Array = None,
) -> Tuple[chex.Array, chex.Array]:
    """Forward pass through the model.

    Uses the module-level _FORWARD_MODEL instance to avoid reinstantiating
    the model inside JIT, which is a Flax anti-pattern.

    Args:
        params: Model parameters
        card: (B, 9, 2) card tensor
        action_hist: (B, 32, 8) action history
        action_mask: (B, 32) action mask
        history: (B, 64) opponent history features (optional)

    Returns:
        logits: (B, NUM_ACTIONS) action logits
        value: (B,) value estimates
    """
    logits, value = _FORWARD_MODEL.apply(
        {"params": params},
        card,
        action_hist,
        action_mask,
        history,
    )
    return logits, value


def masked_categorical(logits: chex.Array, legal_mask: chex.Array) -> distrax.Categorical:
    """Create a categorical distribution with illegal actions masked out."""
    neg_inf = -1e9
    masked_logits = jnp.where(legal_mask, logits, neg_inf)
    return distrax.Categorical(logits=masked_logits)


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
    flat_history = unroll.history.reshape(T * E, *unroll.history.shape[2:])  # NEW
    flat_legal = unroll.legal.reshape(T * E, *unroll.legal.shape[2:])
    flat_action = unroll.action.reshape(T * E)

    # OPTIMIZATION: Single forward pass for both main batch and bootstrap
    # Concatenate main (T*E,) with bootstrap (E,) -> (T*E + E,)
    all_card = jnp.concatenate([flat_card, unroll.bootstrap_card], axis=0)
    all_action_hist = jnp.concatenate([flat_action_hist, unroll.bootstrap_action_hist], axis=0)
    all_action_mask = jnp.concatenate([flat_action_mask, unroll.bootstrap_action_mask], axis=0)
    all_history = jnp.concatenate([flat_history, unroll.bootstrap_history], axis=0)  # NEW

    # Single forward pass for all samples
    all_logits, all_values = forward_apply(params, all_card, all_action_hist, all_action_mask, all_history)

    # Split outputs back into main and bootstrap
    logits = all_logits[:T * E]
    values_flat = all_values[:T * E]
    bootstrap_values = all_values[T * E:]  # (E,)

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

    # Bootstrap value: V(s_T) - zero out where the last transition was terminal
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
    # Also return raw logits for illegal action penalty
    return logp_pi_flat, values_flat, vs, advantages, pg_rhos, masked_logits, flat_legal, logits


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
    logp_pi_flat, values_flat, vs, advantages, pg_rhos, masked_logits, flat_legal, raw_logits = compute_vtrace_on_unroll(params, unroll)

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

    # Illegal action penalty - penalize high logits for illegal actions
    # This teaches the model to not prefer actions that are illegal
    illegal_mask = ~flat_legal  # True where action is illegal
    # Use softplus(logits) to penalize positive logits for illegal actions
    illegal_logits = jnp.where(illegal_mask, jax.nn.softplus(raw_logits), 0.0)
    illegal_penalty_per_sample = jnp.sum(illegal_logits, axis=-1)  # Sum over actions
    illegal_penalty = jnp.sum(illegal_penalty_per_sample * valid_f) / denom

    # Total loss
    total = policy_loss + CFG.vf_coef * value_loss - ent_coef * entropy + CFG.illegal_action_coef * illegal_penalty

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
        "illegal_penalty": illegal_penalty,
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
    opp_stats: VecOpponentStats  # NEW: Opponent statistics for history encoding


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
    
    # NEW: Initialize opponent stats
    opp_stats = vec_opponent_stats_init(n)

    return VecEnvState(
        env_state=s, 
        aux=aux, 
        main_is_p0=main_is_p0, 
        rng=ks,
        opp_stats=opp_stats,
    ), key


@partial(jax.jit, static_argnums=(5,))
def policy_act(
    params: Any,
    state: State,
    aux: AuxState,
    key: chex.PRNGKey,
    history: chex.Array = None,
    deterministic: bool = False,
) -> Tuple[chex.Array, chex.Array, chex.Array, chex.PRNGKey]:
    """Select action for the current player.

    Args:
        params: Model parameters
        state: Current game state
        aux: Auxiliary state for action history
        key: Random key
        history: (64,) opponent history features (optional)
        deterministic: If True, use argmax (for stable ELO evaluation).
                       If False, use Gumbel-argmax sampling (for training).

    Returns:
        action: Selected action
        logp: Log probability of action
        value: Value estimate
        key: Updated random key
    """
    pid = state.current_player.astype(jnp.int32)
    card, action_hist, action_mask, legal, _ = encode_obs(state, aux, pid, history)

    # Add batch dimension
    card_b = card[None, ...]
    action_hist_b = action_hist[None, ...]
    action_mask_b = action_mask[None, ...]
    history_b = history[None, ...] if history is not None else None

    logits_b, value_b = forward_apply(params, card_b, action_hist_b, action_mask_b, history_b)
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
def reset_opp_stats_where_done(opp_stats: VecOpponentStats, done: chex.Array) -> VecOpponentStats:
    """Reset opponent stats for completed episodes (new opponent)."""
    # When done, we're playing a new opponent, so reset their stats
    # Keep accumulated stats but reset per-round tracking
    done_2d = done[:, None]
    done_3d = done[:, None, None]
    
    # Reset frequency and average stats to 0 for new opponents
    freq_stats = jnp.where(done_3d, jnp.zeros_like(opp_stats.freq_stats), opp_stats.freq_stats)
    avg_stats = jnp.where(done_3d, jnp.zeros_like(opp_stats.avg_stats), opp_stats.avg_stats)
    rounds_observed = jnp.where(done, jnp.zeros_like(opp_stats.rounds_observed), opp_stats.rounds_observed)
    
    # Reset per-round state
    preflop_aggressor = jnp.where(done, jnp.full_like(opp_stats.preflop_aggressor, -1), opp_stats.preflop_aggressor)
    opp_bet_flop = jnp.where(done, jnp.zeros_like(opp_stats.opp_bet_flop), opp_stats.opp_bet_flop)
    opp_bet_turn = jnp.where(done, jnp.zeros_like(opp_stats.opp_bet_turn), opp_stats.opp_bet_turn)
    opp_checked_this_street = jnp.where(done, jnp.zeros_like(opp_stats.opp_checked_this_street), opp_stats.opp_checked_this_street)
    pot_at_street_start = jnp.where(done, jnp.zeros_like(opp_stats.pot_at_street_start), opp_stats.pot_at_street_start)
    
    return opp_stats.replace(
        freq_stats=freq_stats,
        avg_stats=avg_stats,
        rounds_observed=rounds_observed,
        preflop_aggressor=preflop_aggressor,
        opp_bet_flop=opp_bet_flop,
        opp_bet_turn=opp_bet_turn,
        opp_checked_this_street=opp_checked_this_street,
        pot_at_street_start=pot_at_street_start,
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
    - Now includes opponent history features
    """
    # Select opponent from pool
    opp_params, key = pool_select_opponent(pool, key)
    # When pool only has 1 agent (self), use main_params for self-play
    opp_params = jax.tree_util.tree_map(
        lambda o, m: jax.lax.cond(pool.size <= 1, lambda _: m, lambda _: o, operand=None),
        opp_params,
        main_params,
    )

    env_ids = jnp.arange(CFG.num_envs, dtype=jnp.uint32)

    def step_fn(carry, t):
        vec = carry
        s = vec.env_state
        aux = vec.aux
        main_is_p0 = vec.main_is_p0
        opp_stats = vec.opp_stats
        base_keys = vec.rng
        learner_pid = jnp.where(main_is_p0, jnp.int32(0), jnp.int32(1))
        pid_cur = s.current_player.astype(jnp.int32)
        is_learner_turn = pid_cur == learner_pid

        # Get opponent history tensor for learner's model
        opp_history_tensor = vec_opponent_stats_to_tensor(opp_stats)  # (E, 64)

        # Encode from current player's perspective (for action selection)
        # Legal mask is player-independent, so we can reuse it
        card_cur_b, action_hist_cur_b, action_mask_cur_b, legal_b, _ = jax.vmap(
            lambda s_i, aux_i, pid_i, hist_i: encode_obs(s_i, aux_i, pid_i, hist_i)
        )(s, aux, pid_cur, opp_history_tensor)

        # For storage: need learner's perspective
        # When learner == current player, reuse encoding; otherwise compute learner's view
        def _get_learner_obs(s_i, aux_i, pid_cur_i, learner_pid_i, card_cur_i, ahist_cur_i, amask_cur_i, hist_i):
            return jax.lax.cond(
                pid_cur_i == learner_pid_i,
                lambda _: (card_cur_i, ahist_cur_i, amask_cur_i, hist_i),
                lambda _: (*encode_obs(s_i, aux_i, learner_pid_i, hist_i)[:3], hist_i),
                operand=None,
            )

        card_b, action_hist_b, action_mask_b, history_b = jax.vmap(_get_learner_obs)(
            s, aux, pid_cur, learner_pid, card_cur_b, action_hist_cur_b, action_mask_cur_b, opp_history_tensor
        )

        # Forward pass for main agent (learner)
        logits_main_b, _ = forward_apply(
            main_params, card_cur_b, action_hist_cur_b, action_mask_cur_b, opp_history_tensor
        )

        # Opponent forward (conditional)
        # Only compute separate opponent forward pass if pool has >1 agent
        # When pool.size <= 1, we're in self-play mode - reuse main logits
        has_opponent = pool.size > 1

        def compute_opp_logits():
            # Opponent doesn't use our history of them - they'd use history of us
            # For simplicity, opponent uses neutral history
            neutral_history = jnp.full((CFG.num_envs, HISTORY_INPUT_DIM), 0.5, dtype=jnp.float32)
            return forward_apply(opp_params, card_cur_b, action_hist_cur_b, action_mask_cur_b, neutral_history)[0]

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

        # Store learner's observation (with private cards and history)
        tmp = {
            "card": card_b,
            "action_hist": action_hist_b,
            "action_mask": action_mask_b,
            "history": history_b,  # NEW
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
        
        # Reset opponent stats for completed episodes (new opponent next game)
        opp_stats2 = reset_opp_stats_where_done(opp_stats, done)

        # Re-randomize seating on resets
        k_seat = jax.vmap(lambda k, eid: _mk_key(k, eid, 262143))(base_keys, env_ids)
        new_seat = jax.vmap(lambda k: jax.random.bernoulli(k, 0.5))(k_seat)
        main_is_p0_2 = jnp.where(done, new_seat, main_is_p0)

        vec2 = VecEnvState(
            env_state=s_cont, 
            aux=aux3, 
            main_is_p0=main_is_p0_2, 
            rng=base_keys,
            opp_stats=opp_stats2,
        )
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
    bootstrap_opp_history = vec_opponent_stats_to_tensor(vec2.opp_stats)
    bootstrap_card, bootstrap_action_hist, bootstrap_action_mask, _, bootstrap_history = jax.vmap(
        lambda s_i, aux_i, pid_i, hist_i: encode_obs(s_i, aux_i, pid_i, hist_i)
    )(vec2.env_state, vec2.aux, bootstrap_learner_pid, bootstrap_opp_history)
    # Track whether the last transition was terminal (bootstrap should be 0 if so)
    bootstrap_done = traj["done"][-1, :]  # (E,) - done flag at last timestep

    # Store unroll WITHOUT computing V-trace - that happens at training time
    # traj has shape (T, E, ...) which is what we want
    unroll = Unroll(
        card=traj["card"],
        action_hist=traj["action_hist"],
        action_mask=traj["action_mask"],
        history=traj["history"],  # NEW
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
        bootstrap_history=bootstrap_history,  # NEW
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

    Uses STOCHASTIC action selection for proper poker evaluation.
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
        # Use stochastic sampling for evaluation - matches training behavior and
        # is correct for poker (mixed strategies). deterministic=True causes policy
        # collapse issues where the model always picks "safe" actions like fold.
        # Use neutral history for evaluation
        neutral_history = jnp.full(HISTORY_INPUT_DIM, 0.5, dtype=jnp.float32)
        a, _, _, key = policy_act(params, s, aux, key, neutral_history, deterministic=False)
        aux2 = aux_record_action(aux, s, a)
        key, k_step = jax.random.split(key)
        s2 = ENV.step(s, k_step, a)
        return (s2, aux2, key)

    sT, _, key = jax.lax.while_loop(cond_fn, body_fn, (s, aux, key))
    r0, _ = terminal_rewards(sT)
    return r0, key


@partial(jax.jit, static_argnums=(2,))
def evaluate_pair(
    params_a: Any,
    params_b: Any,
    n_hands: int,
    key: chex.PRNGKey,
) -> Tuple[chex.Array, chex.PRNGKey]:
    """Evaluate params_a vs params_b over n_hands.

    Returns average chip delta for player A (positive = A wins).
    Plays half the hands with A as player0, half as player1 to reduce seat bias.
    """
    n_half = n_hands // 2
    keys = jax.random.split(key, n_hands + 1)
    key_out = keys[-1]
    hand_keys = keys[:-1]

    # First half: A as player0
    def play_a_as_p0(k):
        r, _ = play_one_hand(params_a, params_b, k)
        return r

    # Second half: A as player1 (swap params, negate result)
    def play_a_as_p1(k):
        r, _ = play_one_hand(params_b, params_a, k)
        return -r

    results_p0 = jax.vmap(play_a_as_p0)(hand_keys[:n_half])
    results_p1 = jax.vmap(play_a_as_p1)(hand_keys[n_half:])

    total = jnp.sum(results_p0) + jnp.sum(results_p1)
    mean_delta = total / n_hands

    return mean_delta, key_out


def eval_and_update_pool(
    pool: AgentPool,
    ts: TrainState,
    main_elo: chex.Array,
    key: chex.PRNGKey,
) -> Tuple[AgentPool, chex.Array, chex.PRNGKey]:
    """Evaluate current policy vs pool and update ELOs.

    Uses sigmoid scoring based on chip delta (tau=100 BBs).
    This means:
    - +100 chips (50 BBs) → ~73% expected score
    - +200 chips (100 BBs) → ~88% expected score
    - ±0 chips → 50% expected score
    """
    current_params = ts.params

    # Play against each agent in the pool
    def eval_vs_agent(carry, i):
        key = carry
        is_valid = i < pool.size

        agent_params = jax.tree_util.tree_map(lambda p: p[i], pool.params)
        agent_elo = pool.elo[i]

        delta, key = evaluate_pair(current_params, agent_params, CFG.eval_hands, key)

        # Sigmoid scoring: score = 1 / (1 + exp(-delta / tau))
        # tau controls how many chips = "decisive win"
        # With tau=100 (50 BBs), +100 chips gives ~73% score
        score = jax.nn.sigmoid(delta / CFG.elo_tau)

        # Update both ELOs
        new_main, new_agent = elo_update(main_elo, agent_elo, score, CFG.elo_k)

        # Only apply updates if this slot is valid
        main_elo_out = jnp.where(is_valid, new_main, main_elo)
        agent_elo_out = jnp.where(is_valid, new_agent, agent_elo)

        return key, (main_elo_out, agent_elo_out, is_valid)

    key, (main_elos, agent_elos, valids) = jax.lax.scan(
        eval_vs_agent, key, jnp.arange(CFG.pool_max)
    )

    # Take the last valid main_elo update
    final_main_elo = main_elos[-1]

    # Update pool ELOs
    new_pool_elo = jnp.where(valids, agent_elos, pool.elo)
    pool = pool.replace(elo=new_pool_elo)

    # Compute style stats for current agent
    # This is a simple approximation - track raise/fold/call rates during eval
    # For now, use placeholder stats
    style_stats = jnp.array([0.3, 0.3, 0.4], dtype=jnp.float32)  # [raise, fold, call]

    # Add current agent to pool
    pool = pool_add(pool, current_params, final_main_elo, style_stats)

    return pool, final_main_elo, key


# =============================================================================
# 10) Update Phase (Training Loop)
# =============================================================================


def make_update_phase():
    """Create the update phase function."""

    def update_phase(
        ts: TrainState,
        buf: UnrollBuffer,
        key: chex.PRNGKey,
        ent_coef: chex.Array,
    ) -> Tuple[TrainState, Dict[str, chex.Array], chex.PRNGKey]:
        """Single update phase: sample unrolls and train."""

        def _epoch_step(epoch_carry, _):
            ts, key = epoch_carry
            unrolls, key = unroll_buffer_sample(buf, key, CFG.num_minibatches)

            def _unroll_step(ts, i):
                unroll_i = jax.tree_util.tree_map(lambda x: x[i], unrolls)
                new_ts, metrics = train_step_on_unroll(ts, unroll_i, ent_coef)
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

    # Initialize pool with initial agent
    pool = pool_init(ts.params)
    # Initial style stats: use zeros (will be updated on first pool evaluation)
    init_style_stats = jnp.zeros(3, dtype=jnp.float32)
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
                f"Ill: {m['illegal_penalty']:.4f} | "
                f"EntC: {float(m['ent_coef']):.4f} | "
                f"rho: {m['rho_mean']:.2f}"
            )

        # Checkpointing
        if end_step % CFG.checkpoint_every == 0:
            save_path = f"{ckpt_dir}/step_{end_step}"
            if os.path.exists(save_path):
                import shutil
                shutil.rmtree(save_path)
            ckpter.save(save_path, ts, save_args=save_args)
            print(f"Saved checkpoint to {save_path}")

    print("Training complete!")


if __name__ == "__main__":
    main()