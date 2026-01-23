"""Transformer-based IMPALA V-trace training for Toss-or-Hold'em (JAX/Flax).

This training script implements IMPALA-style training:
  1) Transformer-based card/action encoding
  2) Pure V-trace off-policy correction computed at TRAINING TIME
  3) Short replay window storing complete unrolls for proper V-trace computation
  4) K-Best self-play league training with ELO-based opponent selection

Key architectural choices:
  - Card tensor: (B, 9, 2) - card index + type embedding
  - Action tensor: (B, 30, 7) - action history with context features
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

import numpy as np
import chex
import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
import orbax.checkpoint
from flax.training import orbax_utils
from jax import debug as jax_debug

# Import the transformer model
from model import (
    TossHoldemModel,
    apply_legal_action_mask,
    create_action_mask,
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
    PHASE_PREFLOP,
    PHASE_DISCARD_OOP,
    PHASE_DISCARD_DLR,
    PHASE_FLOP,
    PHASE_TURN,
    PHASE_RIVER,
    PHASE_SHOWDOWN,
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
    num_envs: int = 2_048  # 2,048 parallel environments (not 2^4=16)
    rollout_len: int = 64

    # V-trace hyperparameters (IMPALA-style)
    gamma: float = 0.999
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
    num_minibatches: int = 4
    update_epochs: int = 1  # Single epoch per update (more on-policy)
    warmup_steps: int = 100_000  # Shorter warmup (in transitions)

    # Short replay window (key for V-trace) - only sample from recent data
    recent_window_transitions: int = 50_000  # Sample only from recent N transitions (0 = no limit)

    # League training (K-best self-play)
    k_best: int = 5
    pool_max: int = 8
    eval_hands: int = 512
    add_to_pool_every_updates: int = 50
    elo_k: float = 32.0

    # Runtime
    seed: int = 0
    total_updates: int = 20_000
    debug_asserts: bool = False  # Enable expensive assertions (for debugging only)

    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    checkpoint_every: int = 1000
    log_every: int = 50


CFG = Config()

# Shared env instance (avoid re-instantiating inside jit)
ENV = TossHoldemEnv()


# =============================================================================
# 2) State Representation (Transformer-compatible)
# =============================================================================


def _phase_to_round_idx(phase: chex.Array) -> chex.Array:
    """Map env phases into action-tensor rounds.

    We use 5 rounds for Toss-or-Hold'em:
      0: PREFLOP
      1: DISCARD (OOP + DLR discards)
      2: FLOP
      3: TURN
      4: RIVER
    """
    return jnp.select(
        [
            phase == PHASE_PREFLOP,
            (phase == PHASE_DISCARD_OOP) | (phase == PHASE_DISCARD_DLR),
            phase == PHASE_FLOP,
            phase == PHASE_TURN,
            phase == PHASE_RIVER,
        ],
        [0, 1, 2, 3, 4],
        default=4,
    ).astype(jnp.int32)


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
    # FIX: Use consistent pot size (chips already committed to the middle)
    pot_committed = (state.pot + state.p0_bet + state.p1_bet).astype(jnp.float32)

    eff_cap = jnp.minimum(cur_stack, opp_stack + to_call)

    street_id = phase_to_street(state.phase).astype(jnp.float32)

    token = jnp.array(
        [
            state.current_player.astype(jnp.float32),
            action.astype(jnp.float32),
            street_id,
            time_id.astype(jnp.float32),
            to_call,
            pot_committed,
            eff_cap,
        ],
        dtype=jnp.float32,
    )
    return token


def _assert_card_tensor(card_tensor: chex.Array) -> None:
    """Sanity check card tensor contract.

    card[...,0] in {-1} ∪ [0, 51]
    card[...,1] in {-1} ∪ [0, 8]
    """
    card_idx = card_tensor[..., 0]
    type_id = card_tensor[..., 1]

    ok_idx = (card_idx == -1) | ((card_idx >= 0) & (card_idx <= 51))
    ok_type = (type_id == -1) | ((type_id >= 0) & (type_id <= 8))

    jax_debug.assert_(jnp.all(ok_idx), "Invalid card_index in card_tensor")
    jax_debug.assert_(jnp.all(ok_type), "Invalid type_id in card_tensor")


def _assert_legal_action(legal_mask: chex.Array, action: chex.Array) -> None:
    """Sanity check: sampled action must be legal."""
    chosen = jnp.take_along_axis(legal_mask, action[:, None], axis=-1)[:, 0]
    jax_debug.assert_(jnp.all(chosen), "Sampled illegal action")


@chex.dataclass(frozen=True)
class AuxState:
    """Extra per-env state to build action history tensors.

        Action tensor shape: (ACTION_SEQ_LEN, ACTION_INPUT_DIM) = (30, 7)
        Each action entry contains:
            [0] actor_id          int {0,1}
            [1] event_type        int {0..12}
            [2] street_id         int {0..4}
            [3] time_id           int
            [4] to_call           float
            [5] pot_committed     float (chips already in middle)
            [6] effective_stack   float
    """
    action_tensor: chex.Array  # (30, 7)
    action_ptr: chex.Array     # Current position in action history
    round_idx: chex.Array      # Current round index
    slot_idx: chex.Array       # Position within current round


@jax.jit
def aux_reset() -> AuxState:
    return AuxState(
        action_tensor=jnp.zeros((ACTION_SEQ_LEN, ACTION_INPUT_DIM), dtype=jnp.float32),
        action_ptr=jnp.int32(0),
        round_idx=jnp.int32(0),
        slot_idx=jnp.int32(0),
    )


@jax.jit
def aux_maybe_advance_round(aux: AuxState, phase: chex.Array) -> AuxState:
    """Update round index when phase changes."""
    new_round = _phase_to_round_idx(phase)
    changed = new_round != aux.round_idx
    return aux.replace(
        round_idx=new_round,
        slot_idx=jnp.where(changed, jnp.int32(0), aux.slot_idx),
    )


@jax.jit
def aux_record_action(
    aux: AuxState,
    state: State,
    action: chex.Array,
) -> AuxState:
    """Record an action into the action history tensor.
    """

    # Use a monotonically increasing time id to avoid saturation loss
    time_id = aux.action_ptr.astype(jnp.int32)
    action_entry = make_action_token(state, action, time_id)
    
    # Write to action tensor (circular buffer behavior if > 30 actions)
    write_idx = aux.action_ptr % ACTION_SEQ_LEN
    action_tensor = aux.action_tensor.at[write_idx].set(action_entry)
    
    return aux.replace(
        action_tensor=action_tensor,
        action_ptr=aux.action_ptr + 1,
        slot_idx=aux.slot_idx + 1,
    )


@jax.jit
def encode_cards(state: State, player_id: chex.Array) -> chex.Array:
        """Encode cards into (9, 2) tensor.
    
        Card positions / type_id mapping:
            0: heropriv0
            1: heropriv1
            2: heropriv2
            3: flop0
            4: flop1
            5: herodiscard
            6: villaindiscard
            7: turn
            8: river
    
        Each card entry: [card_index, type_id]
            - card_index: 0-51 (or -1 for undealt/invalid)
            - type_id: 0-8 (or -1 for invalid)
        """
        is_p0 = player_id == 0
    
        # Get hole cards for the viewing player
        hole = jnp.where(is_p0, state.p0_hole, state.p1_hole)
        active_mask = jnp.where(is_p0, state.p0_active_mask, state.p1_active_mask)
    
        # Initialize with invalids
        card_index = jnp.full((CARD_SEQ_LEN,), -1, dtype=jnp.int32)
        type_id = jnp.full((CARD_SEQ_LEN,), -1, dtype=jnp.int32)

        # Hero hole cards (positions 0-2)
        hole_valid = active_mask.astype(jnp.bool_)
        card_index = card_index.at[0:3].set(jnp.where(hole_valid, hole, -1))
        type_id = type_id.at[0:3].set(
                jnp.where(hole_valid, jnp.arange(3, dtype=jnp.int32), -1)
        )

        # Flop (positions 3-4)
        flop_cards = state.board[0:2]
        flop_valid = flop_cards >= 0
        card_index = card_index.at[3:5].set(jnp.where(flop_valid, flop_cards, -1))
        type_id = type_id.at[3:5].set(
                jnp.where(flop_valid, jnp.array([3, 4], dtype=jnp.int32), -1)
        )

        # Discards: board[2] = OOP discard, board[3] = dealer discard
        hero_is_dealer = jnp.where(is_p0, state.is_dealer_p0, jnp.logical_not(state.is_dealer_p0))
        hero_discard = jnp.where(hero_is_dealer, state.board[3], state.board[2])
        villain_discard = jnp.where(hero_is_dealer, state.board[2], state.board[3])

        hero_discard_valid = hero_discard >= 0
        villain_discard_valid = villain_discard >= 0
        card_index = card_index.at[5].set(jnp.where(hero_discard_valid, hero_discard, -1))
        type_id = type_id.at[5].set(jnp.where(hero_discard_valid, jnp.int32(5), -1))
        card_index = card_index.at[6].set(jnp.where(villain_discard_valid, villain_discard, -1))
        type_id = type_id.at[6].set(jnp.where(villain_discard_valid, jnp.int32(6), -1))

        # Turn/River (positions 7-8)
        turn_card = state.board[4]
        river_card = state.board[5]
        turn_valid = turn_card >= 0
        river_valid = river_card >= 0
        card_index = card_index.at[7].set(jnp.where(turn_valid, turn_card, -1))
        type_id = type_id.at[7].set(jnp.where(turn_valid, jnp.int32(7), -1))
        card_index = card_index.at[8].set(jnp.where(river_valid, river_card, -1))
        type_id = type_id.at[8].set(jnp.where(river_valid, jnp.int32(8), -1))
    
        return jnp.stack([card_index, type_id], axis=-1)


@jax.jit
def encode_actions(aux: AuxState, player_id: chex.Array) -> Tuple[chex.Array, chex.Array]:
    """Encode action history into (30, 7) tensor with ego-relative player IDs.

    Returns:
        action_tensor: (30, 7) action history
        action_mask: (30,) boolean mask for valid actions (pointer-based)
    """
    # Re-center circular buffer so the most recent action is at the end
    ptr = aux.action_ptr % ACTION_SEQ_LEN
    action_tensor = jnp.roll(aux.action_tensor, shift=-ptr, axis=0)

    # Make player IDs ego-relative (0 = self, 1 = opponent)
    # Original: actor_id in action_tensor[..., 0]
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
        card_tensor: (9, 2) int32
        action_tensor: (30, 7) float32
        action_mask: (30,) bool
        legal_mask: (NUM_ACTIONS,) bool
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
    action_hist: chex.Array # (T, E, 30, 7) float32 - action history
    action_mask: chex.Array # (T, E, 30) bool - action history mask
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
    bootstrap_action_hist: chex.Array # (E, 30, 7) - action history for s_T
    bootstrap_action_mask: chex.Array # (E, 30) - action mask for s_T
    bootstrap_done: chex.Array        # (E,) - whether last transition was terminal


@chex.dataclass(frozen=True)
class UnrollBuffer:
    """Buffer storing complete unrolls for V-trace computation.

    Each entry is a full trajectory of shape (rollout_len, num_envs, ...).
    This enables proper importance weight computation across time.
    """
    card: chex.Array        # (cap, T, E, 9, 2)
    action_hist: chex.Array # (cap, T, E, 30, 7)
    action_mask: chex.Array # (cap, T, E, 30)
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
    bootstrap_action_hist: chex.Array # (cap, E, 30, 7)
    bootstrap_action_mask: chex.Array # (cap, E, 30)
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
    """Pool of historical agent parameters for self-play."""
    params: Any              # Stacked params along axis 0
    elo: chex.Array          # (pool_max,) ELO ratings
    size: chex.Array         # Current number of agents


def pool_init(dummy_params: Any) -> AgentPool:
    """Initialize empty agent pool."""
    def _zeros_like(p):
        return jnp.zeros((CFG.pool_max,) + p.shape, dtype=p.dtype)
    
    params = jax.tree_util.tree_map(_zeros_like, dummy_params)
    elo = jnp.zeros((CFG.pool_max,), dtype=jnp.float32)
    return AgentPool(params=params, elo=elo, size=jnp.int32(0))


@jax.jit
def pool_add(pool: AgentPool, new_params: Any, initial_elo: chex.Array) -> AgentPool:
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
    size2 = jnp.where(can_append, size + 1, size)
    
    return pool.replace(params=params, elo=elo, size=size2)


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
    """Select an opponent from the K-best agents in the pool."""
    key, sub = jax.random.split(key)
    size = pool.size
    
    # Get top-K by ELO
    valid_elo = jnp.where(jnp.arange(CFG.pool_max) < size, pool.elo, -1e9)
    _, topk_idx = jax.lax.top_k(valid_elo, k=CFG.k_best)
    
    # Randomly select from top-K
    k_eff = jnp.minimum(size, jnp.int32(CFG.k_best))
    pick = jax.random.randint(sub, (), 0, jnp.maximum(k_eff, 1))
    idx = topk_idx[pick]
    
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
    """
    d2 = contrib_self.astype(jnp.float32)
    d3 = contrib_opp.astype(jnp.float32)
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
        action_hist: (B, 30, 7) action history
        action_mask: (B, 30) action mask

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
        unroll.bootstrap_card,      # (E, 9, 2)
        unroll.bootstrap_action_hist,  # (E, 30, 7)
        unroll.bootstrap_action_mask,  # (E, 30)
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

    rho_exp = jnp.exp(logp_pi_flat - unroll.logp_mu.reshape(-1))
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
    aux = jax.vmap(aux_maybe_advance_round)(aux, s.phase)

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
    round_idx = jnp.where(done, jnp.zeros_like(aux.round_idx), aux.round_idx)
    slot_idx = jnp.where(done, jnp.zeros_like(aux.slot_idx), aux.slot_idx)
    
    return aux.replace(
        action_tensor=action_tensor,
        action_ptr=action_ptr,
        round_idx=round_idx,
        slot_idx=slot_idx,
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
    by clipping importance weights to bound variance. This allows learning
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

        aux = jax.vmap(aux_maybe_advance_round)(aux, s.phase)
        is_learner_turn = s.current_player.astype(jnp.int32) == learner_pid

        # Encode observations from LEARNER's perspective (private cards included)
        # This is used for both actor and critic - critic now sees private cards
        card_b, action_hist_b, action_mask_b, legal_b = jax.vmap(encode_obs)(
            s, aux, learner_pid
        )
        # Optional debug assertions (disable for performance)
        if CFG.debug_asserts:
            _assert_card_tensor(card_b)

        # For action selection, we need current player's perspective
        pid_cur = s.current_player.astype(jnp.int32)
        card_cur_b, action_hist_cur_b, action_mask_cur_b, legal_cur_b = jax.vmap(encode_obs)(
            s, aux, pid_cur
        )

        # Forward pass for main agent (learner)
        logits_main_b, _ = forward_apply(
            main_params, card_cur_b, action_hist_cur_b, action_mask_cur_b
        )

        # Opponent forward (conditional)
        has_opponent = pool.size > 0
        need_opp_logits = has_opponent & jnp.any(~is_learner_turn)

        def compute_opp_logits():
            return forward_apply(opp_params, card_cur_b, action_hist_cur_b, action_mask_cur_b)[0]

        def use_main_logits():
            return logits_main_b

        logits_opp_b = jax.lax.cond(need_opp_logits, compute_opp_logits, use_main_logits)

        # Select logits based on whose turn it is
        logits = jnp.where(is_learner_turn[:, None], logits_main_b, logits_opp_b)

        # Sample actions
        def _mk_key(k, eid, salt):
            k = jax.random.fold_in(k, eid)
            return jax.random.fold_in(k, t + jnp.uint32(salt))

        k_act = jax.vmap(lambda k, eid: _mk_key(k, eid, 0))(base_keys, env_ids)
        g = jax.vmap(lambda k: jax.random.gumbel(k, (NUM_ACTIONS,), dtype=jnp.float32))(k_act)
        neg_inf = -1e9
        masked_logits = jnp.where(legal_cur_b, logits, neg_inf)
        action = jnp.argmax(masked_logits + g, axis=-1).astype(jnp.int32)
        # Optional debug assertions (disable for performance)
        if CFG.debug_asserts:
            _assert_legal_action(legal_cur_b, action)

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

        # Compute rewards
        r0, r1 = terminal_rewards(s_next)
        rew_term = jnp.where(main_is_p0, r0, r1)
        rew = jnp.where(done, rew_term, jnp.float32(0.0))

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
        aux3 = jax.vmap(aux_maybe_advance_round)(aux3, s_cont.phase)

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

    Uses the global ENV instance.
    """
    key, k_reset = jax.random.split(key)
    s = ENV.reset(k_reset)
    aux = aux_reset()
    aux = aux_maybe_advance_round(aux, s.phase)

    def cond_fn(carry):
        s, aux, key = carry
        return ~s.done

    def body_fn(carry):
        s, aux, key = carry
        aux = aux_maybe_advance_round(aux, s.phase)
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
    score = jnp.where(r0 > 0, 1.0, jnp.where(r0 < 0, 0.0, 0.5)).astype(jnp.float32)
    return score, key


@jax.jit
def eval_and_update_pool(
    pool: AgentPool,
    main_state: TrainState,
    main_elo: jnp.ndarray,
    key: chex.PRNGKey,
) -> Tuple[AgentPool, jnp.ndarray, chex.PRNGKey]:
    """Evaluate main agent against pool and update ELOs."""
    pool_max = CFG.pool_max
    eval_hands = CFG.eval_hands
    k_best = CFG.k_best
    elo_k = CFG.elo_k
    
    # Generate keys for matches
    key, sub = jax.random.split(key)
    keys_grid = jax.random.split(sub, pool_max * eval_hands).reshape((pool_max, eval_hands, -1))
    
    valid_mask = jnp.arange(pool_max) < pool.size
    valid_f = valid_mask.astype(jnp.float32)
    n_valid = jnp.maximum(1.0, jnp.sum(valid_f))
    
    # Play matches
    def play_vs_one_opp(opp_params, keys_n):
        scores, _ = jax.vmap(play_one_hand, in_axes=(None, None, 0))(
            main_state.params, opp_params, keys_n
        )
        return jnp.mean(scores)
    
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
    
    # Add main to pool
    valid_elos = jnp.where(valid_mask, pool.elo, -1e9)
    _, topk_idx = jax.lax.top_k(valid_elos, k=k_best)
    
    k_eff = jnp.minimum(pool.size, jnp.int32(k_best))
    topk_mask = (jnp.arange(k_best) < k_eff).astype(jnp.float32)
    
    topk_vals = pool.elo[topk_idx]
    denom = jnp.maximum(1.0, jnp.sum(topk_mask))
    avg_top_elo = jnp.sum(topk_vals * topk_mask) / denom
    
    entry_elo = jnp.where(pool.size > 0, avg_top_elo, 0.0)
    pool = pool_add(pool, main_state.params, entry_elo)
    
    # Trim to K-best
    size_curr = pool.size
    valid_mask_final = jnp.arange(pool_max) < size_curr
    valid_elos_final = jnp.where(valid_mask_final, pool.elo, -1e9)
    
    _, keep_idx = jax.lax.top_k(valid_elos_final, k=k_best)
    n_keep = jnp.minimum(size_curr, jnp.int32(k_best))
    
    def _gather_pad(x):
        picked = x[keep_idx]
        mask = (jnp.arange(k_best) < n_keep).reshape((-1,) + (1,) * (x.ndim - 1))
        picked = picked * mask.astype(picked.dtype)
        pad_len = pool_max - k_best
        pad = jnp.zeros((pad_len,) + x.shape[1:], dtype=x.dtype)
        return jnp.concatenate([picked, pad], axis=0)
    
    new_params = jax.tree_util.tree_map(_gather_pad, pool.params)
    
    picked_elo = pool.elo[keep_idx]
    picked_elo = picked_elo * (jnp.arange(k_best) < n_keep).astype(jnp.float32)
    new_elo = jnp.concatenate(
        [picked_elo, jnp.zeros(pool_max - k_best, dtype=pool.elo.dtype)],
        axis=0,
    )
    
    pool_final = pool.replace(params=new_params, elo=new_elo, size=n_keep)
    
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
    unroll_capacity = CFG.replay_capacity // transitions_per_unroll
    unroll_capacity = max(16, unroll_capacity)  # Minimum capacity for stability
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
    pool = pool_add(pool, ts.params, jnp.float32(0.0))
    main_elo = jnp.float32(0.0)

    # Initialize environment
    key, k_reset = jax.random.split(key)
    vec, key = vec_reset(k_reset, CFG.num_envs)

    # Compile functions
    print("Compiling update functions...")
    do_update = make_update_step()
    do_collect = jax.jit(collect_rollout)

    # Warmup unroll buffer
    print("Warming up unroll buffer...")
    warm_updates = max(1, CFG.warmup_steps // (CFG.num_envs * CFG.rollout_len))
    for _ in range(warm_updates):
        buf, vec, key = do_collect(ts.params, pool, buf, vec, key)
    print(f"Buffer size after warmup: {buf.size} unrolls")

    # Training loop
    print("Starting training...")
    import time
    start_time = time.time()

    for upd in range(CFG.total_updates):
        # Collect data
        buf, vec, key = do_collect(ts.params, pool, buf, vec, key)

        # Compute scheduled entropy coefficient
        ent_coef = get_entropy_coef(upd)

        # Update policy with scheduled entropy coefficient
        ts, metrics, key = do_update(ts, buf, key, ent_coef)

        # Pool maintenance
        if (upd + 1) % CFG.add_to_pool_every_updates == 0:
            pool, main_elo, key = eval_and_update_pool(pool, ts, main_elo, key)

        # Logging
        if (upd + 1) % CFG.log_every == 0:
            m = jax.device_get(metrics)

            elapsed = time.time() - start_time
            sps = (CFG.log_every * CFG.num_envs * CFG.rollout_len) / elapsed
            start_time = time.time()

            print(
                f"Upd {upd+1}/{CFG.total_updates} | "
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
        if (upd + 1) % CFG.checkpoint_every == 0:
            save_path = f"{ckpt_dir}/step_{upd+1}"
            ckpter.save(save_path, ts, save_args=save_args)
            print(f"Saved checkpoint to {save_path}")

    print("Training complete!")


if __name__ == "__main__":
    main()