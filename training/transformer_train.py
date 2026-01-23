"""Transformer-based AlphaHoldem training for Toss-or-Hold'em (JAX/Flax).

This training script integrates the transformer model architecture with
AlphaHoldem's training methodology:
  1) Transformer-based card/action encoding (replacing pseudo-siamese ConvNets)
  2) Actor-critic with Trinal-Clip PPO (policy + clipped value target)
  3) Off-policy training from a replay buffer
  4) K-Best self-play with ELO-based opponent pool selection

Key changes from original AlphaHoldem:
  - Card tensor: (B, 8, 2) instead of (B, 4, 13, 7)
    - Action tensor: (B, 30, 7) instead of (B, 4, 13, 30)
  - Transformer encoders instead of ConvNets
  - Cross-attention fusion instead of element-wise addition
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
    # Parallel sampling
    num_envs: int = 1_024
    rollout_len: int = 128

    # PPO/GAE
    gamma: float = 0.999
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    delta1: float = 3.0  # Trinal-clip upper bound for policy ratio
    ent_coef: float = 0.01
    vf_coef: float = 0.5

    # Optimization
    lr: float = 3e-4
    max_grad_norm: float = 1.0

    # Off-policy replay + updates
    replay_capacity: int = 1_000_000
    batch_size: int = 32_768
    num_minibatches: int = 4
    update_epochs: int = 4
    warmup_steps: int = 400_000

    # Mild staleness control
    recent_window: int = 200_000

    # K-Best self-play
    k_best: int = 5
    pool_max: int = 8
    eval_hands: int = 512
    add_to_pool_every_updates: int = 50
    elo_k: float = 32.0

    # Runtime
    seed: int = 0
    total_updates: int = 20_000
    
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

    time_id is per-street (slot index within the current round).
    """
    is_p0 = state.current_player == 0

    cur_stack = jnp.where(is_p0, state.p0_stack, state.p1_stack).astype(jnp.float32)
    cur_bet = jnp.where(is_p0, state.p0_bet, state.p1_bet).astype(jnp.float32)
    opp_stack = jnp.where(is_p0, state.p1_stack, state.p0_stack).astype(jnp.float32)
    opp_bet = jnp.where(is_p0, state.p1_bet, state.p0_bet).astype(jnp.float32)

    to_call = (opp_bet - cur_bet).astype(jnp.float32)
    pot_base = (state.pot + state.p0_bet + state.p1_bet).astype(jnp.float32) + to_call

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
    
        Action tensor shape: (ACTION_SEQ_LEN, ACTION_INPUT_DIM) = (30, 7)
        Each action entry contains:
            [0] actor_id          int {0,1}
            [1] event_type        int {0..12}
            [2] street_id         int {0..4}
            [3] time_id           int
            [4] to_call           float
            [5] pot_base          float
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

    time_id = aux.slot_idx.astype(jnp.int32)
    action_entry = make_action_token(state, action, time_id)
    
    # Write to action tensor (circular buffer behavior if > 30 actions)
    write_idx = aux.action_ptr % ACTION_SEQ_LEN
    action_tensor = aux.action_tensor.at[write_idx].set(action_entry)
    
    return aux.replace(
        action_tensor=action_tensor,
        action_ptr=aux.action_ptr + 1,
        slot_idx=jnp.minimum(aux.slot_idx + 1, jnp.int32(5)),
    )


@jax.jit
def encode_cards(state: State, player_id: chex.Array) -> chex.Array:
    """Encode cards into (8, 2) tensor.
    
    Card positions:
      0-1: Player's hole cards (active only)
      2-3: Initial flop cards (board[0:2])
      4-5: Discard cards (board[2:4]) - marked with dealer_id of who discarded
      6: Turn card (board[4])
      7: River card (board[5])
    
    Each card entry: [dealer_id, card_index]
      - dealer_id: 0=non-dealer, 1=dealer, 2=community
      - card_index: 0-51 (or -1 for undealt/invalid)
    """
    is_p0 = player_id == 0
    
    # Get hole cards for the viewing player
    hole = jnp.where(is_p0, state.p0_hole, state.p1_hole)
    active_mask = jnp.where(is_p0, state.p0_active_mask, state.p1_active_mask)
    
    # OPTIMIZATION D4: Vectorize card tensor construction to avoid Python loops
    # Initialize card tensor with -1 (invalid)
    card_tensor = jnp.full((CARD_SEQ_LEN, CARD_INPUT_DIM), -1, dtype=jnp.int32)

    # Hole cards (positions 0-1): only active cards
    # Get indices of active hole cards (up to 2)
    active_indices = jnp.where(active_mask, jnp.arange(3), jnp.int32(-1))
    # Sort to get valid indices first (use jnp.flip instead of [::-1] for JIT compatibility)
    sorted_active = jnp.flip(jnp.sort(active_indices))[:2]

    # Vectorized hole card filling (positions 0-1)
    valid_hole = sorted_active >= 0
    card_vals_hole = jnp.where(valid_hole, hole[jnp.clip(sorted_active, 0, 2)], -1)
    owners_hole = jnp.where(valid_hole, player_id, -1)
    card_tensor = card_tensor.at[0:2, 0].set(owners_hole)
    card_tensor = card_tensor.at[0:2, 1].set(card_vals_hole)

    # Vectorized flop cards (positions 2-3): community cards
    flop_cards = state.board[0:2]
    valid_flop = flop_cards >= 0
    owners_flop = jnp.where(valid_flop, 2, -1)  # 2 = community
    card_tensor = card_tensor.at[2:4, 0].set(owners_flop)
    card_tensor = card_tensor.at[2:4, 1].set(flop_cards)
    
    # Vectorized discard cards (positions 4-5): marked with who discarded
    # board[2] = OOP discard, board[3] = dealer discard
    oop_id = jnp.where(state.is_dealer_p0, 1, 0)
    dlr_id = jnp.where(state.is_dealer_p0, 0, 1)
    discard_cards = state.board[2:4]
    discard_ids = jnp.array([oop_id, dlr_id], dtype=jnp.int32)
    valid_discards = discard_cards >= 0
    owners_discards = jnp.where(valid_discards, discard_ids, -1)
    card_tensor = card_tensor.at[4:6, 0].set(owners_discards)
    card_tensor = card_tensor.at[4:6, 1].set(discard_cards)

    # Vectorized turn and river (positions 6-7): community cards
    turn_river = state.board[4:6]
    valid_turn_river = turn_river >= 0
    owners_turn_river = jnp.where(valid_turn_river, 2, -1)
    card_tensor = card_tensor.at[6:8, 0].set(owners_turn_river)
    card_tensor = card_tensor.at[6:8, 1].set(turn_river)
    
    return card_tensor


@jax.jit
def encode_actions(aux: AuxState, player_id: chex.Array) -> Tuple[chex.Array, chex.Array]:
    """Encode action history into (30, 7) tensor with ego-relative player IDs.
    
    Returns:
        action_tensor: (30, 7) action history
        action_mask: (30,) boolean mask for valid actions
    """
    action_tensor = aux.action_tensor
    
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
    
    # Create mask: action is valid if any feature is non-zero
    action_mask = jnp.sum(jnp.abs(action_tensor), axis=-1) > 0
    
    return action_tensor, action_mask


@jax.jit
def encode_obs(
    state: State,
    aux: AuxState,
    player_id: chex.Array,
) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
    """Build card/action tensors + legal mask from state.
    
    Returns:
        card_tensor: (8, 2) int32
        action_tensor: (30, 7) float32
        action_mask: (30,) bool
        legal_mask: (NUM_ACTIONS,) bool
    """
    card_tensor = encode_cards(state, player_id)
    action_tensor, action_mask = encode_actions(aux, player_id)
    legal_mask = ENV.get_legal_actions(state)
    
    return card_tensor, action_tensor, action_mask, legal_mask


# =============================================================================
# 3) Replay Buffer
# =============================================================================


@chex.dataclass(frozen=True)
class Transition:
    """Single transition for training."""
    card: chex.Array        # (8, 2) int32 - card tensor
    action_hist: chex.Array # (30, 7) float32 - action history
    action_mask: chex.Array # (30,) bool - action history mask
    card_v: chex.Array      # (8, 2) - critic view cards
    action_hist_v: chex.Array  # (30, 7) - critic view actions
    action_mask_v: chex.Array  # (30,) - critic view action mask
    legal: chex.Array       # (13,) bool - legal actions
    action: chex.Array      # int32 - action taken
    logp_old: chex.Array    # float32 - log prob under old policy
    reward: chex.Array      # float32
    done: chex.Array        # bool
    adv: chex.Array         # float32 - GAE advantage
    ret: chex.Array         # float32 - discounted return
    contrib_self: chex.Array   # int32 - chips contributed by self
    contrib_opp: chex.Array    # int32 - chips contributed by opponent
    valid: chex.Array       # bool - is this a learner turn


@chex.dataclass(frozen=True)
class ReplayBuffer:
    """Off-policy replay buffer with fixed capacity."""
    card: chex.Array        # (cap, 8, 2)
    action_hist: chex.Array # (cap, 30, 7)
    action_mask: chex.Array # (cap, 30)
    card_v: chex.Array
    action_hist_v: chex.Array
    action_mask_v: chex.Array
    legal: chex.Array       # (cap, 13)
    action: chex.Array      # (cap,)
    logp_old: chex.Array
    reward: chex.Array
    done: chex.Array
    adv: chex.Array
    ret: chex.Array
    contrib_self: chex.Array
    contrib_opp: chex.Array
    valid: chex.Array
    ptr: chex.Array
    size: chex.Array


def replay_init(capacity: int) -> ReplayBuffer:
    """Initialize empty replay buffer."""
    return ReplayBuffer(
        card=jnp.zeros((capacity, CARD_SEQ_LEN, CARD_INPUT_DIM), dtype=jnp.int32),
        action_hist=jnp.zeros((capacity, ACTION_SEQ_LEN, ACTION_INPUT_DIM), dtype=jnp.float32),
        action_mask=jnp.zeros((capacity, ACTION_SEQ_LEN), dtype=jnp.bool_),
        card_v=jnp.zeros((capacity, CARD_SEQ_LEN, CARD_INPUT_DIM), dtype=jnp.int32),
        action_hist_v=jnp.zeros((capacity, ACTION_SEQ_LEN, ACTION_INPUT_DIM), dtype=jnp.float32),
        action_mask_v=jnp.zeros((capacity, ACTION_SEQ_LEN), dtype=jnp.bool_),
        legal=jnp.zeros((capacity, NUM_ACTIONS), dtype=jnp.bool_),
        action=jnp.zeros((capacity,), dtype=jnp.int32),
        logp_old=jnp.zeros((capacity,), dtype=jnp.float32),
        reward=jnp.zeros((capacity,), dtype=jnp.float32),
        done=jnp.zeros((capacity,), dtype=jnp.bool_),
        adv=jnp.zeros((capacity,), dtype=jnp.float32),
        ret=jnp.zeros((capacity,), dtype=jnp.float32),
        contrib_self=jnp.zeros((capacity,), dtype=jnp.int32),
        contrib_opp=jnp.zeros((capacity,), dtype=jnp.int32),
        valid=jnp.zeros((capacity,), dtype=jnp.bool_),
        ptr=jnp.int32(0),
        size=jnp.int32(0),
    )


@jax.jit
def replay_add_batch(buf: ReplayBuffer, tr: Transition) -> ReplayBuffer:
    """Add a batch of transitions to the replay buffer."""
    cap = buf.card.shape[0]
    B = tr.card.shape[0]
    idxs = (buf.ptr + jnp.arange(B, dtype=jnp.int32)) % cap

    buf2 = buf.replace(
        card=buf.card.at[idxs].set(tr.card),
        action_hist=buf.action_hist.at[idxs].set(tr.action_hist),
        action_mask=buf.action_mask.at[idxs].set(tr.action_mask),
        card_v=buf.card_v.at[idxs].set(tr.card_v),
        action_hist_v=buf.action_hist_v.at[idxs].set(tr.action_hist_v),
        action_mask_v=buf.action_mask_v.at[idxs].set(tr.action_mask_v),
        legal=buf.legal.at[idxs].set(tr.legal),
        action=buf.action.at[idxs].set(tr.action),
        logp_old=buf.logp_old.at[idxs].set(tr.logp_old),
        reward=buf.reward.at[idxs].set(tr.reward),
        done=buf.done.at[idxs].set(tr.done),
        adv=buf.adv.at[idxs].set(tr.adv),
        ret=buf.ret.at[idxs].set(tr.ret),
        contrib_self=buf.contrib_self.at[idxs].set(tr.contrib_self),
        contrib_opp=buf.contrib_opp.at[idxs].set(tr.contrib_opp),
        valid=buf.valid.at[idxs].set(tr.valid),
        ptr=(buf.ptr + B) % cap,
        size=jnp.minimum(buf.size + B, cap),
    )
    return buf2


@partial(jax.jit, static_argnums=(2,))
def replay_sample(
    buf: ReplayBuffer,
    key: chex.PRNGKey,
    batch_size: int,
) -> Tuple[Transition, chex.PRNGKey]:
    """Sample a batch from the replay buffer."""
    key, sub = jax.random.split(key)
    cap = buf.card.shape[0]
    
    # Use recent window for sampling
    limit = jnp.where(
        CFG.recent_window > 0,
        jnp.int32(CFG.recent_window),
        cap,
    )
    available = jnp.minimum(buf.size, limit)
    
    # Sample random offsets
    offsets = jax.random.randint(sub, (batch_size,), 0, available)
    idx = (buf.ptr - 1 - offsets) % cap
    
    tr = Transition(
        card=buf.card[idx],
        action_hist=buf.action_hist[idx],
        action_mask=buf.action_mask[idx],
        card_v=buf.card_v[idx],
        action_hist_v=buf.action_hist_v[idx],
        action_mask_v=buf.action_mask_v[idx],
        legal=buf.legal[idx],
        action=buf.action[idx],
        logp_old=buf.logp_old[idx],
        reward=buf.reward[idx],
        done=buf.done[idx],
        adv=buf.adv[idx],
        ret=buf.ret[idx],
        contrib_self=buf.contrib_self[idx],
        contrib_opp=buf.contrib_opp[idx],
        valid=buf.valid[idx],
    )
    return tr, key


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
# 6) Trinal-Clip PPO
# =============================================================================


@jax.jit
def trinal_clip_policy_objective(
    rt: chex.Array,
    adv: chex.Array,
    eps: float,
    delta1: float,
) -> chex.Array:
    """Trinal-Clip PPO objective from AlphaHoldem Eq. (3).
    
    L_tcp = clip(r_t, clip(r_t, 1-eps, 1+eps), delta1) * A_t
    
    Inner clip bounds r_t to [1-eps, 1+eps]
    Outer clip uses inner result as lower bound and delta1 as upper bound
    """
    rt_clip = jnp.clip(rt, 1.0 - eps, 1.0 + eps)
    rt_tcp = jnp.clip(rt, rt_clip, delta1)
    return rt_tcp * adv


@jax.jit
def trinal_clip_value_target(
    ret: chex.Array,
    contrib_self: chex.Array,
    contrib_opp: chex.Array,
) -> chex.Array:
    """Clip discounted return to [-d2, d3] where d2/d3 are chip contributions."""
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


def make_train_state(rng: chex.PRNGKey) -> TrainState:
    """Initialize model and optimizer."""
    model = TossHoldemModel()
    
    # Dummy inputs for initialization
    dummy_card = jnp.zeros((1, CARD_SEQ_LEN, CARD_INPUT_DIM), dtype=jnp.int32)
    dummy_action = jnp.zeros((1, ACTION_SEQ_LEN, ACTION_INPUT_DIM), dtype=jnp.float32)
    dummy_mask = jnp.ones((1, ACTION_SEQ_LEN), dtype=jnp.bool_)
    
    variables = model.init(rng, dummy_card, dummy_action, dummy_mask)
    
    tx = optax.chain(
        optax.clip_by_global_norm(CFG.max_grad_norm),
        optax.adam(CFG.lr),
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
def ppo_loss_and_metrics(
    state: TrainState,
    batch: Transition,
) -> Tuple[chex.Array, Dict[str, chex.Array]]:
    """Compute PPO loss and metrics."""

    # Forward pass for policy (actor view)
    logits, _ = forward_apply(
        state.params,
        batch.card,
        batch.action_hist,
        batch.action_mask,
    )

    # Forward pass for value (critic view)
    _, value = forward_apply(
        state.params,
        batch.card_v,
        batch.action_hist_v,
        batch.action_mask_v,
    )

    # Policy loss
    dist = masked_categorical(logits, batch.legal)
    logp = dist.log_prob(batch.action)
    log_ratio = logp - batch.logp_old
    rt = jnp.where(batch.valid, jnp.exp(log_ratio), 1.0)

    obj = trinal_clip_policy_objective(rt, batch.adv, CFG.clip_eps, CFG.delta1)

    valid_f = batch.valid.astype(jnp.float32)
    denom = jnp.maximum(1.0, jnp.sum(valid_f))
    policy_loss = -jnp.sum(obj * valid_f) / denom

    # Value loss
    target = trinal_clip_value_target(batch.ret, batch.contrib_self, batch.contrib_opp)
    v_loss = (target - value) ** 2
    value_loss = jnp.mean(v_loss)

    # Entropy bonus
    ent = dist.entropy()
    entropy = jnp.sum(ent * valid_f) / denom

    # Total loss
    total = (
        policy_loss
        + CFG.vf_coef * value_loss
        - CFG.ent_coef * entropy
    )

    # Metrics
    err = target - value
    rmse = jnp.sqrt(jnp.mean(err ** 2))
    t_mean = jnp.mean(target)
    t_var = jnp.mean((target - t_mean) ** 2)
    t_std = jnp.sqrt(t_var) + 1e-8
    value_fit_ratio = rmse / t_std

    metrics = {
        "loss": total,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": entropy,
        "value_mean": jnp.mean(value),
        "adv_mean": jnp.sum(batch.adv * valid_f) / denom,
        "ret_mean": jnp.sum(batch.ret * valid_f) / denom,
        "valid_frac": jnp.mean(valid_f),
        "mean_ratio": jnp.sum(rt * valid_f) / denom,
        "clip_frac": jnp.sum(
            ((jnp.abs(rt - 1.0) > CFG.clip_eps).astype(jnp.float32)) * valid_f
        ) / denom,
        "value_fit_ratio": value_fit_ratio,
    }

    return total, metrics


@jax.jit
def train_step(
    state: TrainState,
    batch: Transition,
) -> Tuple[TrainState, Dict[str, chex.Array]]:
    """Single training step."""
    def _loss_fn(params, batch_):
        return ppo_loss_and_metrics(state.replace(params=params), batch_)
    
    (loss, metrics), grads = jax.value_and_grad(_loss_fn, has_aux=True)(
        state.params, batch
    )
    new_state = state.apply_gradients(grads=grads)
    metrics = dict(metrics)
    metrics["loss"] = loss
    
    return new_state, metrics


# =============================================================================
# 8) Rollout Collection
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


@jax.jit
def policy_act(
    params: Any,
    state: State,
    aux: AuxState,
    key: chex.PRNGKey,
) -> Tuple[chex.Array, chex.Array, chex.Array, chex.PRNGKey]:
    """Sample action for the current player using Gumbel-argmax.

    Uses the same Gumbel-argmax sampling as rollout collection to ensure
    evaluation ELO ratings reflect performance under the training distribution.
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

    # Use Gumbel-argmax sampling (same as rollout collection)
    key, key_sample = jax.random.split(key)
    g = jax.random.gumbel(key_sample, (NUM_ACTIONS,), dtype=jnp.float32)
    neg_inf = -1e9
    masked_logits = jnp.where(legal, logits, neg_inf)
    action = jnp.argmax(masked_logits + g).astype(jnp.int32)

    # Compute log prob for this action (for consistency, though not used in eval)
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
def compute_gae_and_returns(
    rewards: chex.Array,
    values: chex.Array,
    dones: chex.Array,
    last_value: chex.Array,
) -> Tuple[chex.Array, chex.Array]:
    """Compute GAE advantages and discounted returns.
    
    Args:
        rewards: (T,) rewards at each timestep
        values: (T,) value estimates at each timestep
        dones: (T,) episode termination flags
        last_value: scalar, bootstrap value for final state
    
    Returns:
        advantages: (T,) GAE advantages
        returns: (T,) discounted returns (advantages + values)
    """
    not_done = 1.0 - dones.astype(jnp.float32)
    
    # Append bootstrap value
    values_extended = jnp.concatenate([values, last_value[None]])
    
    # TD residuals: δ_t = r_t + γ * (1 - done_t) * V(s_{t+1}) - V(s_t)
    deltas = rewards + CFG.gamma * not_done * values_extended[1:] - values_extended[:-1]
    
    # Reverse for backward scan
    deltas_rev = jnp.flip(deltas, axis=0)
    not_done_rev = jnp.flip(not_done, axis=0)
    
    # GAE: A_t = δ_t + γλ(1-done_t) * A_{t+1}
    def gae_step(adv_next, inputs):
        delta, nd = inputs
        adv = delta + CFG.gamma * CFG.gae_lambda * nd * adv_next
        return adv, adv
    
    _, advantages_rev = jax.lax.scan(gae_step, 0.0, (deltas_rev, not_done_rev))
    advantages = jnp.flip(advantages_rev, axis=0)
    
    # Returns = Advantages + Values (standard GAE identity)
    returns = advantages + values
    
    return advantages, returns  


@jax.jit
def collect_rollout(
    main_params: Any,
    pool: AgentPool,
    buf: ReplayBuffer,
    vec: VecEnvState,
    key: chex.PRNGKey,
) -> Tuple[ReplayBuffer, VecEnvState, chex.PRNGKey]:
    """Collect rollout_len transitions per environment using the global ENV instance."""
    # Select opponent from pool
    opp_params, key = pool_select_opponent(pool, key)
    opp_params = jax.tree_util.tree_map(
        lambda o, m: jax.lax.cond(pool.size == 0, lambda _: m, lambda _: o, operand=None),
        opp_params,
        main_params,
    )
    
    env_ids = jnp.arange(CFG.num_envs, dtype=jnp.uint32)
    
    def step_fn(carry, t):
        buf, vec = carry
        s = vec.env_state
        aux = vec.aux
        main_is_p0 = vec.main_is_p0
        base_keys = vec.rng
        learner_pid = jnp.where(main_is_p0, jnp.int32(0), jnp.int32(1))

        aux = jax.vmap(aux_maybe_advance_round)(aux, s.phase)
        is_learner_turn = s.current_player.astype(jnp.int32) == learner_pid

        # ========================================================================
        # PERFORMANCE OPTIMIZATIONS (addressing TPU throughput issues)
        # ========================================================================
        # D1: Reduced from 3 forward passes to 2 per step
        #     - Before: critic(learner), actor(main), actor(opponent) = 3 forwards
        #     - After: actor(current), critic(learner), conditionally actor(opponent) = 2-2.5 forwards
        # D2: Skip opponent forward when pool.size == 0 (common during warmup)
        # D3: Note: We still encode twice (current player + learner) because:
        #     - Actor needs current player perspective for correct observation
        #     - Critic needs learner perspective (always)
        #     - In batched setting, some envs have learner acting, some opponent
        #     - Conditionally reusing encodings per-env would add complex branching
        # D4: Vectorized encode_cards to eliminate Python loops
        # ========================================================================

        # OPTIMIZATION D3: Encode observations only once from current player perspective
        pid_cur = s.current_player.astype(jnp.int32)
        card_cur_b, action_hist_cur_b, action_mask_cur_b, legal_a_b = jax.vmap(encode_obs)(
            s, aux, pid_cur
        )

        # OPTIMIZATION D1: Reduce forward passes
        # Strategy:
        # - Always forward main_params with current player view (for learner logits when it's learner turn)
        # - Conditionally forward opp_params only if pool has opponents (for opponent logits)
        # - Always forward main_params with learner view for value (critic always needs learner perspective)
        # Result: 2 forwards when opponent acts, 2 forwards when learner acts (can't avoid both logits+value needs)

        # Forward #1: Main agent from current player perspective (for learner logits)
        logits_main_b, _ = forward_apply(
            main_params, card_cur_b, action_hist_cur_b, action_mask_cur_b
        )

        # Forward #2 (conditional): Opponent from current player perspective (for opponent logits)
        # OPTIMIZATION D2: Skip when pool is empty
        has_opponent = pool.size > 0

        def compute_opp_logits():
            return forward_apply(opp_params, card_cur_b, action_hist_cur_b, action_mask_cur_b)[0]

        def use_main_logits():
            return logits_main_b

        logits_opp_b = jax.lax.cond(has_opponent, compute_opp_logits, use_main_logits)

        # Select logits based on whose turn it is
        logits = jnp.where(is_learner_turn[:, None], logits_main_b, logits_opp_b)

        # Forward #3: Value from learner perspective (always needed for critic)
        # We can reuse current player encoding if learner is acting, but need separate
        # encoding if opponent is acting. To keep it simple and avoid per-env branching,
        # we always do a separate forward from learner perspective for value.
        # This is unavoidable: critic MUST see from learner perspective.
        card_learner_b, action_hist_learner_b, action_mask_learner_b, _ = jax.vmap(encode_obs)(
            s, aux, learner_pid
        )
        _, value_b = forward_apply(
            main_params, card_learner_b, action_hist_learner_b, action_mask_learner_b
        )
        value = value_b.astype(jnp.float32)
        
        # Sample actions
        def _mk_key(k, eid, salt):
            k = jax.random.fold_in(k, eid)
            return jax.random.fold_in(k, t + jnp.uint32(salt))
        
        k_act = jax.vmap(lambda k, eid: _mk_key(k, eid, 0))(base_keys, env_ids)
        g = jax.vmap(lambda k: jax.random.gumbel(k, (NUM_ACTIONS,), dtype=jnp.float32))(k_act)
        # Use -1e9 instead of jnp.finfo(float32).min to avoid numerical issues
        neg_inf = -1e9
        masked_logits = jnp.where(legal_a_b, logits, neg_inf)
        action = jnp.argmax(masked_logits + g, axis=-1).astype(jnp.int32)
        
        # Compute log probabilities
        logp_all = jax.nn.log_softmax(masked_logits, axis=-1)
        logp = jnp.take_along_axis(logp_all, action[:, None], axis=-1)[:, 0].astype(jnp.float32)
        logp_old = jnp.where(is_learner_turn, logp, jnp.float32(0.0))
        
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

        tmp = {
            "card": card_cur_b,
            "action_hist": action_hist_cur_b,
            "action_mask": action_mask_cur_b,
            "card_v": card_learner_b,
            "action_hist_v": action_hist_learner_b,
            "action_mask_v": action_mask_learner_b,
            "legal": legal_a_b,
            "action": action,
            "logp_old": logp_old,
            "value": value,
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
        return (buf, vec2), tmp
    
    (buf, vec2), traj = jax.lax.scan(
        step_fn,
        (buf, vec),
        jnp.arange(CFG.rollout_len, dtype=jnp.uint32),
    )
    
    # Advance RNGs
    key, sub = jax.random.split(key)
    vec2 = vec2.replace(rng=jax.random.split(sub, CFG.num_envs))
    
    # Compute GAE
    rewards = traj["reward"].astype(jnp.float32)
    values = traj["value"].astype(jnp.float32)
    dones = traj["done"]
    
    # Last value prediction
    last_pid = jnp.where(vec2.main_is_p0, jnp.int32(0), jnp.int32(1))
    last_card_b, last_action_hist_b, last_action_mask_b, _ = jax.vmap(encode_obs)(
        vec2.env_state, vec2.aux, last_pid
    )
    _, last_value = forward_apply(main_params, last_card_b, last_action_hist_b, last_action_mask_b)
    _, last_value = forward_apply(main_params, last_card_b, last_action_hist_b, last_action_mask_b)
    last_value = last_value.astype(jnp.float32)
    
    def _one(rew_n, val_n, done_n, last_v_n):
        return compute_gae_and_returns(rew_n, val_n, done_n, last_v_n)
    
    adv, ret = jax.vmap(_one, in_axes=(1, 1, 1, 0), out_axes=(1, 1))(
        rewards, values, dones, last_value
    )
    
    # Flatten trajectory
    flat = lambda x: x.reshape((-1,) + x.shape[2:])
    
    tr_batch = Transition(
        card=flat(traj["card"]),
        action_hist=flat(traj["action_hist"]),
        action_mask=flat(traj["action_mask"]),
        card_v=flat(traj["card_v"]),
        action_hist_v=flat(traj["action_hist_v"]),
        action_mask_v=flat(traj["action_mask_v"]),
        legal=flat(traj["legal"]),
        action=flat(traj["action"]).reshape(-1),
        logp_old=flat(traj["logp_old"]).reshape(-1),
        reward=flat(traj["reward"]).reshape(-1),
        done=flat(traj["done"]).reshape(-1),
        adv=flat(adv).reshape(-1),
        ret=flat(ret).reshape(-1),
        contrib_self=flat(traj["contrib_self"]).reshape(-1),
        contrib_opp=flat(traj["contrib_opp"]).reshape(-1),
        valid=flat(traj["valid"]).reshape(-1),
    )
    
    buf = replay_add_batch(buf, tr_batch)
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
        a, _, _, key = policy_act(params, s, aux, key)
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
    """Create JIT-compiled update function."""
    
    @jax.jit
    def update_phase(
        ts: TrainState,
        buf: ReplayBuffer,
        key: chex.PRNGKey,
    ) -> Tuple[TrainState, Dict[str, chex.Array], chex.PRNGKey]:
        
        def _epoch_step(carry, _):
            ts, key = carry
            
            # Sample batch
            batch, key = replay_sample(buf, key, CFG.batch_size)
            
            # Normalize advantages
            vmask = batch.valid.astype(jnp.float32)
            denom = jnp.maximum(1.0, jnp.sum(vmask))
            adv_mean = jnp.sum(batch.adv * vmask) / denom
            adv_var = jnp.sum(((batch.adv - adv_mean) ** 2) * vmask) / denom
            norm_adv = (batch.adv - adv_mean) / (jnp.sqrt(adv_var) + 1e-8)
            batch = batch.replace(adv=norm_adv)
            
            # Shuffle for minibatches
            key, sub = jax.random.split(key)
            perm = jax.random.permutation(sub, CFG.batch_size)
            
            mb_size = CFG.batch_size // CFG.num_minibatches
            
            def _minibatch_step(ts_local, i):
                idxs = jax.lax.dynamic_slice(perm, (i * mb_size,), (mb_size,))
                b_mini = jax.tree_util.tree_map(lambda x: x[idxs], batch)
                new_ts, metrics = train_step(ts_local, b_mini)
                return new_ts, metrics
            
            ts, metrics_stack = jax.lax.scan(
                _minibatch_step,
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
    
    ts = make_train_state(k_init)
    buf = replay_init(CFG.replay_capacity)
    
    # Count parameters
    num_params = sum(x.size for x in jax.tree_util.tree_leaves(ts.params))
    print(f"Model parameters: {num_params:,}")
    
    # Initialize checkpointing
    ckpt_dir = os.path.abspath(CFG.checkpoint_dir)
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpter = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(ts)
    
    # Initialize pool
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
    
    # Warmup replay buffer
    print("Warming up replay buffer...")
    warm_updates = max(1, CFG.warmup_steps // (CFG.num_envs * CFG.rollout_len))
    for _ in range(warm_updates):
        buf, vec, key = do_collect(ts.params, pool, buf, vec, key)
    print(f"Buffer size after warmup: {buf.size}")
    
    # Training loop
    print("Starting training...")
    import time
    start_time = time.time()
    
    for upd in range(CFG.total_updates):
        # Collect data
        buf, vec, key = do_collect(ts.params, pool, buf, vec, key)
        
        # Update policy
        ts, metrics, key = do_update(ts, buf, key)
        
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
                f"Clip: {m['clip_frac']:.3f} | "
                f"VFit: {m['value_fit_ratio']:.3f}"
            )
        
        # Checkpointing
        if (upd + 1) % CFG.checkpoint_every == 0:
            save_path = f"{ckpt_dir}/step_{upd+1}"
            ckpter.save(save_path, ts, save_args=save_args)
            print(f"Saved checkpoint to {save_path}")
    
    print("Training complete!")


if __name__ == "__main__":
    main()