"""AlphaHoldem-style training on Toss-or-Hold'em (JAX/Flax).

Goal: stay as faithful as possible to the AlphaHoldem paper while adapting to the
Toss-or-Hold'em variant implemented in game.py.

Key components:
  1) Card/action tensor state representation.
  2) Actor-critic with Trinal-Clip PPO (policy + clipped value target).
  3) Off-policy training from a replay buffer.
  4) K-Best self-play with ELO-based opponent pool selection.
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
from flax.traverse_util import flatten_dict

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
)


# -------------------------
# 1) Configuration
# -------------------------


@dataclass(frozen=True)
class Config:
    # Parallel sampling
    num_envs: int = 1_024
    rollout_len: int = 128

    # PPO/GAE
    gamma: float = 0.999
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    delta1: float = 3.0
    ent_coef: float = 0.01
    vf_coef: float = 0.5

    # Optimization
    lr: float = 5e-4
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


CFG = Config()

# Shared env instance (avoid re-instantiating inside jit)
ENV = TossHoldemEnv()


# -------------------------
# 2) Representation
# -------------------------


NB_ACTIONS = 13
# Card tensor channels: add a dedicated channel for revealed discarded cards (board[2:4]).
CARD_CHANS = 7
# Toss-or-Hold'em introduces an additional DISCARD round before FLOP betting.
# We model 5 rounds with 6 slots per round.
ACT_CHANS = 30
SUITS = 4
RANKS = 13
_CARD_LUT_NP = np.zeros((52, SUITS, RANKS), dtype=np.float32)
for c in range(52):
    _CARD_LUT_NP[c, c // 13, c % 13] = 1.0

CARD_LUT = jnp.asarray(_CARD_LUT_NP)

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

@jax.jit
def _cards_to_plane(cards: chex.Array) -> chex.Array:
    """Fast gather-based version using CARD_LUT. cards in [-1, 51]."""

    cards_i = cards.astype(jnp.int32)
    valid = (cards_i >= 0) & (cards_i < 52)

    idx = jnp.clip(cards_i, 0, 51)          # safe for gather
    planes = CARD_LUT[idx]                  # (..., 4, 13)
    mask = valid.astype(jnp.float32)[..., None, None]

    return jnp.sum(planes * mask, axis=0)

@chex.dataclass(frozen=True)
class AuxState:
    """Extra per-env state to build AlphaHoldem tensors."""

    # Action history tensor in TPU-friendly NHWC layout: (4, NB_ACTIONS, ACT_CHANS)
    # Height=4 rows: [p0, p1, sum, legal] (absolute semantics)
    action_tensor: chex.Array
    round_idx: chex.Array
    slot_idx: chex.Array


@jax.jit
def aux_reset() -> AuxState:
    return AuxState(
        action_tensor=jnp.zeros((4, NB_ACTIONS, ACT_CHANS), dtype=jnp.float32),
        round_idx=jnp.int32(0),
        slot_idx=jnp.int32(0),
    )


@jax.jit
def aux_maybe_advance_round(aux: AuxState, phase: chex.Array) -> AuxState:
    new_round = _phase_to_round_idx(phase)
    changed = new_round != aux.round_idx
    return aux.replace(
        round_idx=new_round,
        slot_idx=jnp.where(changed, jnp.int32(0), aux.slot_idx),
    )


@jax.jit
def aux_record_action(
    aux: AuxState,
    player_id: chex.Array,
    action: chex.Array,
    pot: chex.Array,
    legal_mask: chex.Array,
) -> AuxState:
    """Record an action into the AlphaHoldem-style action tensor.

    30 channels = 5 rounds * 6 sequential actions (PREFLOP, DISCARD, FLOP, TURN, RIVER).
    If there are more than 6 sequential actions in a round, keep the most recent 6.

    IMPORTANT: action tensor rows are absolute (p0, p1, sum, legal), not ego-relative.
    """

    base = aux.round_idx * 6
    chan = base + jnp.minimum(aux.slot_idx, jnp.int32(5))

    def _shift_and_write(tensor_hwk):
        # Shift the 6-slot segment inside the round along the channel axis (last dim).
        seg = jax.lax.dynamic_slice(tensor_hwk, (0, 0, base), (4, NB_ACTIONS, 6))
        # seg = jnp.concatenate([seg[:, :, 1:], seg[:, :, -1:]], axis=2)
        return jax.lax.dynamic_update_slice(tensor_hwk, seg, (0, 0, base))

    action_tensor = jax.lax.cond(aux.slot_idx >= 6, _shift_and_write, lambda t: t, aux.action_tensor)

    a_oh = jax.nn.one_hot(action.astype(jnp.int32), NB_ACTIONS, dtype=jnp.float32)
    legal_f = legal_mask.astype(jnp.float32)

    row_p0 = jnp.where(player_id == 0, a_oh, jnp.zeros_like(a_oh))
    row_p1 = jnp.where(player_id == 1, a_oh, jnp.zeros_like(a_oh))
    row_pot = a_oh * pot

    chan_mat = jnp.stack([row_p0, row_p1, row_pot, legal_f], axis=0)  # (4, NB_ACTIONS)
    action_tensor = action_tensor.at[:, :, chan].set(chan_mat)

    return aux.replace(action_tensor=action_tensor, slot_idx=aux.slot_idx + 1)


@jax.jit
def encode_obs(state: State, aux: AuxState, player_id: chex.Array) -> Tuple[chex.Array, chex.Array, chex.Array]:
    """Build AlphaHoldem card/action tensors + legal mask from a state.

    Returns:
      card_tensor: (4,13,CARD_CHANS)  (NHWC)
      action_tensor: (4,NB_ACTIONS,ACT_CHANS)  (NHWC)
      legal_mask: (NB_ACTIONS,) bool

    Card channels (variant-aware):
      0: hole cards (masked if discarded)
      1: community flop (2 cards: board[0:2])
      2: community turn (board[4])
      3: community river (board[5])
      4: revealed discarded cards (board[2:4])
      5: all public (community + revealed discards)
      6: all (hole + all public)
    """

    is_p0 = player_id == 0
    hole = jnp.where(is_p0, state.p0_hole, state.p1_hole)
    active = jnp.where(is_p0, state.p0_active_mask, state.p1_active_mask)
    hole_active = jnp.where(active, hole, jnp.int32(-1))

    flop_cards = state.board[0:2]
    discards = state.board[2:4]
    turn_card = state.board[4:5]
    river_card = state.board[5:6]

    # All public information in this variant includes the revealed discard cards.
    public_cards = jnp.where(state.board >= 0, state.board, jnp.int32(-1))

    ch0_hole = _cards_to_plane(hole)
    ch1_flop = _cards_to_plane(jnp.where(flop_cards >= 0, flop_cards, jnp.int32(-1)))
    ch2_turn = _cards_to_plane(jnp.where(turn_card >= 0, turn_card, jnp.int32(-1)))
    ch3_river = _cards_to_plane(jnp.where(river_card >= 0, river_card, jnp.int32(-1)))
    ch4_disc = _cards_to_plane(jnp.where(discards >= 0, discards, jnp.int32(-1)))
    ch5_public = _cards_to_plane(public_cards)
    ch6_all = _cards_to_plane(jnp.concatenate([hole_active, public_cards], axis=0))

    # Build card tensor directly in NHWC (4,13,C).
    card_tensor = jnp.stack(
        [ch0_hole, ch1_flop, ch2_turn, ch3_river, ch4_disc, ch5_public, ch6_all],
        axis=-1,
    )

    # Action tensor already stored in NHWC (4,13,K) where K=ACT_CHANS.
    action_tensor = aux.action_tensor
    
    def make_ego_relative(tensor, player_id):
        SWAP01 = jnp.array([1, 0, 2, 3], dtype=jnp.int32)
        swapped = jnp.take(tensor, SWAP01, axis=0)

        return jnp.where(player_id == 0, tensor, swapped)
    
    action_tensor = make_ego_relative(action_tensor, player_id)
    legal = ENV.get_legal_actions(state)
    return card_tensor, action_tensor, legal


# -------------------------
# 3) Network (ResNet + FC)
# -------------------------


class ResidualBlock(nn.Module):
    channels: int

    @nn.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        y = nn.Conv(
            self.channels,
            kernel_size=(3, 3),
            padding="SAME",
            use_bias=True,
            dtype=jnp.bfloat16,
            param_dtype=jnp.float32,
        )(x)
        y = nn.relu(y)
        y = nn.Conv(
            self.channels,
            kernel_size=(3, 3),
            padding="SAME",
            use_bias=True,
            dtype=jnp.bfloat16,
            param_dtype=jnp.float32,
        )(y)
        return nn.relu(x + y)


class AlphaHoldemNet(nn.Module):
    """Conv-ResNet trunk + large FC head (paper-style)."""

    # TPU MXU-friendly widths (multiples of 128 where it matters).
    trunk_channels: int = 128
    num_blocks: int = 6
    fc_dim: int = 512

    @nn.compact
    def __call__(
        self,
        card_tensor: chex.Array,  # (B,4,13,7)   NHWC
        action_tensor: chex.Array,  # (B,4,13,ACT_CHANS) NHWC
    ) -> Tuple[chex.Array, chex.Array]:
        # Fuse embeddings
        # Mixed precision: keep params in fp32, run matmuls in bf16 on TPU.
        card_emb = nn.Dense(128, dtype=jnp.bfloat16, param_dtype=jnp.float32)(card_tensor)
        action_emb = nn.Dense(128, dtype=jnp.bfloat16, param_dtype=jnp.float32)(action_tensor)

        x = card_emb + action_emb

        # Ensure downstream ops stay in bf16.
        x = jax.lax.convert_element_type(x, jnp.bfloat16)

        x = nn.Conv(
            self.trunk_channels,
            kernel_size=(3, 3),
            padding="SAME",
            use_bias=True,
            dtype=jnp.bfloat16,
            param_dtype=jnp.float32,
        )(x)
        x = nn.relu(x)

        for _ in range(self.num_blocks):
            x = ResidualBlock(self.trunk_channels)(x)

        x = x.reshape((x.shape[0], -1))

        x = nn.Dense(self.fc_dim, dtype=jnp.bfloat16, param_dtype=jnp.float32)(x)
        x = nn.relu(x)
        x = nn.Dense(self.fc_dim, dtype=jnp.bfloat16, param_dtype=jnp.float32)(x)
        x = nn.relu(x)

        logits = nn.Dense(NB_ACTIONS, dtype=jnp.bfloat16, param_dtype=jnp.float32)(x)  # (B,13)
        value = nn.Dense(1, dtype=jnp.bfloat16, param_dtype=jnp.float32)(x)[:, 0]  # (B,)

        # Loss computations should happen in float32.
        logits = logits.astype(jnp.float32)
        value = value.astype(jnp.float32)
        return logits, value


def masked_categorical(logits: chex.Array, legal: chex.Array) -> distrax.Categorical:
    neg_inf = jnp.finfo(jnp.float32).min
    masked_logits = jnp.where(legal, logits, neg_inf)
    return distrax.Categorical(logits=masked_logits)


# -------------------------
# 4) Replay Buffer
# -------------------------


@chex.dataclass(frozen=True)
class Transition:
    # TPU-friendly NHWC tensors
    card: chex.Array  # (B?,4,13,CARD_CHANS) float32
    act: chex.Array  # (B?,4,13,ACT_CHANS) float32
    # Critic-view encoding (learner perspective on every timestep)
    card_v: chex.Array  # (B?,4,13,CARD_CHANS) float32
    act_v: chex.Array  # (B?,4,13,ACT_CHANS) float32
    legal: chex.Array  # (B?,13) bool
    action: chex.Array  # (B?) int32
    logp_old: chex.Array  # (B?) float32
    reward: chex.Array
    done: chex.Array
    adv: chex.Array
    ret: chex.Array  # discounted return R^gamma
    contrib_self: chex.Array
    contrib_opp: chex.Array
    valid: chex.Array


@chex.dataclass(frozen=True)
class ReplayBuffer:
    # Store sparse/binary tensors as uint8 for memory.
    card_u8: chex.Array  # (cap,4,13,CARD_CHANS) uint8
    act_u8: chex.Array  # (cap,4,13,ACT_CHANS) uint8
    # Critic-view encoding (learner perspective)
    card_v_u8: chex.Array  # (cap,4,13,CARD_CHANS) uint8
    act_v_u8: chex.Array  # (cap,4,13,ACT_CHANS) uint8
    legal: chex.Array  # (cap,13) bool
    action: chex.Array
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
    return ReplayBuffer(
        # NHWC: (4,13,C)
        card_u8=jnp.zeros((capacity, SUITS, RANKS, CARD_CHANS), dtype=jnp.uint8),
        act_u8=jnp.zeros((capacity, 4, NB_ACTIONS, ACT_CHANS), dtype=jnp.uint8),
        card_v_u8=jnp.zeros((capacity, SUITS, RANKS, CARD_CHANS), dtype=jnp.uint8),
        act_v_u8=jnp.zeros((capacity, 4, NB_ACTIONS, ACT_CHANS), dtype=jnp.uint8),
        legal=jnp.zeros((capacity, NB_ACTIONS), dtype=jnp.bool_),
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
    """Add a batch of transitions to the replay buffer in one vectorized update.

    `tr` fields must have leading dimension B.
    """

    cap = buf.card_u8.shape[0]
    B = tr.card.shape[0]
    idxs = (buf.ptr + jnp.arange(B, dtype=jnp.int32)) % cap

    card_u8 = (tr.card > 0).astype(jnp.uint8)
    act_u8 = (tr.act > 0).astype(jnp.uint8)
    card_v_u8 = (tr.card_v > 0).astype(jnp.uint8)
    act_v_u8 = (tr.act_v > 0).astype(jnp.uint8)

    buf2 = buf.replace(
        card_u8=buf.card_u8.at[idxs].set(card_u8),
        act_u8=buf.act_u8.at[idxs].set(act_u8),
        card_v_u8=buf.card_v_u8.at[idxs].set(card_v_u8),
        act_v_u8=buf.act_v_u8.at[idxs].set(act_v_u8),
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


@jax.jit
def replay_add(buf: ReplayBuffer, tr: Transition) -> ReplayBuffer:
    idx = buf.ptr
    cap = buf.card_u8.shape[0]

    card_u8 = (tr.card > 0).astype(jnp.uint8)
    act_u8 = (tr.act > 0).astype(jnp.uint8)
    card_v_u8 = (tr.card_v > 0).astype(jnp.uint8)
    act_v_u8 = (tr.act_v > 0).astype(jnp.uint8)

    buf = buf.replace(
        card_u8=buf.card_u8.at[idx].set(card_u8),
        act_u8=buf.act_u8.at[idx].set(act_u8),
        card_v_u8=buf.card_v_u8.at[idx].set(card_v_u8),
        act_v_u8=buf.act_v_u8.at[idx].set(act_v_u8),
        legal=buf.legal.at[idx].set(tr.legal),
        action=buf.action.at[idx].set(tr.action),
        logp_old=buf.logp_old.at[idx].set(tr.logp_old),
        reward=buf.reward.at[idx].set(tr.reward),
        done=buf.done.at[idx].set(tr.done),
        adv=buf.adv.at[idx].set(tr.adv),
        ret=buf.ret.at[idx].set(tr.ret),
        contrib_self=buf.contrib_self.at[idx].set(tr.contrib_self),
        contrib_opp=buf.contrib_opp.at[idx].set(tr.contrib_opp),
        valid=buf.valid.at[idx].set(tr.valid),
        ptr=(idx + 1) % cap,
        size=jnp.minimum(buf.size + 1, cap),
    )
    return buf


@partial(jax.jit, static_argnums=(2,))
def replay_sample(buf: ReplayBuffer, key: chex.PRNGKey, batch_size: int) -> Tuple[Transition, chex.PRNGKey]:
    key, sub = jax.random.split(key)
    
    cap = buf.card_u8.shape[0]
    
    # 1. Determine the effective window size
    # If recent_window > 0, we cap the lookback. 
    # Otherwise (0), we look back at the entire buffer size.
    limit = jnp.where(
        CFG.recent_window > 0, 
        jnp.int32(CFG.recent_window), 
        cap
    )
    
    # We can't look back further than the number of items actually in the buffer
    available = jnp.minimum(buf.size, limit)
    
    # 2. Generate random offsets relative to the 'head' (buf.ptr)
    # We want indices: [ptr-1, ptr-2, ..., ptr-available]
    offsets = jax.random.randint(sub, (batch_size,), 0, available)
    
    # 3. Calculate circular indices
    # (ptr - 1) is the index of the most recently inserted item.
    # We subtract the random offset.
    # Modulo cap handles the wraparound (Python/JAX % operator handles negatives correctly: -1 % 10 = 9)
    idx = (buf.ptr - 1 - offsets) % cap

    # De-quantize
    tr = Transition(
        card=buf.card_u8[idx].astype(jnp.float32),
        act=buf.act_u8[idx].astype(jnp.float32),
        card_v=buf.card_v_u8[idx].astype(jnp.float32),
        act_v=buf.act_v_u8[idx].astype(jnp.float32),
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


# -------------------------
# 5) Self-play pool (K-Best)
# -------------------------


@chex.dataclass(frozen=True)
class AgentPool:
    params: Any  # params stacked along axis 0 per leaf
    elo: chex.Array
    size: chex.Array


def pool_init(dummy_params: Any) -> AgentPool:
    def _zeros_like(p):
        return jnp.zeros((CFG.pool_max,) + p.shape, dtype=p.dtype)

    params = jax.tree_util.tree_map(_zeros_like, dummy_params)
    elo = jnp.zeros((CFG.pool_max,), dtype=jnp.float32)
    return AgentPool(params=params, elo=elo, size=jnp.int32(0))


@jax.jit
def pool_add(pool: AgentPool, new_params: Any, initial_elo: chex.Array) -> AgentPool:
    size = pool.size
    can_append = size < CFG.pool_max

    worst = jnp.argmin(jnp.where(jnp.arange(CFG.pool_max) < size, pool.elo, 1e9))
    write_idx = jnp.where(can_append, size, worst)

    def _write_leaf(pool_leaf, new_leaf):
        return pool_leaf.at[write_idx].set(new_leaf)

    params = jax.tree_util.tree_map(_write_leaf, pool.params, new_params)                                                                
    elo = pool.elo.at[write_idx].set(initial_elo)
    size2 = jnp.where(can_append, size + 1, size)
    return pool.replace(params=params, elo=elo, size=size2)


def elo_expected(ra: chex.Array, rb: chex.Array) -> chex.Array:
    return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))


@jax.jit
def elo_update(ra: chex.Array, rb: chex.Array, score_a: chex.Array, k: float) -> Tuple[chex.Array, chex.Array]:
    ea = elo_expected(ra, rb)
    eb = 1.0 - ea
    ra2 = ra + k * (score_a - ea)
    rb2 = rb + k * ((1.0 - score_a) - eb)
    return ra2, rb2


@jax.jit
def pool_select_opponent(pool: AgentPool, key: chex.PRNGKey) -> Tuple[Any, chex.PRNGKey]:
    key, sub = jax.random.split(key)
    size = pool.size
    valid_elo = jnp.where(jnp.arange(CFG.pool_max) < size, pool.elo, -1e9)
    # Use a fixed-K top_k to keep slice sizes static under JIT.
    _, topk_idx = jax.lax.top_k(valid_elo, k=CFG.k_best)
    k_eff = jnp.minimum(size, jnp.int32(CFG.k_best))
    pick = jax.random.randint(sub, (), 0, jnp.maximum(k_eff, 1))
    idx = topk_idx[pick]
    opp_params = jax.tree_util.tree_map(lambda p: p[idx], pool.params)
    return opp_params, key


# -------------------------
# 6) Rewards (chip delta)
# -------------------------


@jax.jit
def terminal_rewards(state: State) -> Tuple[chex.Array, chex.Array]:
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


# -------------------------
# 7) PPO (Trinal-Clip)
# -------------------------


@jax.jit
def trinal_clip_policy_objective(rt: chex.Array, adv: chex.Array, eps: float, delta1: float) -> chex.Array:
    """Trinal-Clip PPO objective per-sample.

    Eq. (3) in AlphaHoldem uses the *trinal-clipped ratio*:
      r_tcp = clip(rt, clip(rt, 1-eps, 1+eps), delta1)

    and the policy objective is r_tcp * A_t (no extra PPO min/max between
    candidates).
    """

    rt_clip = jnp.clip(rt, 1.0 - eps, 1.0 + eps)
    rt_tcp = jnp.clip(rt, rt_clip, delta1)
    return rt_tcp * adv


@jax.jit
def trinal_clip_value_target(ret: chex.Array, contrib_self: chex.Array, contrib_opp: chex.Array) -> chex.Array:
    """Clip discounted return to [-d2, d3], where d2/d3 are chips contributed."""

    d2 = contrib_self.astype(jnp.float32)
    d3 = contrib_opp.astype(jnp.float32)
    return jnp.clip(ret, -d2, d3)


def make_train_state(rng: chex.PRNGKey) -> TrainState:
    model = AlphaHoldemNet()

    # NHWC layouts that match encode_obs() + AlphaHoldemNet.__call__()
    dummy_card = jnp.zeros((1, SUITS, RANKS, CARD_CHANS), dtype=jnp.float32)          # (1,4,13,7)
    dummy_act  = jnp.zeros((1, 4, NB_ACTIONS, ACT_CHANS), dtype=jnp.float32)         # (1,4,13,30)

    variables = model.init(rng, dummy_card, dummy_act)

    tx = optax.chain(
        optax.clip_by_global_norm(CFG.max_grad_norm),
        optax.adam(CFG.lr),
    )
    return TrainState.create(apply_fn=model.apply, params=variables["params"], tx=tx)


@jax.jit
def forward_apply(params: Any, card_b: chex.Array, act_b: chex.Array) -> Tuple[chex.Array, chex.Array]:
    """Forward pass (no BatchNorm)."""

    model = AlphaHoldemNet()
    logits, value = model.apply({"params": params}, card_b, act_b)
    return logits, value


@jax.jit
def ppo_loss_and_metrics(state: TrainState, batch: Transition) -> Tuple[chex.Array, Dict[str, chex.Array]]:
    """Compute loss + metrics."""

    logits, _ = forward_apply(state.params, batch.card, batch.act)
    _, value = forward_apply(state.params, batch.card_v, batch.act_v)

    dist = masked_categorical(logits, batch.legal)
    logp = dist.log_prob(batch.action)
    log_ratio = logp - batch.logp_old
    rt = jnp.where(
    batch.valid,
        jnp.exp(log_ratio),
        1.0,   # neutral ratio for invalid samples
    )

    obj = trinal_clip_policy_objective(rt, batch.adv, CFG.clip_eps, CFG.delta1)

    # AlphaHoldem clips the discounted return R^gamma_t into [-d2, d3]
    target = trinal_clip_value_target(batch.ret, batch.contrib_self, batch.contrib_opp)
    v_loss = (target - value) ** 2
    err = target - value
    rmse = jnp.sqrt(jnp.mean(err ** 2))

    t_mean = jnp.mean(target)
    t_var = jnp.mean((target - t_mean) ** 2)
    t_std = jnp.sqrt(t_var) + 1e-8

    value_fit_ratio = rmse / t_std
    ent = dist.entropy()

    valid_f = batch.valid.astype(jnp.float32)
    denom = jnp.maximum(1.0, jnp.sum(valid_f))
    # NOTE: policy loss is averaged over learner-turn samples only (denom).
    # We still train the critic on all states, but scale its average so the
    # critic doesn't implicitly get a larger weight just because it sees more
    # samples than the actor.
    policy_loss = -jnp.sum(obj * valid_f) / denom
    # Train the critic on *all* states (including opponent-turn states), since the
    # rollout/GAE uses V(s) at every timestep.
    denom = jnp.sum(valid_f) + 1e-8                     # valid_f is 0/1 float mask
    value_loss = jnp.mean(v_loss)

    entropy = jnp.sum(ent * valid_f) / denom

    total = policy_loss + CFG.vf_coef * value_loss - CFG.ent_coef * entropy

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
        "clip_frac": jnp.sum(((jnp.abs(rt - 1.0) > CFG.clip_eps).astype(jnp.float32)) * valid_f) / denom,
        "valid_frac": jnp.mean(valid_f),
        "value_fit_ratio": value_fit_ratio
    }
    return total, metrics


@jax.jit
def train_step(state: TrainState, batch: Transition) -> Tuple[TrainState, Dict[str, chex.Array]]:
    def _loss_and_metrics(params, batch_):
        return ppo_loss_and_metrics(state.replace(params=params), batch_)

    (loss, metrics), grads = jax.value_and_grad(_loss_and_metrics, has_aux=True)(state.params, batch)
    new_state = state.apply_gradients(grads=grads)
    metrics = dict(metrics)
    metrics["loss"] = loss
    return new_state, metrics


# -------------------------
# 8) Rollout collection
# -------------------------


@chex.dataclass(frozen=True)
class VecEnvState:
    env_state: State
    aux: AuxState
    # If True, main agent plays as player0; else it plays as player1.
    main_is_p0: chex.Array
    # Per-env PRNG keys (avoid splitting a single global key each step).
    rng: chex.Array


def vec_reset(env: TossHoldemEnv, key: chex.PRNGKey, n: int) -> Tuple[VecEnvState, chex.PRNGKey]:
    # Randomize which seat the main policy plays in for each parallel env.
    key, k_seat, k_env = jax.random.split(key, 3)
    main_is_p0 = jax.random.bernoulli(k_seat, 0.5, (n,))

    ks = jax.random.split(k_env, n)
    s = jax.vmap(env.reset)(ks)
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
    """Sample action for the current player in `state`."""

    pid = state.current_player.astype(jnp.int32)
    card, act, legal = encode_obs(state, aux, pid)

    # Add batch dimension
    card_b = card[None, ...]
    act_b = act[None, ...]

    logits_b, value_b = forward_apply(params, card_b, act_b)
    logits = logits_b[0]
    value = value_b[0]

    dist = masked_categorical(logits, legal)
    # `key` is already unique per-env and per-timestep (derived outside this fn).
    action = dist.sample(seed=key)
    logp = dist.log_prob(action)
    return action.astype(jnp.int32), logp.astype(jnp.float32), value.astype(jnp.float32), key


@jax.jit
def env_step_and_optional_reset(
    s: State,
    key: chex.PRNGKey,
    action: chex.Array,
) -> Tuple[State, State, chex.Array, chex.PRNGKey]:
    """Step one env and then (optionally) reset after producing terminal transition."""

    # Avoid per-step `random.split` on TPU; derive subkeys deterministically.
    k_step = jax.random.fold_in(key, jnp.uint32(0))
    k_reset = jax.random.fold_in(key, jnp.uint32(1))
    s_next = ENV.step(s, k_step, action)
    done = s_next.done
    s_reset = ENV.reset(k_reset)
    s_cont = jax.tree_util.tree_map(lambda a, b: jnp.where(done, a, b), s_reset, s_next)
    return s_next, s_cont, done, key


@jax.jit
def reset_aux_where_done(aux: AuxState, done: chex.Array) -> AuxState:
    done_at = done[:, None, None, None]
    at = jnp.where(done_at, jnp.zeros_like(aux.action_tensor), aux.action_tensor)
    round_idx = jnp.where(done, jnp.zeros_like(aux.round_idx), aux.round_idx)
    slot_idx = jnp.where(done, jnp.zeros_like(aux.slot_idx), aux.slot_idx)
    return aux.replace(action_tensor=at, round_idx=round_idx, slot_idx=slot_idx)


@jax.jit
def compute_gae_and_returns(
    rewards: chex.Array,
    values: chex.Array,
    dones: chex.Array,
    last_value: chex.Array,
) -> Tuple[chex.Array, chex.Array]:
    """Compute (GAE advantages, discounted returns R^gamma) over time axis 0.

    AlphaHoldem uses the discounted return R^gamma_t as the value target.
    We still use GAE advantages for the policy gradient.
    """

    T = rewards.shape[0]
    values_next = jnp.concatenate([values[1:], last_value[None]], axis=0)
    deltas = rewards + CFG.gamma * (1.0 - dones.astype(jnp.float32)) * values_next - values

    def adv_scan(carry, t):
        adv_next = carry
        adv_t = deltas[t] + CFG.gamma * CFG.gae_lambda * (1.0 - dones[t].astype(jnp.float32)) * adv_next
        return adv_t, adv_t

    _, adv_rev = jax.lax.scan(adv_scan, jnp.float32(0.0), jnp.arange(T - 1, -1, -1))
    adv = adv_rev[::-1]

    def ret_scan(carry, t):
        g_next = carry
        g = rewards[t] + CFG.gamma * (1.0 - dones[t].astype(jnp.float32)) * g_next
        return g, g

    _, ret_rev = jax.lax.scan(ret_scan, last_value, jnp.arange(T - 1, -1, -1))
    ret = ret_rev[::-1]

    return adv, ret


@jax.jit
def collect_rollout(
    main_params: Any,
    pool: AgentPool,
    buf: ReplayBuffer,
    vec: VecEnvState,
    key: chex.PRNGKey,
) -> Tuple[ReplayBuffer, VecEnvState, chex.PRNGKey]:
    """Collect CFG.rollout_len transitions per env and add to replay buffer."""

    # Use a single shared env instance inside jit to avoid repeated instantiation.
    env = ENV

    opp_params, key = pool_select_opponent(pool, key)
    # If pool is empty, opponent is current main.
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

        # --- Batched network forward (TPU-friendly) ---
        # 1) Critic/value from learner perspective for *every* env.
        card_v_b, act_v_b, _ = jax.vmap(encode_obs)(s, aux, learner_pid)
        _, value_b = forward_apply(main_params, card_v_b, act_v_b)
        value = value_b.astype(jnp.float32)

        # 2) Policy logits from the *acting* player's perspective.
        pid_cur = s.current_player.astype(jnp.int32)
        card_a_b, act_a_b, legal_a_b = jax.vmap(encode_obs)(s, aux, pid_cur)

        logits_main_b, _ = forward_apply(main_params, card_a_b, act_a_b)
        logits_opp_b, _ = forward_apply(opp_params, card_a_b, act_a_b)
        logits = jnp.where(is_learner_turn[:, None], logits_main_b, logits_opp_b)

        # Derive per-env, per-timestep keys without splitting a single global key.
        def _mk_key(k, eid, salt):
            k = jax.random.fold_in(k, eid)
            return jax.random.fold_in(k, t + jnp.uint32(salt))

        # Sample actions using Gumbel-max on masked logits (vectorized over envs).
        k_act = jax.vmap(lambda k, eid: _mk_key(k, eid, 0))(base_keys, env_ids)
        g = jax.vmap(lambda k: jax.random.gumbel(k, (NB_ACTIONS,), dtype=jnp.float32))(k_act)
        neg_inf = jnp.finfo(jnp.float32).min
        masked_logits = jnp.where(legal_a_b, logits, neg_inf)
        action = jnp.argmax(masked_logits + g, axis=-1).astype(jnp.int32)

        # Log-prob under the selected policy. Only learner-turn samples are valid.
        logp_all = jax.nn.log_softmax(masked_logits, axis=-1)
        logp = jnp.take_along_axis(logp_all, action[:, None], axis=-1)[:, 0].astype(jnp.float32)
        logp_old = jnp.where(is_learner_turn, logp, jnp.float32(0.0))


        # Legal actions depend only on the current state (current_player), so compute once.
        legal = legal_a_b
        pot = s.pot + s.p0_bet + s.p1_bet
        aux2 = jax.vmap(aux_record_action)(aux, s.current_player.astype(jnp.int32), action, pot, legal)

        k_env = jax.vmap(lambda k, eid: _mk_key(k, eid, 131071))(base_keys, env_ids)
        s_next, s_cont, done, _ = jax.vmap(env_step_and_optional_reset)(s, k_env, action)

        # Terminal chip-delta reward, from learner seat perspective
        r0, r1 = terminal_rewards(s_next)
        rew_term = jnp.where(main_is_p0, r0, r1)
        rew = jnp.where(done, rew_term, jnp.float32(0.0))

        def _encode_one(si, auxi):
            # Card/action tensors are encoded from the acting player's view. The legal mask
            # is identical to `legal` computed above, so we don't recompute it here.
            pid = si.current_player.astype(jnp.int32)
            c, a_t, _ = encode_obs(si, auxi, pid)
            return c, a_t

        card_t, act_t = jax.vmap(_encode_one)(s, aux)
        legal_t = legal

        def _encode_v_one(si, auxi, pid_i):
            # Critic encoding is always from learner perspective (pid_i).
            c, a_t, _ = encode_obs(si, auxi, pid_i)
            return c, a_t

        card_v_t, act_v_t = jax.vmap(_encode_v_one)(s, aux, learner_pid)

        # Trinal-Clip value target bounds (-d2, d3) should correspond to the state
        # whose value we're training (V(s_t)). For terminal transitions, use the
        # post-action commitment so we don't miss the final bet/call.
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
            "card": card_t,
            "act": act_t,
            "card_v": card_v_t,
            "act_v": act_v_t,
            "legal": legal_t,
            "action": action,
            "logp_old": logp_old,
            "value": value,
            "reward": rew,
            "done": done,
            "contrib_self": contrib_self,
            "contrib_opp": contrib_opp,
            "valid": is_learner_turn,
        }

        aux3 = reset_aux_where_done(aux2, done)
        aux3 = jax.vmap(aux_maybe_advance_round)(aux3, s_cont.phase)

        # Re-randomize seating on episode resets so we train both positions.
        k_seat = jax.vmap(lambda k, eid: _mk_key(k, eid, 262143))(base_keys, env_ids)
        new_seat = jax.vmap(lambda k: jax.random.bernoulli(k, 0.5))(k_seat)
        main_is_p0_2 = jnp.where(done, new_seat, main_is_p0)

        vec2 = VecEnvState(env_state=s_cont, aux=aux3, main_is_p0=main_is_p0_2, rng=base_keys)
        return (buf, vec2), tmp

    (buf, vec2), traj = jax.lax.scan(step_fn, (buf, vec), jnp.arange(CFG.rollout_len, dtype=jnp.uint32))

    # Advance per-env RNGs once per rollout call to avoid repeating the same fold_in sequence.
    key, sub = jax.random.split(key)
    vec2 = vec2.replace(rng=jax.random.split(sub, CFG.num_envs))

    rewards = traj["reward"].astype(jnp.float32)
    values = traj["value"].astype(jnp.float32)
    dones = traj["done"]



    # Batched last-value prediction (learner perspective) to avoid batch=1 forwards.
    last_pid = jnp.where(vec2.main_is_p0, jnp.int32(0), jnp.int32(1))
    last_card_b, last_act_b, _ = jax.vmap(encode_obs)(vec2.env_state, vec2.aux, last_pid)
    _, last_value = forward_apply(main_params, last_card_b, last_act_b)
    last_value = last_value.astype(jnp.float32)


    def _one(rew_n, val_n, done_n, last_v_n):
        return compute_gae_and_returns(rew_n, val_n, done_n, last_v_n)

    adv, ret = jax.vmap(_one, in_axes=(1, 1, 1, 0), out_axes=(1, 1))(rewards, values, dones, last_value)

    flat = lambda x: x.reshape((-1,) + x.shape[2:])
    card_f = flat(traj["card"])
    act_f = flat(traj["act"])
    card_v_f = flat(traj["card_v"])
    act_v_f = flat(traj["act_v"])
    legal_f = flat(traj["legal"])
    action_f = flat(traj["action"]).reshape(-1)
    logp_f = flat(traj["logp_old"]).reshape(-1)
    r_f = flat(traj["reward"]).reshape(-1)
    d_f = flat(traj["done"]).reshape(-1)
    adv_f = flat(adv).reshape(-1)
    ret_f = flat(ret).reshape(-1)
    cs_f = flat(traj["contrib_self"]).reshape(-1)
    co_f = flat(traj["contrib_opp"]).reshape(-1)
    valid_f = flat(traj["valid"]).reshape(-1)

    tr_batch = Transition(
        card=card_f,
        act=act_f,
        card_v=card_v_f,
        act_v=act_v_f,
        legal=legal_f,
        action=action_f,
        logp_old=logp_f,
        reward=r_f,
        done=d_f,
        adv=adv_f,
        ret=ret_f,
        contrib_self=cs_f,
        contrib_opp=co_f,
        valid=valid_f,
    )

    buf = replay_add_batch(buf, tr_batch)
    return buf, vec2, key


# -------------------------
# 9) Pool evaluation matches
# -------------------------


@jax.jit
def play_one_hand(
    params_a: Any,
    params_b: Any,
    key: chex.PRNGKey,
) -> Tuple[chex.Array, chex.PRNGKey]:
    """Play one hand: player0 uses params_a, player1 uses params_b."""

    # Use the shared (module-level) environment instance. Constructing an env
    # inside a jitted function is not JIT-safe.
    env = ENV
    key, k_reset = jax.random.split(key)
    s = env.reset(k_reset)
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
        # Advance RNG through action sampling.
        a, _, _, key = policy_act(params, s, aux, key)
        legal = env.get_legal_actions(s)
        aux2 = aux_record_action(aux, pid, a, legal)
        key, k_step = jax.random.split(key)
        s2 = env.step(s, k_step, a)
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
    """Vectorized pool eval + *similar-behavior* ELO update (stable magnitude)."""
    
    # Capture config constants to avoid recompilation if passed as args
    pool_max = CFG.pool_max
    eval_hands = CFG.eval_hands
    k_best = CFG.k_best
    elo_k = CFG.elo_k

    # --- 1) Generate keys for (pool_max, eval_hands) ---
    key, sub = jax.random.split(key)
    # Shape: (pool_max, eval_hands, 2)
    keys_grid = jax.random.split(sub, pool_max * eval_hands).reshape((pool_max, eval_hands, -1)) 

    # Valid opponents are [0, pool.size)
    valid_mask = jnp.arange(pool_max) < pool.size
    valid_f = valid_mask.astype(jnp.float32)
    # Avoid div by zero if pool is empty
    n_valid = jnp.maximum(1.0, jnp.sum(valid_f))

    # --- 2) Play matches in parallel: main vs each pool opponent ---
    def play_vs_one_opp(opp_params, keys_n):
        # Play 'eval_hands' matches against ONE opponent
        # vmap over the 'hands' dimension
        scores, _ = jax.vmap(play_one_hand, in_axes=(None, None, 0))(
            main_state.params, opp_params, keys_n
        )
        return jnp.mean(scores)

    # vmap over the 'pool' dimension
    # scores_pool: (pool_max,)
    scores_pool = jax.vmap(play_vs_one_opp, in_axes=(0, 0))(pool.params, keys_grid)

    # Mask out invalid slots (force score 0.0 so they don't impact logic, though masked later anyway)
    scores_pool = jnp.where(valid_mask, scores_pool, 0.0)

    # --- 3) Batch ELO update using *frozen* pre-update ELOs ---
    # Calculate Expected Scores
    ea_pool = elo_expected(main_elo, pool.elo)       # Main's expected vs each opp
    eb_pool = 1.0 - ea_pool

    # Calculate Deltas
    delta_main = elo_k * (scores_pool - ea_pool)
    delta_opp  = elo_k * ((1.0 - scores_pool) - eb_pool)

    # Apply Updates
    # Main: Average the deltas to keep magnitude stable regardless of pool size
    main_change = jnp.sum(delta_main * valid_f) / n_valid
    main_elo_next = main_elo + main_change

    # Pool: Update individual opponents (masked)
    pool_elo_next = pool.elo + jnp.where(valid_mask, delta_opp, 0.0)
    pool = pool.replace(elo=pool_elo_next)

    # --- 4) Add main to pool with entry ELO ---
    # Entry ELO = mean(top-k valid ELOs) (or 0 if empty)
    valid_elos = jnp.where(valid_mask, pool.elo, -1e9)
    _, topk_idx = jax.lax.top_k(valid_elos, k=k_best)

    # Calculate how many valid agents we actually have for the mean
    k_eff = jnp.minimum(pool.size, jnp.int32(k_best))
    topk_mask = (jnp.arange(k_best) < k_eff).astype(jnp.float32)

    topk_vals = pool.elo[topk_idx]
    denom = jnp.maximum(1.0, jnp.sum(topk_mask))
    avg_top_elo = jnp.sum(topk_vals * topk_mask) / denom

    entry_elo = jnp.where(pool.size > 0, avg_top_elo, 0.0)
    
    # Add new agent (increments pool.size)
    pool = pool_add(pool, main_state.params, entry_elo)

    # --- 5) Trim to K-best (Elite Pool) ---
    # We now have (size+1) agents. We strictly keep the top k_best.
    size_curr = pool.size
    # Validity mask for the post-add pool
    valid_mask_final = jnp.arange(pool_max) < size_curr
    valid_elos_final = jnp.where(valid_mask_final, pool.elo, -1e9)

    # Find top K indices
    _, keep_idx = jax.lax.top_k(valid_elos_final, k=k_best)
    n_keep = jnp.minimum(size_curr, jnp.int32(k_best))

    # Helper to gather params and pad back to pool_max
    def _gather_pad(x):
        # Take top-k_best
        picked = x[keep_idx]  # shape: (k_best, ...)
        # Mask out entries if we have fewer than k_best valid
        mask = (jnp.arange(k_best) < n_keep).reshape((-1,) + (1,) * (x.ndim - 1))
        picked = picked * mask.astype(picked.dtype)
        # Pad back to pool_max to keep shapes static
        pad_len = pool_max - k_best
        pad = jnp.zeros((pad_len,) + x.shape[1:], dtype=x.dtype)
        return jnp.concatenate([picked, pad], axis=0)

    new_params = jax.tree_util.tree_map(_gather_pad, pool.params)

    # Gather ELOs similarly
    picked_elo = pool.elo[keep_idx]
    # Zero out ELOs that are "padding" (beyond n_keep)
    picked_elo = picked_elo * (jnp.arange(k_best) < n_keep).astype(jnp.float32)
    new_elo = jnp.concatenate([picked_elo, jnp.zeros(pool_max - k_best, dtype=pool.elo.dtype)], axis=0)

    # Return trimmed pool
    pool_final = pool.replace(params=new_params, elo=new_elo, size=n_keep)
    
    return pool_final, main_elo_next, key



# -------------------------
# 10) Optimized Main Loop (TPU v6-1)
# -------------------------

def make_update_step():
    """
    Creates a JIT-compiled function that runs ALL update epochs and minibatches
    on the TPU in one go. This eliminates Python overhead.
    """
    
    # We define the update logic inside a function to JIT the whole block
    @jax.jit
    def update_phase(
        ts: TrainState,
        buf: ReplayBuffer,
        key: chex.PRNGKey
    ) -> Tuple[TrainState, Dict[str, chex.Array], chex.PRNGKey]:
        
        # 1. Define the loop over epochs (e.g., 4 epochs)
        def _epoch_step(carry, _):
            ts, key = carry
            
            # A. Sample whole batch
            batch, key = replay_sample(buf, key, CFG.batch_size)
            
            # B. Normalize advantages (crucial for PPO stability)
            vmask = batch.valid.astype(jnp.float32)
            denom = jnp.maximum(1.0, jnp.sum(vmask))
            adv_mean = jnp.sum(batch.adv * vmask) / denom
            # Variance calc
            adv_var = jnp.sum(((batch.adv - adv_mean) ** 2) * vmask) / denom
            norm_adv = (batch.adv - adv_mean) / (jnp.sqrt(adv_var) + 1e-8)
            batch = batch.replace(adv=norm_adv)
            
            # C. Shuffle for minibatches
            key, sub = jax.random.split(key)
            perm = jax.random.permutation(sub, CFG.batch_size)
            
            # D. Loop over minibatches (e.g., 8 minibatches)
            mb_size = CFG.batch_size // CFG.num_minibatches
            
            def _minibatch_step(ts_local, i):
                # Slice the batch using the permutation
                idxs = jax.lax.dynamic_slice(perm, (i * mb_size,), (mb_size,))
                
                # Helper to slice the Transition object
                b_mini = jax.tree_util.tree_map(lambda x: x[idxs], batch)
                
                # Perform one gradient step
                new_ts, metrics = train_step(ts_local, b_mini)
                return new_ts, metrics

            # Scan over minibatches
            ts, metrics_stack = jax.lax.scan(_minibatch_step, ts, jnp.arange(CFG.num_minibatches))
            
            # Average metrics across the minibatches for this epoch
            metrics_avg = jax.tree_util.tree_map(lambda x: jnp.mean(x), metrics_stack)
            
            return (ts, key), metrics_avg

        # Scan over epochs
        (ts_new, key_new), metrics_epochs = jax.lax.scan(
            _epoch_step, 
            (ts, key), 
            None, 
            length=CFG.update_epochs
        )
        
        # Average metrics across all epochs
        final_metrics = jax.tree_util.tree_map(lambda x: jnp.mean(x), metrics_epochs)
        
        return ts_new, final_metrics, key_new

    return update_phase


def main() -> None:
    print(f"JAX Devices: {jax.devices()}")

    # TPU efficiency guards (MXU prefers multiples of 128).
    assert CFG.num_envs % 128 == 0, "num_envs should be a multiple of 128 for TPU MXU efficiency"
    assert CFG.rollout_len % 128 == 0, "rollout_len should be a multiple of 128 for TPU MXU efficiency"
    assert CFG.batch_size % 128 == 0, "batch_size should be a multiple of 128 for TPU MXU efficiency"
    assert (CFG.batch_size // CFG.num_minibatches) % 128 == 0, "minibatch size should be a multiple of 128"
    
    # 1. Initialization
    key = jax.random.PRNGKey(CFG.seed)
    key, k_init = jax.random.split(key)
    
    ts = make_train_state(k_init)
    buf = replay_init(CFG.replay_capacity)

    # Initialize checkpointing for the training state.
    ckpt_dir = os.path.abspath("/content/drive/MyDrive/checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpter = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(ts)
    
    # Initialize Pool
    pool = pool_init(ts.params)
    pool = pool_add(pool, ts.params, jnp.float32(0.0))
    main_elo = jnp.float32(0.0)
    # Initialize Environment
    # Note: ENV is global in the original script, which works for single-device JIT.
    key, k_reset = jax.random.split(key)
    vec, key = vec_reset(ENV, k_reset, CFG.num_envs)

    # 2. Compilation Warmup
    # We create the JIT functions once here
    print("Compiling update functions...")
    do_update = make_update_step()
    
    # Optional: JIT the collection wrapper for cleaner call sites
    # (Though collect_rollout is already JIT-friendly, wrapping it ensures the python loop is clean)
    do_collect = jax.jit(collect_rollout)

    # 3. Fill Replay Buffer (Warmup)
    print("Warming up replay buffer...")
    warm_updates = max(1, CFG.warmup_steps // (CFG.num_envs * CFG.rollout_len))
    for _ in range(warm_updates):
        buf, vec, key = do_collect(ts.params, pool, buf, vec, key)
    print(f"Buffer size after warmup: {buf.size}")

    # 4. Training Loop
    print("Starting training...")
    import time
    start_time = time.time()
    
    for upd in range(CFG.total_updates):
        
        # A. Collect Data (Runs completely on TPU)
        #    Output: updated buffer, updated env state, new rng key
        buf, vec, key = do_collect(ts.params, pool, buf, vec, key)

        # B. Update Policy/Critic (Runs completely on TPU)
        #    Output: new params, metrics, new rng key
        ts, metrics, key = do_update(ts, buf, key)

        # C. Periodic Pool Maintenance (Runs on TPU)
        if (upd + 1) % CFG.add_to_pool_every_updates == 0:
            pool, main_elo, key = eval_and_update_pool(pool, ts, main_elo, key)


        # D. Logging (Runs on CPU, minimal overhead)
        if (upd + 1) % 50 == 0:
            # Transfer metrics to CPU only when printing
            # block_until_ready ensures accurate timing
            m = jax.device_get(metrics) 
            
            elapsed = time.time() - start_time
            sps = (50 * CFG.num_envs * CFG.rollout_len) / elapsed
            start_time = time.time()
            
            print(
                f"Upd {upd+1}/{CFG.total_updates} | "
                f"Buf: {int(buf.size)} | Pool: {int(pool.size)} | "
                f"SPS: {sps:.0f} | "
                f"Loss: {m['loss']:.4f} | "
                f"Pol: {m['policy_loss']:.4f} | "
                f"Val: {m['value_loss']:.4f} | "
                f"Ent: {m['entropy']:.4f} | "
                f"Clip: {m['clip_frac']:.3f} |",
                f"VFrac: {m['valid_frac']:.3f} |",
                f"VFit: {m['value_fit_ratio']:.3f}"
            )

        # SAVE CHECKPOINT every 1000 updates
        if (upd + 1) % 1000 == 0:
            save_path = f"{ckpt_dir}/step_{upd+1}"
            ckpter.save(save_path, ts, save_args=save_args)
            print(f"Saved checkpoint to {save_path}")

if __name__ == "__main__":
    main()
