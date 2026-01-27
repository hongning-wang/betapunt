"""Transformer-based IMPALA V-trace training for Toss-or-Hold'em (JAX/Flax).

This training script implements IMPALA-style training with opponent modeling:
  1) Transformer-based card/action encoding
  2) Pure V-trace off-policy correction computed at TRAINING TIME
  3) Short replay window storing complete unrolls for proper V-trace computation
  4) K-Best self-play league training with ELO-based opponent selection
  5) Opponent history encoding - tracks opponent behavior within each rollout

Key architectural choices:
  - Card tensor: (B, 9, 2) - [card_index, type_id] matching model.py CardEncoder
  - Action tensor: (B, 32, 8) - action history matching model.py ActionEncoder (includes is_dealer)
  - History tensor: (B, 64) - opponent statistics tracked during rollout
  - Transformer encoders with cross-attention fusion
  - Critic trained on PRIVATE cards (same info as actor)

Opponent tracking design:
  - Stats are reset at START of each rollout (new opponent selected from pool)
  - Stats accumulate across hands within rollout (same pool opponent for 32 steps)
  - Per-hand state (preflop aggressor, etc.) resets on hand completion
  - Model learns to exploit opponent tendencies from accumulated stats
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import os
import shutil
import time
from typing import Any, Dict, Tuple

import chex
import distrax
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
import orbax.checkpoint as ocp

from model import (
    TossHoldemModel,
    NUM_ACTIONS,
    CARD_SEQ_LEN,
    ACTION_SEQ_LEN,
    CARD_INPUT_DIM,
    ACTION_INPUT_DIM,
    HISTORY_INPUT_DIM,
)

from opponent_stats import (
    VecOpponentStats,
    vec_opponent_stats_init,
    vec_opponent_stats_to_tensor,
    vec_update_opp_stats,
    vec_reset_round_state,
    vec_update_showdown,
    PHASE_PREFLOP,
    PHASE_FLOP,
    PHASE_TURN,
    PHASE_RIVER,
    ACT_FOLD,
)

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
    """IMPALA-style V-trace configuration."""

    num_envs: int = 2_048
    rollout_len: int = 32

    gamma: float = 0.99
    rho_bar: float = 1.0
    c_bar: float = 1.0
    pg_rho_bar: float = 1.0
    vf_coef: float = 0.25

    ent_coef_init: float = 0.01
    ent_coef_final: float = 0.001
    ent_decay_steps: int = 40_000

    illegal_action_coef: float = 0.1

    lr_init: float = 1e-4
    lr_peak: float = 3e-4
    lr_final: float = 1e-5
    warmup_updates: int = 500
    max_grad_norm: float = 0.5

    replay_capacity: int = 200_000
    num_minibatches: int = 1
    update_epochs: int = 1
    warmup_steps: int = 100_000
    recent_window_transitions: int = 500_000

    pool_max: int = 8
    k_best: int = 4
    eval_hands: int = 512
    add_to_pool_every_updates: int = 100
    elo_k: float = 32.0
    elo_tau: float = 100.0

    seed: int = 0
    total_updates: int = 40_000
    block_size: int = 100

    checkpoint_dir: str = "./drive/MyDrive/checkpoints"
    checkpoint_every: int = 1000
    log_every: int = 50
    resume_weights_from: str = ""


CFG = Config()
ENV = TossHoldemEnv()


# =============================================================================
# 2) State Representation
# =============================================================================


STREET_PREFLOP = 0
STREET_DISCARD = 1
STREET_FLOP = 2
STREET_TURN = 3
STREET_RIVER = 4


@jax.jit
def phase_to_street(phase: chex.Array) -> chex.Array:
    return jax.lax.switch(
        phase,
        [
            lambda: jnp.int32(STREET_PREFLOP),
            lambda: jnp.int32(STREET_DISCARD),
            lambda: jnp.int32(STREET_DISCARD),
            lambda: jnp.int32(STREET_FLOP),
            lambda: jnp.int32(STREET_TURN),
            lambda: jnp.int32(STREET_RIVER),
            lambda: jnp.int32(STREET_RIVER),
        ],
    )


@jax.jit
def make_action_token(state: State, action: chex.Array, time_id: chex.Array) -> chex.Array:
    is_p0 = state.current_player == 0
    cur_stack = jnp.where(is_p0, state.p0_stack, state.p1_stack).astype(jnp.float32)
    cur_bet = jnp.where(is_p0, state.p0_bet, state.p1_bet).astype(jnp.float32)
    opp_stack = jnp.where(is_p0, state.p1_stack, state.p0_stack).astype(jnp.float32)
    opp_bet = jnp.where(is_p0, state.p1_bet, state.p0_bet).astype(jnp.float32)

    to_call = (opp_bet - cur_bet).astype(jnp.float32)
    pot_base = (state.pot + state.p0_bet + state.p1_bet + to_call).astype(jnp.float32)
    eff_cap = jnp.minimum(cur_stack, opp_stack + to_call)
    street_id = phase_to_street(state.phase).astype(jnp.float32)
    current_is_dealer = jnp.where(is_p0, state.is_dealer_p0, ~state.is_dealer_p0)
    is_dealer = current_is_dealer.astype(jnp.float32)

    return jnp.array([
        state.current_player.astype(jnp.float32),
        action.astype(jnp.float32),
        street_id,
        time_id.astype(jnp.float32),
        to_call,
        pot_base,
        eff_cap,
        is_dealer,
    ], dtype=jnp.float32)


@chex.dataclass(frozen=True)
class AuxState:
    action_tensor: chex.Array
    action_ptr: chex.Array


@jax.jit
def aux_reset() -> AuxState:
    return AuxState(
        action_tensor=jnp.zeros((ACTION_SEQ_LEN, ACTION_INPUT_DIM), dtype=jnp.float32),
        action_ptr=jnp.int32(0),
    )


@jax.jit
def aux_record_action(aux: AuxState, state: State, action: chex.Array) -> AuxState:
    action_entry = make_action_token(state, action, aux.action_ptr)
    write_idx = aux.action_ptr % ACTION_SEQ_LEN
    action_tensor = aux.action_tensor.at[write_idx].set(action_entry)
    return aux.replace(action_tensor=action_tensor, action_ptr=aux.action_ptr + 1)


def _canonize_suits(cards: chex.Array) -> chex.Array:
    valid = cards >= 0
    ranks = jnp.where(valid, cards % 13, 0).astype(jnp.int32)
    suits = jnp.where(valid, cards // 13, 0).astype(jnp.int32)

    def scan_fn(carry, x):
        suit, is_valid = x
        mapping, next_canonical = carry
        current_canonical = mapping[suit]
        is_new = is_valid & (current_canonical < 0)
        new_mapping = jnp.where(is_new, mapping.at[suit].set(next_canonical), mapping)
        new_next = jnp.where(is_new, next_canonical + 1, next_canonical)
        return (new_mapping, new_next), None

    init = (jnp.full(4, -1, dtype=jnp.int32), jnp.int32(0))
    (final_mapping, _), _ = jax.lax.scan(scan_fn, init, (suits, valid))
    canonical_suits = jnp.take(final_mapping, suits, axis=0)
    return jnp.where(valid, canonical_suits * 13 + ranks, -1)


@jax.jit
def encode_cards(state: State, player_id: chex.Array) -> chex.Array:
    is_p0 = player_id == 0
    hero_hole = jnp.where(is_p0, state.p0_hole, state.p1_hole)
    hero_active = jnp.where(is_p0, state.p0_active_mask, state.p1_active_mask)

    hero_rank = hero_hole % 13
    hero_suit = hero_hole // 13
    hero_sort_keys = jnp.where(hero_active, hero_rank * 4 + hero_suit, jnp.int32(999))
    hero_sorted_idx = jnp.argsort(hero_sort_keys)
    hero_sorted = hero_hole[hero_sorted_idx]
    hero_active_sorted = hero_active[hero_sorted_idx]

    hero_is_dealer = jnp.where(is_p0, state.is_dealer_p0, ~state.is_dealer_p0)
    hero_discard_idx = jnp.where(hero_is_dealer, 3, 2)
    villain_discard_idx = jnp.where(hero_is_dealer, 2, 3)

    flop_cards = jnp.array([state.board[0], state.board[1]], dtype=jnp.int32)
    flop_valid = flop_cards >= 0
    flop_rank = flop_cards % 13
    flop_suit = flop_cards // 13
    flop_sort_keys = jnp.where(flop_valid, flop_rank * 4 + flop_suit, jnp.int32(999))
    flop_sorted_idx = jnp.argsort(flop_sort_keys)
    flop_sorted = flop_cards[flop_sorted_idx]

    card_indices_raw = jnp.array([
        jnp.where(hero_active_sorted[0], hero_sorted[0], -1),
        jnp.where(hero_active_sorted[1], hero_sorted[1], -1),
        jnp.where(hero_active_sorted[2], hero_sorted[2], -1),
        flop_sorted[0], flop_sorted[1],
        state.board[hero_discard_idx], state.board[villain_discard_idx],
        state.board[4], state.board[5],
    ], dtype=jnp.int32)

    card_indices = _canonize_suits(card_indices_raw)
    type_ids_raw = jnp.array([0, 0, 0, 1, 1, 2, 3, 4, 5], dtype=jnp.int32)
    valid_mask = jnp.array([
        hero_active_sorted[0], hero_active_sorted[1], hero_active_sorted[2],
        card_indices[3] >= 0, card_indices[4] >= 0, card_indices[5] >= 0,
        card_indices[6] >= 0, card_indices[7] >= 0, card_indices[8] >= 0,
    ])
    type_ids = jnp.where(valid_mask, type_ids_raw, -1)
    return jnp.stack([card_indices, type_ids], axis=-1)


@jax.jit
def encode_actions(aux: AuxState, player_id: chex.Array) -> Tuple[chex.Array, chex.Array]:
    ptr = aux.action_ptr % ACTION_SEQ_LEN
    gather_idx = (jnp.arange(ACTION_SEQ_LEN) + ptr) % ACTION_SEQ_LEN
    action_tensor = aux.action_tensor[gather_idx]

    def swap_player_ids(tensor, pid):
        original_pid = tensor[..., 0]
        valid = (original_pid >= 0) & (original_pid <= 1)
        swapped = 1.0 - original_pid
        new_pid = jnp.where(pid == 0, original_pid, jnp.where(valid, swapped, original_pid))
        return tensor.at[..., 0].set(new_pid)

    action_tensor = swap_player_ids(action_tensor, player_id)
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
    card_tensor = encode_cards(state, player_id)
    action_tensor, action_mask = encode_actions(aux, player_id)
    legal_mask = ENV.get_legal_actions(state)
    history_tensor = opp_history if opp_history is not None else jnp.full(HISTORY_INPUT_DIM, 0.5, dtype=jnp.float32)
    return card_tensor, action_tensor, action_mask, legal_mask, history_tensor


# =============================================================================
# 3) Replay Buffer
# =============================================================================


@chex.dataclass(frozen=True)
class Unroll:
    card: chex.Array
    action_hist: chex.Array
    action_mask: chex.Array
    history: chex.Array
    legal: chex.Array
    action: chex.Array
    logp_mu: chex.Array
    reward: chex.Array
    done: chex.Array
    contrib_self: chex.Array
    contrib_opp: chex.Array
    valid: chex.Array
    bootstrap_card: chex.Array
    bootstrap_action_hist: chex.Array
    bootstrap_action_mask: chex.Array
    bootstrap_history: chex.Array
    bootstrap_done: chex.Array


@chex.dataclass(frozen=True)
class UnrollBuffer:
    card: chex.Array
    action_hist: chex.Array
    action_mask: chex.Array
    history: chex.Array
    legal: chex.Array
    action: chex.Array
    logp_mu: chex.Array
    reward: chex.Array
    done: chex.Array
    contrib_self: chex.Array
    contrib_opp: chex.Array
    valid: chex.Array
    bootstrap_card: chex.Array
    bootstrap_action_hist: chex.Array
    bootstrap_action_mask: chex.Array
    bootstrap_history: chex.Array
    bootstrap_done: chex.Array
    ptr: chex.Array
    size: chex.Array


def unroll_buffer_init(capacity: int, rollout_len: int, num_envs: int) -> UnrollBuffer:
    T, E = rollout_len, num_envs
    return UnrollBuffer(
        card=jnp.zeros((capacity, T, E, CARD_SEQ_LEN, CARD_INPUT_DIM), dtype=jnp.int32),
        action_hist=jnp.zeros((capacity, T, E, ACTION_SEQ_LEN, ACTION_INPUT_DIM), dtype=jnp.float32),
        action_mask=jnp.zeros((capacity, T, E, ACTION_SEQ_LEN), dtype=jnp.bool_),
        history=jnp.zeros((capacity, T, E, HISTORY_INPUT_DIM), dtype=jnp.float32),
        legal=jnp.zeros((capacity, T, E, NUM_ACTIONS), dtype=jnp.bool_),
        action=jnp.zeros((capacity, T, E), dtype=jnp.int32),
        logp_mu=jnp.zeros((capacity, T, E), dtype=jnp.float32),
        reward=jnp.zeros((capacity, T, E), dtype=jnp.float32),
        done=jnp.zeros((capacity, T, E), dtype=jnp.bool_),
        contrib_self=jnp.zeros((capacity, T, E), dtype=jnp.int32),
        contrib_opp=jnp.zeros((capacity, T, E), dtype=jnp.int32),
        valid=jnp.zeros((capacity, T, E), dtype=jnp.bool_),
        bootstrap_card=jnp.zeros((capacity, E, CARD_SEQ_LEN, CARD_INPUT_DIM), dtype=jnp.int32),
        bootstrap_action_hist=jnp.zeros((capacity, E, ACTION_SEQ_LEN, ACTION_INPUT_DIM), dtype=jnp.float32),
        bootstrap_action_mask=jnp.zeros((capacity, E, ACTION_SEQ_LEN), dtype=jnp.bool_),
        bootstrap_history=jnp.zeros((capacity, E, HISTORY_INPUT_DIM), dtype=jnp.float32),
        bootstrap_done=jnp.zeros((capacity, E), dtype=jnp.bool_),
        ptr=jnp.int32(0),
        size=jnp.int32(0),
    )


@jax.jit
def unroll_buffer_add(buf: UnrollBuffer, unroll: Unroll) -> UnrollBuffer:
    cap = buf.card.shape[0]
    idx = buf.ptr % cap
    return buf.replace(
        card=buf.card.at[idx].set(unroll.card),
        action_hist=buf.action_hist.at[idx].set(unroll.action_hist),
        action_mask=buf.action_mask.at[idx].set(unroll.action_mask),
        history=buf.history.at[idx].set(unroll.history),
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
        bootstrap_history=buf.bootstrap_history.at[idx].set(unroll.bootstrap_history),
        bootstrap_done=buf.bootstrap_done.at[idx].set(unroll.bootstrap_done),
        ptr=(buf.ptr + 1) % cap,
        size=jnp.minimum(buf.size + 1, cap),
    )


@partial(jax.jit, static_argnums=(2,))
def unroll_buffer_sample(buf: UnrollBuffer, key: chex.PRNGKey, batch_size: int) -> Tuple[Unroll, chex.PRNGKey]:
    key, sub = jax.random.split(key)
    cap = buf.card.shape[0]
    transitions_per_unroll = CFG.rollout_len * CFG.num_envs
    recent_window_unrolls = jnp.where(
        CFG.recent_window_transitions > 0,
        jnp.maximum(1, jnp.int32((CFG.recent_window_transitions + transitions_per_unroll - 1) // transitions_per_unroll)),
        cap,
    )
    available = jnp.minimum(buf.size, recent_window_unrolls)
    age = (buf.ptr - 1 - jnp.arange(cap, dtype=jnp.int32)) % cap
    eligible = jnp.where(available > 0, age < available, jnp.zeros_like(age, dtype=jnp.bool_))
    elig_w = eligible.astype(jnp.float32)
    elig_sum = jnp.sum(elig_w)
    p = jnp.where(elig_sum > 0, elig_w / elig_sum, jnp.ones((cap,), dtype=jnp.float32) / jnp.float32(cap))
    idx = jax.random.choice(sub, cap, shape=(batch_size,), replace=True, p=p)

    return Unroll(
        card=buf.card[idx], action_hist=buf.action_hist[idx], action_mask=buf.action_mask[idx],
        history=buf.history[idx], legal=buf.legal[idx], action=buf.action[idx], logp_mu=buf.logp_mu[idx],
        reward=buf.reward[idx], done=buf.done[idx], contrib_self=buf.contrib_self[idx],
        contrib_opp=buf.contrib_opp[idx], valid=buf.valid[idx], bootstrap_card=buf.bootstrap_card[idx],
        bootstrap_action_hist=buf.bootstrap_action_hist[idx], bootstrap_action_mask=buf.bootstrap_action_mask[idx],
        bootstrap_history=buf.bootstrap_history[idx], bootstrap_done=buf.bootstrap_done[idx],
    ), key


# =============================================================================
# 4) K-Best Self-Play Pool
# =============================================================================


@chex.dataclass(frozen=True)
class AgentPool:
    params: Any
    elo: chex.Array
    style_stats: chex.Array
    add_order: chex.Array
    size: chex.Array


def pool_init(dummy_params: Any) -> AgentPool:
    def _zeros_like(p):
        return jnp.zeros((CFG.pool_max,) + p.shape, dtype=p.dtype)
    params = jax.tree_util.tree_map(_zeros_like, dummy_params)
    return AgentPool(
        params=params, elo=jnp.zeros((CFG.pool_max,), dtype=jnp.float32),
        style_stats=jnp.zeros((CFG.pool_max, 3), dtype=jnp.float32),
        add_order=jnp.zeros((CFG.pool_max,), dtype=jnp.int32), size=jnp.int32(0)
    )


@jax.jit
def pool_add(pool: AgentPool, new_params: Any, initial_elo: chex.Array, new_style_stats: chex.Array) -> AgentPool:
    size = pool.size
    can_append = size < CFG.pool_max
    worst = jnp.argmin(jnp.where(jnp.arange(CFG.pool_max) < size, pool.elo, 1e9))
    write_idx = jnp.where(can_append, size, worst)
    params = jax.tree_util.tree_map(lambda pl, nl: pl.at[write_idx].set(nl), pool.params, new_params)
    elo = pool.elo.at[write_idx].set(initial_elo)
    style_stats = pool.style_stats.at[write_idx].set(new_style_stats)
    next_order = jnp.max(pool.add_order) + 1
    add_order = pool.add_order.at[write_idx].set(next_order)
    size2 = jnp.where(can_append, size + 1, size)
    return pool.replace(params=params, elo=elo, style_stats=style_stats, add_order=add_order, size=size2)


def elo_expected(ra: chex.Array, rb: chex.Array) -> chex.Array:
    return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))


@jax.jit
def elo_update(ra: chex.Array, rb: chex.Array, score_a: chex.Array, k: float) -> Tuple[chex.Array, chex.Array]:
    ea = elo_expected(ra, rb)
    return ra + k * (score_a - ea), rb + k * ((1.0 - score_a) - (1.0 - ea))


@jax.jit
def pool_select_opponent(pool: AgentPool, key: chex.PRNGKey) -> Tuple[Any, chex.PRNGKey]:
    key, k_category, k_top, k_diverse, k_recent = jax.random.split(key, 5)
    size = pool.size
    valid_mask = jnp.arange(CFG.pool_max) < size
    category = jax.random.choice(k_category, 3, p=jnp.array([0.60, 0.10, 0.30]))

    elo_with_invalid = jnp.where(valid_mask, pool.elo, -1e9)
    top_indices = jnp.argsort(elo_with_invalid)[::-1][:CFG.k_best]
    top_mask = jnp.zeros(CFG.pool_max, dtype=jnp.bool_).at[top_indices].set(True) & valid_mask
    top_probs = jnp.where(top_mask, 1.0, 0.0)
    top_probs = top_probs / jnp.maximum(1.0, jnp.sum(top_probs))
    idx_top = jax.random.choice(k_top, CFG.pool_max, p=top_probs)

    mean_style = jnp.sum(pool.style_stats * valid_mask[:, None], axis=0) / jnp.maximum(1.0, size.astype(jnp.float32))
    style_dist = jnp.sum((pool.style_stats - mean_style) ** 2, axis=1)
    style_dist = jnp.where(valid_mask, style_dist, -1e9)
    diverse_indices = jnp.argsort(style_dist)[::-1][:2]
    diverse_mask = jnp.zeros(CFG.pool_max, dtype=jnp.bool_).at[diverse_indices].set(True) & valid_mask
    diverse_probs = jnp.where(diverse_mask, 1.0, 0.0)
    diverse_probs = diverse_probs / jnp.maximum(1.0, jnp.sum(diverse_probs))
    idx_diverse = jax.random.choice(k_diverse, CFG.pool_max, p=diverse_probs)

    add_order_valid = jnp.where(valid_mask, pool.add_order, -1)
    recent_indices = jnp.argsort(add_order_valid)[::-1][:2]
    recent_mask = jnp.zeros(CFG.pool_max, dtype=jnp.bool_).at[recent_indices].set(True) & valid_mask
    recent_probs = jnp.where(recent_mask, 1.0, 0.0)
    recent_probs = recent_probs / jnp.maximum(1.0, jnp.sum(recent_probs))
    idx_recent = jax.random.choice(k_recent, CFG.pool_max, p=recent_probs)

    idx = jax.lax.switch(category, [lambda: idx_top, lambda: idx_diverse, lambda: idx_recent])
    return jax.tree_util.tree_map(lambda p: p[idx], pool.params), key


# =============================================================================
# 5) Rewards and Value Clipping
# =============================================================================


@jax.jit
def terminal_rewards(state: State) -> Tuple[chex.Array, chex.Array]:
    r0 = (state.p0_stack - STACK_SIZE).astype(jnp.float32)
    r1 = (state.p1_stack - STACK_SIZE).astype(jnp.float32)
    return jnp.where(state.done, r0, 0.0), jnp.where(state.done, r1, 0.0)


@jax.jit
def clip_value_target(v_target: chex.Array, contrib_self: chex.Array, contrib_opp: chex.Array) -> chex.Array:
    max_win = contrib_opp.astype(jnp.float32) / STACK_SIZE
    max_loss = -contrib_self.astype(jnp.float32) / STACK_SIZE
    return jnp.clip(v_target, max_loss, max_win)


# =============================================================================
# 6) Training State and Forward Pass
# =============================================================================


def make_lr_schedule(total_updates: int) -> optax.Schedule:
    warmup_fn = optax.linear_schedule(CFG.lr_init, CFG.lr_peak, CFG.warmup_updates)
    decay_fn = optax.cosine_decay_schedule(CFG.lr_peak, max(1, total_updates - CFG.warmup_updates), CFG.lr_final / CFG.lr_peak)
    return optax.join_schedules([warmup_fn, decay_fn], [CFG.warmup_updates])


def get_entropy_coef(update_step: int) -> chex.Array:
    progress = jnp.minimum(1.0, jnp.float32(update_step) / CFG.ent_decay_steps)
    return CFG.ent_coef_init + progress * (CFG.ent_coef_final - CFG.ent_coef_init)


def make_train_state(rng: chex.PRNGKey, total_updates: int) -> TrainState:
    model = TossHoldemModel()
    dummy_card = jnp.zeros((1, CARD_SEQ_LEN, CARD_INPUT_DIM), dtype=jnp.int32)
    dummy_action = jnp.zeros((1, ACTION_SEQ_LEN, ACTION_INPUT_DIM), dtype=jnp.float32)
    dummy_mask = jnp.ones((1, ACTION_SEQ_LEN), dtype=jnp.bool_)
    dummy_history = jnp.zeros((1, HISTORY_INPUT_DIM), dtype=jnp.float32)
    variables = model.init(rng, dummy_card, dummy_action, dummy_mask, dummy_history)
    tx = optax.chain(optax.clip_by_global_norm(CFG.max_grad_norm), optax.adam(make_lr_schedule(total_updates)))
    return TrainState.create(apply_fn=model.apply, params=variables["params"], tx=tx)


def save_checkpoint(path: str, ts: TrainState, step: int = 0):
    """Save checkpoint to path (works with Google Drive on Colab).
    
    Args:
        path: Directory path to save checkpoint (e.g., '/content/drive/MyDrive/checkpoints/step_1000')
        ts: TrainState to save
        step: Current training step (for metadata)
    """
    # Remove existing checkpoint directory if it exists (orbax doesn't overwrite)
    if os.path.exists(path):
        shutil.rmtree(path)
    
    os.makedirs(path, exist_ok=True)
    
    # Save only params (not optimizer state) for smaller checkpoints
    ckpt = {'params': ts.params, 'step': step}
    
    checkpointer = ocp.PyTreeCheckpointer()
    checkpointer.save(path, ckpt)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(path: str, ts: TrainState) -> TrainState:
    """Load checkpoint from path.
    
    Args:
        path: Directory path containing checkpoint
        ts: TrainState template (for structure)
        
    Returns:
        TrainState with loaded params
    """
    checkpointer = ocp.PyTreeCheckpointer()
    
    # Restore with structure hint from current params
    restored = checkpointer.restore(path, item={'params': ts.params, 'step': 0})
    
    # Handle both dict and direct params cases
    if isinstance(restored, dict) and 'params' in restored:
        return ts.replace(params=restored['params'])
    else:
        # Fallback: assume restored is the params directly
        return ts.replace(params=restored)


_FORWARD_MODEL = TossHoldemModel()


@jax.jit
def forward_apply(params: Any, card: chex.Array, action_hist: chex.Array, action_mask: chex.Array, history: chex.Array = None) -> Tuple[chex.Array, chex.Array]:
    return _FORWARD_MODEL.apply({"params": params}, card, action_hist, action_mask, history)


def masked_categorical(logits: chex.Array, legal_mask: chex.Array) -> distrax.Categorical:
    return distrax.Categorical(logits=jnp.where(legal_mask, logits, -1e9))


@jax.jit
def compute_vtrace(rewards, values, dones, last_value, log_rhos, rho_bar, c_bar, pg_rho_bar, gamma):
    T = rewards.shape[0]
    not_done = 1.0 - dones.astype(jnp.float32)
    rhos = jnp.exp(log_rhos)
    rho_v, rho_pg, cs = jnp.minimum(rho_bar, rhos), jnp.minimum(pg_rho_bar, rhos), jnp.minimum(c_bar, rhos)
    values_extended = jnp.concatenate([values, last_value[None]])
    td_targets = rewards + gamma * not_done * values_extended[1:]
    deltas = rho_v * (td_targets - values_extended[:-1])

    def vtrace_step(carry, inputs):
        delta, c, nd = inputs
        return delta + gamma * c * nd * carry, delta + gamma * c * nd * carry

    _, corrections_rev = jax.lax.scan(vtrace_step, 0.0, (jnp.flip(deltas), jnp.flip(cs), jnp.flip(not_done)))
    vs = values + jnp.flip(corrections_rev)
    vs_extended = jnp.concatenate([vs, last_value[None]])
    advantages = rho_pg * (rewards + gamma * not_done * vs_extended[1:] - values)
    return vs, advantages, rho_pg


@jax.jit
def compute_vtrace_on_unroll(params: Any, unroll: Unroll):
    T, E = unroll.card.shape[:2]
    flat_card = unroll.card.reshape(T * E, *unroll.card.shape[2:])
    flat_action_hist = unroll.action_hist.reshape(T * E, *unroll.action_hist.shape[2:])
    flat_action_mask = unroll.action_mask.reshape(T * E, *unroll.action_mask.shape[2:])
    flat_history = unroll.history.reshape(T * E, *unroll.history.shape[2:])
    flat_legal = unroll.legal.reshape(T * E, *unroll.legal.shape[2:])
    flat_action = unroll.action.reshape(T * E)

    all_card = jnp.concatenate([flat_card, unroll.bootstrap_card])
    all_action_hist = jnp.concatenate([flat_action_hist, unroll.bootstrap_action_hist])
    all_action_mask = jnp.concatenate([flat_action_mask, unroll.bootstrap_action_mask])
    all_history = jnp.concatenate([flat_history, unroll.bootstrap_history])

    all_logits, all_values = forward_apply(params, all_card, all_action_hist, all_action_mask, all_history)
    logits, values_flat, bootstrap_values = all_logits[:T * E], all_values[:T * E], all_values[T * E:]

    masked_logits = jnp.where(flat_legal, logits, -1e9)
    logp_pi_flat = jnp.take_along_axis(jax.nn.log_softmax(masked_logits, axis=-1), flat_action[:, None], axis=-1)[:, 0]
    logp_pi_te, values_te = logp_pi_flat.reshape(T, E), values_flat.reshape(T, E)
    log_rhos = jnp.where(unroll.valid, logp_pi_te - unroll.logp_mu, 0.0)
    last_value = jnp.where(unroll.bootstrap_done, 0.0, bootstrap_values)

    vs, advantages, pg_rhos = jax.vmap(
        lambda r, v, d, lv, lr: compute_vtrace(r, v, d, lv, lr, CFG.rho_bar, CFG.c_bar, CFG.pg_rho_bar, CFG.gamma),
        in_axes=(1, 1, 1, 0, 1), out_axes=(1, 1, 1)
    )(unroll.reward.astype(jnp.float32), values_te, unroll.done, last_value, log_rhos)

    return logp_pi_flat, values_flat, jax.lax.stop_gradient(vs), jax.lax.stop_gradient(advantages), jax.lax.stop_gradient(pg_rhos), masked_logits, flat_legal


@jax.jit
def vtrace_loss_on_unroll(params: Any, unroll: Unroll, ent_coef: chex.Array):
    logp_pi_flat, values_flat, vs, advantages, pg_rhos, masked_logits, flat_legal = compute_vtrace_on_unroll(params, unroll)
    vs_flat, advantages_flat, pg_rhos_flat = vs.reshape(-1), advantages.reshape(-1), pg_rhos.reshape(-1)
    valid_flat = unroll.valid.reshape(-1)
    valid_f = valid_flat.astype(jnp.float32)
    denom = jnp.maximum(1.0, jnp.sum(valid_f))

    policy_loss = -jnp.sum(pg_rhos_flat * logp_pi_flat * advantages_flat * valid_f) / denom
    target = clip_value_target(vs_flat, unroll.contrib_self.reshape(-1), unroll.contrib_opp.reshape(-1))
    value_loss = jnp.sum((target - values_flat) ** 2 * valid_f) / denom
    entropy = jnp.sum(masked_categorical(masked_logits, flat_legal).entropy() * valid_f) / denom

    illegal_mask = ~flat_legal
    illegal_penalty = jnp.sum(jax.nn.softplus(jnp.where(illegal_mask, masked_logits, -1e9)) * illegal_mask.astype(jnp.float32), axis=-1)
    illegal_loss = jnp.sum(illegal_penalty * valid_f) / denom

    total = policy_loss + CFG.vf_coef * value_loss - ent_coef * entropy + CFG.illegal_action_coef * illegal_loss

    logp_mu_flat = unroll.logp_mu.reshape(-1)
    rho_valid = jnp.where(valid_flat, jnp.exp(logp_pi_flat - logp_mu_flat), 0.0)

    return total, {
        "loss": total, "policy_loss": policy_loss, "value_loss": value_loss, "entropy": entropy,
        "illegal_loss": illegal_loss, "value_mean": jnp.sum(values_flat * valid_f) / denom,
        "adv_mean": jnp.sum(advantages_flat * valid_f) / denom, "ret_mean": jnp.sum(vs_flat * valid_f) / denom,
        "valid_frac": jnp.mean(valid_f), "rho_mean": jnp.sum(rho_valid * valid_f) / denom, "rho_max": jnp.max(rho_valid),
    }


def train_step_on_unroll(state: TrainState, unroll: Unroll, ent_coef: chex.Array):
    (loss, metrics), grads = jax.value_and_grad(vtrace_loss_on_unroll, has_aux=True)(state.params, unroll, ent_coef)
    new_state = state.apply_gradients(grads=grads)
    metrics["loss"], metrics["ent_coef"] = loss, ent_coef
    return new_state, metrics


# =============================================================================
# 7) Rollout Collection with Opponent Tracking
# =============================================================================


@chex.dataclass(frozen=True)
class VecEnvState:
    env_state: State
    aux: AuxState
    main_is_p0: chex.Array
    rng: chex.Array
    opp_stats: VecOpponentStats


def vec_reset(key: chex.PRNGKey, n: int) -> Tuple[VecEnvState, chex.PRNGKey]:
    key, k_seat, k_env = jax.random.split(key, 3)
    main_is_p0 = jax.random.bernoulli(k_seat, 0.5, (n,))
    ks = jax.random.split(k_env, n)
    s = jax.vmap(ENV.reset)(ks)
    aux = jax.vmap(lambda _: aux_reset())(jnp.arange(n))
    opp_stats = vec_opponent_stats_init(n)
    return VecEnvState(env_state=s, aux=aux, main_is_p0=main_is_p0, rng=ks, opp_stats=opp_stats), key


@partial(jax.jit, static_argnums=(5,))
def policy_act(params, state, aux, key, history=None, deterministic=False):
    pid = state.current_player.astype(jnp.int32)
    card, action_hist, action_mask, legal, _ = encode_obs(state, aux, pid, history)
    history_b = history[None, ...] if history is not None else None
    logits, value = forward_apply(params, card[None], action_hist[None], action_mask[None], history_b)
    masked_logits = jnp.where(legal, logits[0], -1e9)
    key, key_sample = jax.random.split(key)
    action = jax.lax.cond(
        deterministic, lambda: jnp.argmax(masked_logits).astype(jnp.int32),
        lambda: jnp.argmax(masked_logits + jax.random.gumbel(key_sample, (NUM_ACTIONS,))).astype(jnp.int32)
    )
    return action, jax.nn.log_softmax(masked_logits)[action], value[0], key


@jax.jit
def env_step_and_optional_reset(s, key, action):
    s_next = ENV.step(s, jax.random.fold_in(key, 0), action)
    s_reset = ENV.reset(jax.random.fold_in(key, 1))
    s_cont = jax.tree_util.tree_map(lambda a, b: jnp.where(s_next.done, a, b), s_reset, s_next)
    return s_next, s_cont, s_next.done, key


@jax.jit
def reset_aux_where_done(aux, done):
    done_expanded = done[:, None, None]
    return aux.replace(
        action_tensor=jnp.where(done_expanded, jnp.zeros_like(aux.action_tensor), aux.action_tensor),
        action_ptr=jnp.where(done, jnp.zeros_like(aux.action_ptr), aux.action_ptr),
    )


@jax.jit
def collect_rollout(main_params, pool, buf, vec, key):
    """Collect rollout with opponent stat tracking.
    
    Key design: Stats are reset at the START of each rollout (new opponent),
    then accumulated across hands within the rollout (same pool opponent).
    """
    opp_params, key = pool_select_opponent(pool, key)
    opp_params = jax.tree_util.tree_map(
        lambda o, m: jax.lax.cond(pool.size <= 1, lambda _: m, lambda _: o, operand=None),
        opp_params, main_params
    )
    
    # RESET opponent stats for new opponent at start of rollout
    fresh_opp_stats = vec_opponent_stats_init(CFG.num_envs)
    vec = vec.replace(opp_stats=fresh_opp_stats)

    env_ids = jnp.arange(CFG.num_envs, dtype=jnp.uint32)

    def step_fn(carry, t):
        vec = carry
        s = vec.env_state
        aux = vec.aux
        main_is_p0 = vec.main_is_p0
        opp_stats = vec.opp_stats
        base_keys = vec.rng
        
        learner_pid = jnp.where(main_is_p0, jnp.int32(0), jnp.int32(1))
        opp_pid = jnp.where(main_is_p0, jnp.int32(1), jnp.int32(0))
        pid_cur = s.current_player.astype(jnp.int32)
        is_learner_turn = pid_cur == learner_pid
        is_opp_turn = ~is_learner_turn
        
        # Opponent is dealer if: (opp is p0 and p0 is dealer) or (opp is p1 and p1 is dealer)
        opp_is_dealer = jnp.where(main_is_p0, ~s.is_dealer_p0, s.is_dealer_p0)

        # Get opponent history tensor BEFORE action
        opp_history_tensor = vec_opponent_stats_to_tensor(opp_stats)

        # Encode observations
        card_cur_b, action_hist_cur_b, action_mask_cur_b, legal_b, _ = jax.vmap(
            lambda s_i, aux_i, pid_i, hist_i: encode_obs(s_i, aux_i, pid_i, hist_i)
        )(s, aux, pid_cur, opp_history_tensor)

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

        # Forward passes
        logits_main_b, _ = forward_apply(main_params, card_cur_b, action_hist_cur_b, action_mask_cur_b, opp_history_tensor)
        has_opponent = pool.size > 1
        logits_opp_b = jax.lax.cond(
            has_opponent,
            lambda: forward_apply(opp_params, card_cur_b, action_hist_cur_b, action_mask_cur_b,
                                  jnp.full((CFG.num_envs, HISTORY_INPUT_DIM), 0.5))[0],
            lambda: logits_main_b
        )
        logits = jnp.where(is_learner_turn[:, None], logits_main_b, logits_opp_b)

        # Sample actions
        def _mk_key(k, eid, salt):
            return jax.random.fold_in(jax.random.fold_in(k, eid), t + jnp.uint32(salt))

        k_act = jax.vmap(lambda k, eid: _mk_key(k, eid, 0))(base_keys, env_ids)
        g = jax.vmap(lambda k: jax.random.gumbel(k, (NUM_ACTIONS,), dtype=jnp.float32))(k_act)
        masked_logits = jnp.where(legal_b, logits, -1e9)
        action = jnp.argmax(masked_logits + g, axis=-1).astype(jnp.int32)

        logp_all = jax.nn.log_softmax(masked_logits, axis=-1)
        logp = jnp.take_along_axis(logp_all, action[:, None], axis=-1)[:, 0]
        logp_mu = jnp.where(is_learner_turn, logp, 0.0)

        # Compute info needed for opponent stat update BEFORE stepping
        cur_bet = jnp.where(s.current_player == 0, s.p0_bet, s.p1_bet)
        opp_bet = jnp.where(s.current_player == 0, s.p1_bet, s.p0_bet)
        to_call = (opp_bet - cur_bet).astype(jnp.float32)
        pot_before = (s.pot + s.p0_bet + s.p1_bet).astype(jnp.float32)
        
        # Record current player's stack BEFORE stepping (for bet_amount calculation)
        cur_stack_before = jnp.where(s.current_player == 0, s.p0_stack, s.p1_stack).astype(jnp.float32)
        
        # Record action and step environment
        aux2 = jax.vmap(aux_record_action)(aux, s, action)
        k_env = jax.vmap(lambda k, eid: _mk_key(k, eid, 131071))(base_keys, env_ids)
        s_next, s_cont, done, _ = jax.vmap(env_step_and_optional_reset)(s, k_env, action)

        # Compute bet_amount from stack change (correct for any action)
        cur_stack_after = jnp.where(s.current_player == 0, s_next.p0_stack, s_next.p1_stack).astype(jnp.float32)
        bet_amount = cur_stack_before - cur_stack_after  # Chips committed this action
        
        # Update opponent stats with CORRECT bet amount
        opp_stats = vec_update_opp_stats(
            opp_stats,
            phase=s.phase,
            action=action,
            is_opp_turn=is_opp_turn,
            opp_is_dealer=opp_is_dealer,
            to_call=to_call,
            pot_before=pot_before,
            bet_amount=bet_amount,  # FIX: use actual chips committed by action-taker
        )

        # Rewards
        r0, r1 = terminal_rewards(s_next)
        rew = jnp.where(done, jnp.where(main_is_p0, r0, r1), 0.0) / STACK_SIZE

        # Contributions for value clipping
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
            "card": card_b, "action_hist": action_hist_b, "action_mask": action_mask_b,
            "history": history_b, "legal": legal_b, "action": action, "logp_mu": logp_mu,
            "reward": rew, "done": done, "contrib_self": contrib_self, "contrib_opp": contrib_opp,
            "valid": is_learner_turn,
        }

        # Update showdown stats on hand completion
        # went_to_showdown = hand ended without a fold (both players still in)
        last_action_was_fold = action == ACT_FOLD
        went_to_showdown = done & ~last_action_was_fold
        opp_won = jnp.where(main_is_p0, s_next.winner == 1, s_next.winner == 0)
        opp_stats = vec_update_showdown(opp_stats, done, went_to_showdown, opp_won, opp_is_dealer)

        # Reset per-round state on done (keeps accumulated stats!)
        opp_stats = vec_reset_round_state(opp_stats, done)

        # Reset aux and re-randomize seating
        aux3 = reset_aux_where_done(aux2, done)
        k_seat = jax.vmap(lambda k, eid: _mk_key(k, eid, 262143))(base_keys, env_ids)
        new_seat = jax.vmap(lambda k: jax.random.bernoulli(k, 0.5))(k_seat)
        main_is_p0_2 = jnp.where(done, new_seat, main_is_p0)

        vec2 = VecEnvState(env_state=s_cont, aux=aux3, main_is_p0=main_is_p0_2, rng=base_keys, opp_stats=opp_stats)
        return vec2, tmp

    vec2, traj = jax.lax.scan(step_fn, vec, jnp.arange(CFG.rollout_len, dtype=jnp.uint32))

    key, sub = jax.random.split(key)
    vec2 = vec2.replace(rng=jax.random.split(sub, CFG.num_envs))

    bootstrap_learner_pid = jnp.where(vec2.main_is_p0, jnp.int32(0), jnp.int32(1))
    bootstrap_opp_history = vec_opponent_stats_to_tensor(vec2.opp_stats)
    bootstrap_card, bootstrap_action_hist, bootstrap_action_mask, _, bootstrap_history = jax.vmap(
        lambda s_i, aux_i, pid_i, hist_i: encode_obs(s_i, aux_i, pid_i, hist_i)
    )(vec2.env_state, vec2.aux, bootstrap_learner_pid, bootstrap_opp_history)

    unroll = Unroll(
        card=traj["card"], action_hist=traj["action_hist"], action_mask=traj["action_mask"],
        history=traj["history"], legal=traj["legal"], action=traj["action"], logp_mu=traj["logp_mu"],
        reward=traj["reward"], done=traj["done"], contrib_self=traj["contrib_self"],
        contrib_opp=traj["contrib_opp"], valid=traj["valid"],
        bootstrap_card=bootstrap_card, bootstrap_action_hist=bootstrap_action_hist,
        bootstrap_action_mask=bootstrap_action_mask, bootstrap_history=bootstrap_history,
        bootstrap_done=traj["done"][-1, :],
    )

    return unroll_buffer_add(buf, unroll), vec2, key


# =============================================================================
# 8) Pool Evaluation
# =============================================================================


@jax.jit
def play_one_hand(params_a, params_b, key):
    key, k_reset = jax.random.split(key)
    s = ENV.reset(k_reset)
    aux = aux_reset()

    def cond_fn(carry):
        return ~carry[0].done

    def body_fn(carry):
        s, aux, key = carry
        pid = s.current_player.astype(jnp.int32)
        params = jax.lax.cond(pid == 0, lambda _: params_a, lambda _: params_b, operand=None)
        neutral_history = jnp.full(HISTORY_INPUT_DIM, 0.5, dtype=jnp.float32)
        a, _, _, key = policy_act(params, s, aux, key, neutral_history, deterministic=False)
        aux2 = aux_record_action(aux, s, a)
        key, k_step = jax.random.split(key)
        return ENV.step(s, k_step, a), aux2, key

    sT, _, key = jax.lax.while_loop(cond_fn, body_fn, (s, aux, key))
    return (sT.p0_stack - STACK_SIZE).astype(jnp.float32), key


@partial(jax.jit, static_argnums=(2,))
def evaluate_pair(params_a, params_b, n_hands, key):
    n_half = n_hands // 2
    keys = jax.random.split(key, n_hands + 1)
    results_p0 = jax.vmap(lambda k: play_one_hand(params_a, params_b, k)[0])(keys[:n_half])
    results_p1 = jax.vmap(lambda k: -play_one_hand(params_b, params_a, k)[0])(keys[n_half:-1])
    return (jnp.sum(results_p0) + jnp.sum(results_p1)) / n_hands, keys[-1]


def eval_and_update_pool(pool, ts, main_elo, key):
    current_params = ts.params

    def eval_vs_agent(carry, i):
        key, current_main_elo = carry
        is_valid = i < pool.size
        agent_params = jax.tree_util.tree_map(lambda p: p[i], pool.params)
        delta, key = evaluate_pair(current_params, agent_params, CFG.eval_hands, key)
        score = jax.nn.sigmoid(delta / CFG.elo_tau)
        new_main, new_agent = elo_update(current_main_elo, pool.elo[i], score, CFG.elo_k)
        # Chain the main ELO forward only if valid
        updated_main = jnp.where(is_valid, new_main, current_main_elo)
        return (key, updated_main), (jnp.where(is_valid, new_agent, pool.elo[i]), is_valid)

    (key, final_main_elo), (agent_elos, valids) = jax.lax.scan(
        eval_vs_agent, (key, main_elo), jnp.arange(CFG.pool_max)
    )
    pool = pool.replace(elo=jnp.where(valids, agent_elos, pool.elo))
    pool = pool_add(pool, current_params, final_main_elo, jnp.array([0.3, 0.3, 0.4]))
    return pool, final_main_elo, key


# =============================================================================
# 9) Training Loop
# =============================================================================


def make_train_block(block_size):
    @partial(jax.jit, static_argnums=())
    def train_block(ts, pool, buf, vec, key, base_step):
        def one_step(carry, step_offset):
            ts_inner, buf_inner, vec_inner, key_inner = carry
            upd = base_step + step_offset
            buf_inner, vec_inner, key_inner = collect_rollout(ts_inner.params, pool, buf_inner, vec_inner, key_inner)
            ent_coef = get_entropy_coef(upd)

            def _epoch_step(epoch_carry, _):
                ts_e, key_e = epoch_carry
                unrolls, key_e = unroll_buffer_sample(buf_inner, key_e, CFG.num_minibatches)

                def _unroll_step(ts_u, i):
                    return train_step_on_unroll(ts_u, jax.tree_util.tree_map(lambda x: x[i], unrolls), ent_coef)

                ts_e, metrics_stack = jax.lax.scan(_unroll_step, ts_e, jnp.arange(CFG.num_minibatches))
                return (ts_e, key_e), jax.tree_util.tree_map(jnp.mean, metrics_stack)

            (ts_inner, key_inner), metrics_epochs = jax.lax.scan(_epoch_step, (ts_inner, key_inner), None, length=CFG.update_epochs)
            return (ts_inner, buf_inner, vec_inner, key_inner), jax.tree_util.tree_map(jnp.mean, metrics_epochs)

        (ts_out, buf_out, vec_out, key_out), metrics_all = jax.lax.scan(one_step, (ts, buf, vec, key), jnp.arange(block_size, dtype=jnp.int32))
        return ts_out, buf_out, vec_out, key_out, jax.tree_util.tree_map(jnp.mean, metrics_all)

    return train_block


def main():
    print(f"JAX Devices: {jax.devices()}")
    print(f"Configuration: {CFG}")

    key = jax.random.PRNGKey(CFG.seed)
    key, k_init = jax.random.split(key)
    ts = make_train_state(k_init, CFG.total_updates)

    if CFG.resume_weights_from:
        print(f"Loading weights from {CFG.resume_weights_from}...")
        ts = load_checkpoint(CFG.resume_weights_from, ts)

    buf = unroll_buffer_init(4, CFG.rollout_len, CFG.num_envs)
    print(f"Model parameters: {sum(x.size for x in jax.tree_util.tree_leaves(ts.params)):,}")

    os.makedirs(CFG.checkpoint_dir, exist_ok=True)

    pool = pool_add(pool_init(ts.params), ts.params, jnp.float32(0.0), jnp.zeros(3))
    main_elo = jnp.float32(0.0)

    key, k_reset = jax.random.split(key)
    vec, key = vec_reset(k_reset, CFG.num_envs)

    do_collect = jax.jit(collect_rollout)
    do_train_block = make_train_block(CFG.block_size)

    print("Warming up...")
    for _ in range(max(1, CFG.warmup_steps // (CFG.num_envs * CFG.rollout_len))):
        buf, vec, key = do_collect(ts.params, pool, buf, vec, key)
    print(f"Buffer size: {buf.size}")

    print("Starting training...")
    start_time = time.time()

    for block_idx in range(CFG.total_updates // CFG.block_size):
        base_step = block_idx * CFG.block_size
        end_step = base_step + CFG.block_size

        ts, buf, vec, key, metrics = do_train_block(ts, pool, buf, vec, key, jnp.int32(base_step))

        if any((s + 1) % CFG.add_to_pool_every_updates == 0 for s in range(base_step, end_step)):
            pool, main_elo, key = eval_and_update_pool(pool, ts, main_elo, key)

        if end_step % CFG.log_every == 0:
            m = jax.device_get(metrics)
            elapsed = time.time() - start_time
            sps = (CFG.block_size * CFG.num_envs * CFG.rollout_len) / elapsed
            start_time = time.time()
            print(f"Upd {end_step}/{CFG.total_updates} | Pool: {int(pool.size)} | ELO: {float(main_elo):.1f} | "
                  f"SPS: {sps:.0f} | Loss: {m['loss']:.4f} | Pol: {m['policy_loss']:.4f} | "
                  f"Val: {m['value_loss']:.4f} | Ent: {m['entropy']:.4f} | rho: {m['rho_mean']:.2f}")

        if end_step % CFG.checkpoint_every == 0:
            save_checkpoint(f"{CFG.checkpoint_dir}/step_{end_step}", ts, step=end_step)

    print("Training complete!")


if __name__ == "__main__":
    main()