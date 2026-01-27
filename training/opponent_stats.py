"""Opponent statistics tracking for history encoding.

This module provides vectorized opponent stat tracking for the HistoryEncoder.
Stats are updated during rollout collection when the opponent takes actions,
enabling the model to learn opponent-adaptive play.

Key design:
- Stats persist across hands within a rollout (same pool opponent)
- Per-round state (preflop_aggressor, etc.) resets on hand completion
- Accumulated stats (frequencies, averages) persist to build opponent profile

Feature List (64 total):

PREFLOP (8):
- vpip_dealer, vpip_bb (voluntarily put money in pot)
- raise_freq_dealer, raise_freq_bb
- fold_freq_dealer, fold_freq_bb
- open_raise_size_avg_dealer, open_raise_size_avg_bb

DISCARD (2):
- discard_high_card_freq_dealer, discard_high_card_freq_bb

FLOP (14):
- fold/call/raise/bet frequencies by position
- avg_bet_size by position
- check_raise_freq, cbet_freq, donk_bet_freq

TURN (12):
- fold/call/raise/bet frequencies by position
- avg_bet_size by position
- double_barrel_freq by position

RIVER (14):
- fold/call/raise/bet frequencies by position
- avg_bet_size by position
- triple_barrel_freq, overbet_freq by position

AGGREGATED (14):
- showdown_winrate, allin_showdown_winrate
- avg_streets_seen, rounds_observed
- allin_freq, allin_call_freq, went_to_showdown by position
- reraise_freq by position
"""

import jax
import jax.numpy as jnp
import chex

# =============================================================================
# Output tensor indices (64 features)
# =============================================================================

HISTORY_DIM = 64

# PREFLOP (8 features, indices 0-7)
IDX_PREFLOP_VPIP_DEALER = 0
IDX_PREFLOP_VPIP_BB = 1
IDX_PREFLOP_RAISE_FREQ_DEALER = 2
IDX_PREFLOP_RAISE_FREQ_BB = 3
IDX_PREFLOP_FOLD_FREQ_DEALER = 4
IDX_PREFLOP_FOLD_FREQ_BB = 5
IDX_PREFLOP_OPEN_RAISE_SIZE_AVG_DEALER = 6
IDX_PREFLOP_OPEN_RAISE_SIZE_AVG_BB = 7

# DISCARD (2 features, indices 8-9)
IDX_DISCARD_HIGH_CARD_FREQ_DEALER = 8
IDX_DISCARD_HIGH_CARD_FREQ_BB = 9

# FLOP (14 features, indices 10-23)
IDX_FLOP_FOLD_FREQ_DEALER = 10
IDX_FLOP_FOLD_FREQ_BB = 11
IDX_FLOP_CALL_FREQ_DEALER = 12
IDX_FLOP_CALL_FREQ_BB = 13
IDX_FLOP_RAISE_FREQ_DEALER = 14
IDX_FLOP_RAISE_FREQ_BB = 15
IDX_FLOP_BET_FREQ_DEALER = 16
IDX_FLOP_BET_FREQ_BB = 17
IDX_FLOP_AVG_BET_SIZE_DEALER = 18
IDX_FLOP_AVG_BET_SIZE_BB = 19
IDX_FLOP_CHECK_RAISE_FREQ = 20
IDX_FLOP_CBET_FREQ_DEALER = 21
IDX_FLOP_CBET_FREQ_BB = 22
IDX_FLOP_DONK_BET_FREQ_BB = 23

# TURN (12 features, indices 24-35)
IDX_TURN_FOLD_FREQ_DEALER = 24
IDX_TURN_FOLD_FREQ_BB = 25
IDX_TURN_CALL_FREQ_DEALER = 26
IDX_TURN_CALL_FREQ_BB = 27
IDX_TURN_RAISE_FREQ_DEALER = 28
IDX_TURN_RAISE_FREQ_BB = 29
IDX_TURN_BET_FREQ_DEALER = 30
IDX_TURN_BET_FREQ_BB = 31
IDX_TURN_AVG_BET_SIZE_DEALER = 32
IDX_TURN_AVG_BET_SIZE_BB = 33
IDX_TURN_DOUBLE_BARREL_FREQ_DEALER = 34
IDX_TURN_DOUBLE_BARREL_FREQ_BB = 35

# RIVER (14 features, indices 36-49)
IDX_RIVER_FOLD_FREQ_DEALER = 36
IDX_RIVER_FOLD_FREQ_BB = 37
IDX_RIVER_CALL_FREQ_DEALER = 38
IDX_RIVER_CALL_FREQ_BB = 39
IDX_RIVER_RAISE_FREQ_DEALER = 40
IDX_RIVER_RAISE_FREQ_BB = 41
IDX_RIVER_BET_FREQ_DEALER = 42
IDX_RIVER_BET_FREQ_BB = 43
IDX_RIVER_AVG_BET_SIZE_DEALER = 44
IDX_RIVER_AVG_BET_SIZE_BB = 45
IDX_RIVER_TRIPLE_BARREL_FREQ_DEALER = 46
IDX_RIVER_TRIPLE_BARREL_FREQ_BB = 47
IDX_RIVER_OVERBET_FREQ_DEALER = 48
IDX_RIVER_OVERBET_FREQ_BB = 49

# AGGREGATED (14 features, indices 50-63)
IDX_AGG_SHOWDOWN_WINRATE = 50
IDX_AGG_ALLIN_SHOWDOWN_WINRATE = 51
IDX_AGG_AVG_STREETS_SEEN = 52
IDX_AGG_ROUNDS_OBSERVED = 53
IDX_AGG_ALLIN_FREQ_DEALER = 54
IDX_AGG_ALLIN_FREQ_BB = 55
IDX_AGG_ALLIN_CALL_FREQ_DEALER = 56
IDX_AGG_ALLIN_CALL_FREQ_BB = 57
IDX_AGG_WENT_TO_SHOWDOWN_DEALER = 58
IDX_AGG_WENT_TO_SHOWDOWN_BB = 59
IDX_AGG_RERAISE_FREQ_DEALER = 60
IDX_AGG_RERAISE_FREQ_BB = 61

# CAN ADD ONE
IDX_AGG_RESERVED_1 = 62
IDX_AGG_RESERVED_2 = 63

# =============================================================================
# Internal tracking indices
# =============================================================================

NUM_FREQ_STATS = 50

FREQ_IDX_VPIP_DEALER = 0
FREQ_IDX_VPIP_BB = 1
FREQ_IDX_PREFLOP_RAISE_DEALER = 2
FREQ_IDX_PREFLOP_RAISE_BB = 3
FREQ_IDX_PREFLOP_FOLD_DEALER = 4
FREQ_IDX_PREFLOP_FOLD_BB = 5
FREQ_IDX_DISCARD_HIGH_DEALER = 6
FREQ_IDX_DISCARD_HIGH_BB = 7
FREQ_IDX_FLOP_FOLD_DEALER = 8
FREQ_IDX_FLOP_FOLD_BB = 9
FREQ_IDX_FLOP_CALL_DEALER = 10
FREQ_IDX_FLOP_CALL_BB = 11
FREQ_IDX_FLOP_RAISE_DEALER = 12
FREQ_IDX_FLOP_RAISE_BB = 13
FREQ_IDX_FLOP_BET_DEALER = 14
FREQ_IDX_FLOP_BET_BB = 15
FREQ_IDX_FLOP_CHECK_RAISE = 16
FREQ_IDX_FLOP_CBET_DEALER = 17
FREQ_IDX_FLOP_CBET_BB = 18
FREQ_IDX_FLOP_DONK_BB = 19
FREQ_IDX_TURN_FOLD_DEALER = 20
FREQ_IDX_TURN_FOLD_BB = 21
FREQ_IDX_TURN_CALL_DEALER = 22
FREQ_IDX_TURN_CALL_BB = 23
FREQ_IDX_TURN_RAISE_DEALER = 24
FREQ_IDX_TURN_RAISE_BB = 25
FREQ_IDX_TURN_BET_DEALER = 26
FREQ_IDX_TURN_BET_BB = 27
FREQ_IDX_TURN_DOUBLE_BARREL_DEALER = 28
FREQ_IDX_TURN_DOUBLE_BARREL_BB = 29
FREQ_IDX_RIVER_FOLD_DEALER = 30
FREQ_IDX_RIVER_FOLD_BB = 31
FREQ_IDX_RIVER_CALL_DEALER = 32
FREQ_IDX_RIVER_CALL_BB = 33
FREQ_IDX_RIVER_RAISE_DEALER = 34
FREQ_IDX_RIVER_RAISE_BB = 35
FREQ_IDX_RIVER_BET_DEALER = 36
FREQ_IDX_RIVER_BET_BB = 37
FREQ_IDX_RIVER_TRIPLE_BARREL_DEALER = 38
FREQ_IDX_RIVER_TRIPLE_BARREL_BB = 39
FREQ_IDX_RIVER_OVERBET_DEALER = 40
FREQ_IDX_RIVER_OVERBET_BB = 41
FREQ_IDX_SHOWDOWN_WIN = 42
FREQ_IDX_ALLIN_SHOWDOWN_WIN = 43
FREQ_IDX_ALLIN_DEALER = 44
FREQ_IDX_ALLIN_BB = 45
FREQ_IDX_ALLIN_CALL_DEALER = 46
FREQ_IDX_ALLIN_CALL_BB = 47
FREQ_IDX_WENT_SHOWDOWN_DEALER = 48
FREQ_IDX_WENT_SHOWDOWN_BB = 49

NUM_AVG_STATS = 10

AVG_IDX_OPEN_RAISE_SIZE_DEALER = 0
AVG_IDX_OPEN_RAISE_SIZE_BB = 1
AVG_IDX_FLOP_BET_SIZE_DEALER = 2
AVG_IDX_FLOP_BET_SIZE_BB = 3
AVG_IDX_TURN_BET_SIZE_DEALER = 4
AVG_IDX_TURN_BET_SIZE_BB = 5
AVG_IDX_RIVER_BET_SIZE_DEALER = 6
AVG_IDX_RIVER_BET_SIZE_BB = 7
AVG_IDX_STREETS_SEEN = 8
AVG_IDX_RESERVED = 9

# Action and Phase constants (must match game.py)
ACT_FOLD = 0
ACT_CHECK = 1
ACT_CALL = 2
ACT_RAISE_MIN = 3
ACT_ALLIN = 9
ACT_DISCARD_0 = 10
ACT_DISCARD_2 = 12

PHASE_PREFLOP = 0
PHASE_FLOP = 3
PHASE_TURN = 4
PHASE_RIVER = 5


# =============================================================================
# Vectorized Opponent Stats
# =============================================================================


@chex.dataclass(frozen=True)
class VecOpponentStats:
    """Vectorized opponent stats for E parallel environments."""
    freq_stats: chex.Array  # (E, NUM_FREQ_STATS, 2)
    avg_stats: chex.Array   # (E, NUM_AVG_STATS, 2)
    
    rounds_observed: chex.Array  # (E,)
    preflop_aggressor: chex.Array  # (E,) int32: 0=dealer, 1=bb, -1=none
    opp_bet_flop: chex.Array  # (E,) bool
    opp_bet_turn: chex.Array  # (E,) bool
    opp_checked_this_street: chex.Array  # (E,) bool
    pot_at_street_start: chex.Array  # (E,) float32
    current_street: chex.Array  # (E,) int32
    went_allin_this_hand: chex.Array  # (E,) bool


def vec_opponent_stats_init(num_envs: int) -> VecOpponentStats:
    """Initialize fresh vectorized opponent stats."""
    return VecOpponentStats(
        freq_stats=jnp.zeros((num_envs, NUM_FREQ_STATS, 2), dtype=jnp.float32),
        avg_stats=jnp.zeros((num_envs, NUM_AVG_STATS, 2), dtype=jnp.float32),
        rounds_observed=jnp.zeros(num_envs, dtype=jnp.float32),
        preflop_aggressor=jnp.full(num_envs, -1, dtype=jnp.int32),
        opp_bet_flop=jnp.zeros(num_envs, dtype=jnp.bool_),
        opp_bet_turn=jnp.zeros(num_envs, dtype=jnp.bool_),
        opp_checked_this_street=jnp.zeros(num_envs, dtype=jnp.bool_),
        pot_at_street_start=jnp.zeros(num_envs, dtype=jnp.float32),
        current_street=jnp.zeros(num_envs, dtype=jnp.int32),
        went_allin_this_hand=jnp.zeros(num_envs, dtype=jnp.bool_),
    )


@jax.jit
def vec_reset_round_state(opp_stats: VecOpponentStats, done: chex.Array) -> VecOpponentStats:
    """Reset per-round state on hand completion. Keeps accumulated stats!"""
    return opp_stats.replace(
        preflop_aggressor=jnp.where(done, jnp.int32(-1), opp_stats.preflop_aggressor),
        opp_bet_flop=jnp.where(done, False, opp_stats.opp_bet_flop),
        opp_bet_turn=jnp.where(done, False, opp_stats.opp_bet_turn),
        opp_checked_this_street=jnp.where(done, False, opp_stats.opp_checked_this_street),
        pot_at_street_start=jnp.where(done, 0.0, opp_stats.pot_at_street_start),
        current_street=jnp.where(done, jnp.int32(0), opp_stats.current_street),
        went_allin_this_hand=jnp.where(done, False, opp_stats.went_allin_this_hand),
        rounds_observed=jnp.where(done, opp_stats.rounds_observed + 1, opp_stats.rounds_observed),
    )


@jax.jit
def vec_update_opp_stats(
    opp_stats: VecOpponentStats,
    phase: chex.Array,
    action: chex.Array,
    is_opp_turn: chex.Array,
    opp_is_dealer: chex.Array,
    to_call: chex.Array,
    pot_before: chex.Array,
    bet_amount: chex.Array,
) -> VecOpponentStats:
    """Update opponent statistics based on their action."""
    update = is_opp_turn
    
    is_fold = action == ACT_FOLD
    is_check = action == ACT_CHECK
    is_call = action == ACT_CALL
    is_raise = (action >= ACT_RAISE_MIN) & (action <= ACT_ALLIN)
    is_allin = action == ACT_ALLIN
    is_discard = (action >= ACT_DISCARD_0) & (action <= ACT_DISCARD_2)
    
    is_vpip = is_call | is_raise
    facing_bet = to_call > 0
    is_bet = is_raise & ~facing_bet
    is_raise_vs_bet = is_raise & facing_bet
    
    is_preflop = phase == PHASE_PREFLOP
    is_flop = phase == PHASE_FLOP
    is_turn = phase == PHASE_TURN
    is_river = phase == PHASE_RIVER
    
    bet_frac = jnp.where(pot_before > 0, bet_amount / pot_before, 0.0)
    
    freq = opp_stats.freq_stats
    avg = opp_stats.avg_stats
    
    def update_freq(f, idx, success, should_update):
        success_add = jnp.where(should_update, success.astype(jnp.float32), 0.0)
        trial_add = jnp.where(should_update, 1.0, 0.0)
        f = f.at[:, idx, 0].add(success_add)
        f = f.at[:, idx, 1].add(trial_add)
        return f
    
    def update_avg(a, idx, value, should_update):
        sum_add = jnp.where(should_update, value, 0.0)
        count_add = jnp.where(should_update, 1.0, 0.0)
        a = a.at[:, idx, 0].add(sum_add)
        a = a.at[:, idx, 1].add(count_add)
        return a
    
    # PREFLOP
    preflop_action = update & is_preflop & ~is_discard
    freq = update_freq(freq, FREQ_IDX_VPIP_DEALER, is_vpip, preflop_action & opp_is_dealer)
    freq = update_freq(freq, FREQ_IDX_VPIP_BB, is_vpip, preflop_action & ~opp_is_dealer)
    freq = update_freq(freq, FREQ_IDX_PREFLOP_RAISE_DEALER, is_raise, preflop_action & opp_is_dealer)
    freq = update_freq(freq, FREQ_IDX_PREFLOP_RAISE_BB, is_raise, preflop_action & ~opp_is_dealer)
    freq = update_freq(freq, FREQ_IDX_PREFLOP_FOLD_DEALER, is_fold, preflop_action & opp_is_dealer)
    freq = update_freq(freq, FREQ_IDX_PREFLOP_FOLD_BB, is_fold, preflop_action & ~opp_is_dealer)
    avg = update_avg(avg, AVG_IDX_OPEN_RAISE_SIZE_DEALER, bet_frac, preflop_action & is_raise & opp_is_dealer)
    avg = update_avg(avg, AVG_IDX_OPEN_RAISE_SIZE_BB, bet_frac, preflop_action & is_raise & ~opp_is_dealer)
    
    # FLOP
    flop_action = update & is_flop
    freq = update_freq(freq, FREQ_IDX_FLOP_FOLD_DEALER, is_fold, flop_action & opp_is_dealer)
    freq = update_freq(freq, FREQ_IDX_FLOP_FOLD_BB, is_fold, flop_action & ~opp_is_dealer)
    freq = update_freq(freq, FREQ_IDX_FLOP_CALL_DEALER, is_call, flop_action & opp_is_dealer)
    freq = update_freq(freq, FREQ_IDX_FLOP_CALL_BB, is_call, flop_action & ~opp_is_dealer)
    freq = update_freq(freq, FREQ_IDX_FLOP_RAISE_DEALER, is_raise_vs_bet, flop_action & opp_is_dealer)
    freq = update_freq(freq, FREQ_IDX_FLOP_RAISE_BB, is_raise_vs_bet, flop_action & ~opp_is_dealer)
    freq = update_freq(freq, FREQ_IDX_FLOP_BET_DEALER, is_bet, flop_action & opp_is_dealer)
    freq = update_freq(freq, FREQ_IDX_FLOP_BET_BB, is_bet, flop_action & ~opp_is_dealer)
    avg = update_avg(avg, AVG_IDX_FLOP_BET_SIZE_DEALER, bet_frac, flop_action & (is_bet | is_raise) & opp_is_dealer)
    avg = update_avg(avg, AVG_IDX_FLOP_BET_SIZE_BB, bet_frac, flop_action & (is_bet | is_raise) & ~opp_is_dealer)
    
    is_check_raise = is_raise_vs_bet & opp_stats.opp_checked_this_street
    freq = update_freq(freq, FREQ_IDX_FLOP_CHECK_RAISE, is_check_raise, flop_action)
    
    opp_was_aggressor = jnp.where(opp_is_dealer, opp_stats.preflop_aggressor == 0, opp_stats.preflop_aggressor == 1)
    is_cbet = is_bet & opp_was_aggressor
    freq = update_freq(freq, FREQ_IDX_FLOP_CBET_DEALER, is_cbet, flop_action & opp_is_dealer)
    freq = update_freq(freq, FREQ_IDX_FLOP_CBET_BB, is_cbet, flop_action & ~opp_is_dealer)
    
    is_donk = is_bet & ~opp_is_dealer & (opp_stats.preflop_aggressor == 0)
    freq = update_freq(freq, FREQ_IDX_FLOP_DONK_BB, is_donk, flop_action)
    
    # TURN
    turn_action = update & is_turn
    freq = update_freq(freq, FREQ_IDX_TURN_FOLD_DEALER, is_fold, turn_action & opp_is_dealer)
    freq = update_freq(freq, FREQ_IDX_TURN_FOLD_BB, is_fold, turn_action & ~opp_is_dealer)
    freq = update_freq(freq, FREQ_IDX_TURN_CALL_DEALER, is_call, turn_action & opp_is_dealer)
    freq = update_freq(freq, FREQ_IDX_TURN_CALL_BB, is_call, turn_action & ~opp_is_dealer)
    freq = update_freq(freq, FREQ_IDX_TURN_RAISE_DEALER, is_raise_vs_bet, turn_action & opp_is_dealer)
    freq = update_freq(freq, FREQ_IDX_TURN_RAISE_BB, is_raise_vs_bet, turn_action & ~opp_is_dealer)
    freq = update_freq(freq, FREQ_IDX_TURN_BET_DEALER, is_bet, turn_action & opp_is_dealer)
    freq = update_freq(freq, FREQ_IDX_TURN_BET_BB, is_bet, turn_action & ~opp_is_dealer)
    avg = update_avg(avg, AVG_IDX_TURN_BET_SIZE_DEALER, bet_frac, turn_action & (is_bet | is_raise) & opp_is_dealer)
    avg = update_avg(avg, AVG_IDX_TURN_BET_SIZE_BB, bet_frac, turn_action & (is_bet | is_raise) & ~opp_is_dealer)
    
    is_double_barrel = is_bet & opp_stats.opp_bet_flop
    freq = update_freq(freq, FREQ_IDX_TURN_DOUBLE_BARREL_DEALER, is_double_barrel, turn_action & opp_is_dealer)
    freq = update_freq(freq, FREQ_IDX_TURN_DOUBLE_BARREL_BB, is_double_barrel, turn_action & ~opp_is_dealer)
    
    # RIVER
    river_action = update & is_river
    freq = update_freq(freq, FREQ_IDX_RIVER_FOLD_DEALER, is_fold, river_action & opp_is_dealer)
    freq = update_freq(freq, FREQ_IDX_RIVER_FOLD_BB, is_fold, river_action & ~opp_is_dealer)
    freq = update_freq(freq, FREQ_IDX_RIVER_CALL_DEALER, is_call, river_action & opp_is_dealer)
    freq = update_freq(freq, FREQ_IDX_RIVER_CALL_BB, is_call, river_action & ~opp_is_dealer)
    freq = update_freq(freq, FREQ_IDX_RIVER_RAISE_DEALER, is_raise_vs_bet, river_action & opp_is_dealer)
    freq = update_freq(freq, FREQ_IDX_RIVER_RAISE_BB, is_raise_vs_bet, river_action & ~opp_is_dealer)
    freq = update_freq(freq, FREQ_IDX_RIVER_BET_DEALER, is_bet, river_action & opp_is_dealer)
    freq = update_freq(freq, FREQ_IDX_RIVER_BET_BB, is_bet, river_action & ~opp_is_dealer)
    avg = update_avg(avg, AVG_IDX_RIVER_BET_SIZE_DEALER, bet_frac, river_action & (is_bet | is_raise) & opp_is_dealer)
    avg = update_avg(avg, AVG_IDX_RIVER_BET_SIZE_BB, bet_frac, river_action & (is_bet | is_raise) & ~opp_is_dealer)
    
    is_triple_barrel = is_bet & opp_stats.opp_bet_flop & opp_stats.opp_bet_turn
    freq = update_freq(freq, FREQ_IDX_RIVER_TRIPLE_BARREL_DEALER, is_triple_barrel, river_action & opp_is_dealer)
    freq = update_freq(freq, FREQ_IDX_RIVER_TRIPLE_BARREL_BB, is_triple_barrel, river_action & ~opp_is_dealer)
    
    is_overbet = (is_bet | is_raise) & (bet_amount > pot_before)
    freq = update_freq(freq, FREQ_IDX_RIVER_OVERBET_DEALER, is_overbet, river_action & opp_is_dealer)
    freq = update_freq(freq, FREQ_IDX_RIVER_OVERBET_BB, is_overbet, river_action & ~opp_is_dealer)
    
    # ALL-IN
    freq = update_freq(freq, FREQ_IDX_ALLIN_DEALER, is_allin, update & opp_is_dealer)
    freq = update_freq(freq, FREQ_IDX_ALLIN_BB, is_allin, update & ~opp_is_dealer)
    is_big_call = is_call & (to_call > 100)
    freq = update_freq(freq, FREQ_IDX_ALLIN_CALL_DEALER, is_big_call, update & opp_is_dealer)
    freq = update_freq(freq, FREQ_IDX_ALLIN_CALL_BB, is_big_call, update & ~opp_is_dealer)
    
    # Update per-round state
    new_preflop_agg = jnp.where(update & is_preflop & is_raise, jnp.where(opp_is_dealer, 0, 1), opp_stats.preflop_aggressor)
    new_opp_bet_flop = jnp.where(update & is_flop & is_bet, True, opp_stats.opp_bet_flop)
    new_opp_bet_turn = jnp.where(update & is_turn & is_bet, True, opp_stats.opp_bet_turn)
    new_opp_checked = jnp.where(update & is_check, True, opp_stats.opp_checked_this_street)
    street_changed = phase != opp_stats.current_street
    new_opp_checked = jnp.where(street_changed, False, new_opp_checked)
    new_went_allin = jnp.where(update & is_allin, True, opp_stats.went_allin_this_hand)
    
    return opp_stats.replace(
        freq_stats=freq,
        avg_stats=avg,
        preflop_aggressor=new_preflop_agg,
        opp_bet_flop=new_opp_bet_flop,
        opp_bet_turn=new_opp_bet_turn,
        opp_checked_this_street=new_opp_checked,
        current_street=phase,
        went_allin_this_hand=new_went_allin,
    )


@jax.jit
def vec_update_showdown(
    opp_stats: VecOpponentStats,
    done: chex.Array,
    went_to_showdown: chex.Array,
    opp_won: chex.Array,
    opp_is_dealer: chex.Array,
) -> VecOpponentStats:
    """Update showdown statistics on hand completion."""
    update = done
    freq = opp_stats.freq_stats
    avg = opp_stats.avg_stats
    
    def update_freq(f, idx, success, should_update):
        success_add = jnp.where(should_update, success.astype(jnp.float32), 0.0)
        trial_add = jnp.where(should_update, 1.0, 0.0)
        f = f.at[:, idx, 0].add(success_add)
        f = f.at[:, idx, 1].add(trial_add)
        return f
    
    def update_avg(a, idx, value, should_update):
        sum_add = jnp.where(should_update, value, 0.0)
        count_add = jnp.where(should_update, 1.0, 0.0)
        a = a.at[:, idx, 0].add(sum_add)
        a = a.at[:, idx, 1].add(count_add)
        return a
    
    freq = update_freq(freq, FREQ_IDX_SHOWDOWN_WIN, opp_won, update & went_to_showdown)
    freq = update_freq(freq, FREQ_IDX_ALLIN_SHOWDOWN_WIN, opp_won, update & went_to_showdown & opp_stats.went_allin_this_hand)
    freq = update_freq(freq, FREQ_IDX_WENT_SHOWDOWN_DEALER, went_to_showdown, update & opp_is_dealer)
    freq = update_freq(freq, FREQ_IDX_WENT_SHOWDOWN_BB, went_to_showdown, update & ~opp_is_dealer)
    
    streets = opp_stats.current_street.astype(jnp.float32)
    streets_seen = jnp.clip(streets, 1.0, 4.0)
    avg = update_avg(avg, AVG_IDX_STREETS_SEEN, streets_seen, update)
    
    return opp_stats.replace(freq_stats=freq, avg_stats=avg)


# =============================================================================
# Conversion to tensor
# =============================================================================


def _freq(successes: chex.Array, trials: chex.Array) -> chex.Array:
    return jnp.where(trials > 0, successes / trials, 0.5)


def _avg(total: chex.Array, count: chex.Array, default: float = 0.0) -> chex.Array:
    return jnp.where(count > 0, total / count, default)


def _log_rounds(count: chex.Array) -> chex.Array:
    return jnp.log1p(count) / 5.0


def _single_stats_to_tensor(freq_stats, avg_stats, rounds_observed):
    tensor = jnp.zeros(HISTORY_DIM, dtype=jnp.float32)
    
    def get_freq(idx):
        return _freq(freq_stats[idx, 0], freq_stats[idx, 1])
    
    def get_avg(idx, default=0.5):
        return _avg(avg_stats[idx, 0], avg_stats[idx, 1], default)
    
    tensor = tensor.at[IDX_PREFLOP_VPIP_DEALER].set(get_freq(FREQ_IDX_VPIP_DEALER))
    tensor = tensor.at[IDX_PREFLOP_VPIP_BB].set(get_freq(FREQ_IDX_VPIP_BB))
    tensor = tensor.at[IDX_PREFLOP_RAISE_FREQ_DEALER].set(get_freq(FREQ_IDX_PREFLOP_RAISE_DEALER))
    tensor = tensor.at[IDX_PREFLOP_RAISE_FREQ_BB].set(get_freq(FREQ_IDX_PREFLOP_RAISE_BB))
    tensor = tensor.at[IDX_PREFLOP_FOLD_FREQ_DEALER].set(get_freq(FREQ_IDX_PREFLOP_FOLD_DEALER))
    tensor = tensor.at[IDX_PREFLOP_FOLD_FREQ_BB].set(get_freq(FREQ_IDX_PREFLOP_FOLD_BB))
    tensor = tensor.at[IDX_PREFLOP_OPEN_RAISE_SIZE_AVG_DEALER].set(get_avg(AVG_IDX_OPEN_RAISE_SIZE_DEALER, 2.5))
    tensor = tensor.at[IDX_PREFLOP_OPEN_RAISE_SIZE_AVG_BB].set(get_avg(AVG_IDX_OPEN_RAISE_SIZE_BB, 3.0))
    tensor = tensor.at[IDX_DISCARD_HIGH_CARD_FREQ_DEALER].set(0.5)
    tensor = tensor.at[IDX_DISCARD_HIGH_CARD_FREQ_BB].set(0.5)
    tensor = tensor.at[IDX_FLOP_FOLD_FREQ_DEALER].set(get_freq(FREQ_IDX_FLOP_FOLD_DEALER))
    tensor = tensor.at[IDX_FLOP_FOLD_FREQ_BB].set(get_freq(FREQ_IDX_FLOP_FOLD_BB))
    tensor = tensor.at[IDX_FLOP_CALL_FREQ_DEALER].set(get_freq(FREQ_IDX_FLOP_CALL_DEALER))
    tensor = tensor.at[IDX_FLOP_CALL_FREQ_BB].set(get_freq(FREQ_IDX_FLOP_CALL_BB))
    tensor = tensor.at[IDX_FLOP_RAISE_FREQ_DEALER].set(get_freq(FREQ_IDX_FLOP_RAISE_DEALER))
    tensor = tensor.at[IDX_FLOP_RAISE_FREQ_BB].set(get_freq(FREQ_IDX_FLOP_RAISE_BB))
    tensor = tensor.at[IDX_FLOP_BET_FREQ_DEALER].set(get_freq(FREQ_IDX_FLOP_BET_DEALER))
    tensor = tensor.at[IDX_FLOP_BET_FREQ_BB].set(get_freq(FREQ_IDX_FLOP_BET_BB))
    tensor = tensor.at[IDX_FLOP_AVG_BET_SIZE_DEALER].set(get_avg(AVG_IDX_FLOP_BET_SIZE_DEALER))
    tensor = tensor.at[IDX_FLOP_AVG_BET_SIZE_BB].set(get_avg(AVG_IDX_FLOP_BET_SIZE_BB))
    tensor = tensor.at[IDX_FLOP_CHECK_RAISE_FREQ].set(get_freq(FREQ_IDX_FLOP_CHECK_RAISE))
    tensor = tensor.at[IDX_FLOP_CBET_FREQ_DEALER].set(get_freq(FREQ_IDX_FLOP_CBET_DEALER))
    tensor = tensor.at[IDX_FLOP_CBET_FREQ_BB].set(get_freq(FREQ_IDX_FLOP_CBET_BB))
    tensor = tensor.at[IDX_FLOP_DONK_BET_FREQ_BB].set(get_freq(FREQ_IDX_FLOP_DONK_BB))
    tensor = tensor.at[IDX_TURN_FOLD_FREQ_DEALER].set(get_freq(FREQ_IDX_TURN_FOLD_DEALER))
    tensor = tensor.at[IDX_TURN_FOLD_FREQ_BB].set(get_freq(FREQ_IDX_TURN_FOLD_BB))
    tensor = tensor.at[IDX_TURN_CALL_FREQ_DEALER].set(get_freq(FREQ_IDX_TURN_CALL_DEALER))
    tensor = tensor.at[IDX_TURN_CALL_FREQ_BB].set(get_freq(FREQ_IDX_TURN_CALL_BB))
    tensor = tensor.at[IDX_TURN_RAISE_FREQ_DEALER].set(get_freq(FREQ_IDX_TURN_RAISE_DEALER))
    tensor = tensor.at[IDX_TURN_RAISE_FREQ_BB].set(get_freq(FREQ_IDX_TURN_RAISE_BB))
    tensor = tensor.at[IDX_TURN_BET_FREQ_DEALER].set(get_freq(FREQ_IDX_TURN_BET_DEALER))
    tensor = tensor.at[IDX_TURN_BET_FREQ_BB].set(get_freq(FREQ_IDX_TURN_BET_BB))
    tensor = tensor.at[IDX_TURN_AVG_BET_SIZE_DEALER].set(get_avg(AVG_IDX_TURN_BET_SIZE_DEALER))
    tensor = tensor.at[IDX_TURN_AVG_BET_SIZE_BB].set(get_avg(AVG_IDX_TURN_BET_SIZE_BB))
    tensor = tensor.at[IDX_TURN_DOUBLE_BARREL_FREQ_DEALER].set(get_freq(FREQ_IDX_TURN_DOUBLE_BARREL_DEALER))
    tensor = tensor.at[IDX_TURN_DOUBLE_BARREL_FREQ_BB].set(get_freq(FREQ_IDX_TURN_DOUBLE_BARREL_BB))
    tensor = tensor.at[IDX_RIVER_FOLD_FREQ_DEALER].set(get_freq(FREQ_IDX_RIVER_FOLD_DEALER))
    tensor = tensor.at[IDX_RIVER_FOLD_FREQ_BB].set(get_freq(FREQ_IDX_RIVER_FOLD_BB))
    tensor = tensor.at[IDX_RIVER_CALL_FREQ_DEALER].set(get_freq(FREQ_IDX_RIVER_CALL_DEALER))
    tensor = tensor.at[IDX_RIVER_CALL_FREQ_BB].set(get_freq(FREQ_IDX_RIVER_CALL_BB))
    tensor = tensor.at[IDX_RIVER_RAISE_FREQ_DEALER].set(get_freq(FREQ_IDX_RIVER_RAISE_DEALER))
    tensor = tensor.at[IDX_RIVER_RAISE_FREQ_BB].set(get_freq(FREQ_IDX_RIVER_RAISE_BB))
    tensor = tensor.at[IDX_RIVER_BET_FREQ_DEALER].set(get_freq(FREQ_IDX_RIVER_BET_DEALER))
    tensor = tensor.at[IDX_RIVER_BET_FREQ_BB].set(get_freq(FREQ_IDX_RIVER_BET_BB))
    tensor = tensor.at[IDX_RIVER_AVG_BET_SIZE_DEALER].set(get_avg(AVG_IDX_RIVER_BET_SIZE_DEALER))
    tensor = tensor.at[IDX_RIVER_AVG_BET_SIZE_BB].set(get_avg(AVG_IDX_RIVER_BET_SIZE_BB))
    tensor = tensor.at[IDX_RIVER_TRIPLE_BARREL_FREQ_DEALER].set(get_freq(FREQ_IDX_RIVER_TRIPLE_BARREL_DEALER))
    tensor = tensor.at[IDX_RIVER_TRIPLE_BARREL_FREQ_BB].set(get_freq(FREQ_IDX_RIVER_TRIPLE_BARREL_BB))
    tensor = tensor.at[IDX_RIVER_OVERBET_FREQ_DEALER].set(get_freq(FREQ_IDX_RIVER_OVERBET_DEALER))
    tensor = tensor.at[IDX_RIVER_OVERBET_FREQ_BB].set(get_freq(FREQ_IDX_RIVER_OVERBET_BB))
    tensor = tensor.at[IDX_AGG_SHOWDOWN_WINRATE].set(get_freq(FREQ_IDX_SHOWDOWN_WIN))
    tensor = tensor.at[IDX_AGG_ALLIN_SHOWDOWN_WINRATE].set(get_freq(FREQ_IDX_ALLIN_SHOWDOWN_WIN))
    tensor = tensor.at[IDX_AGG_AVG_STREETS_SEEN].set(get_avg(AVG_IDX_STREETS_SEEN, 2.0) / 4.0)
    tensor = tensor.at[IDX_AGG_ROUNDS_OBSERVED].set(_log_rounds(rounds_observed))
    tensor = tensor.at[IDX_AGG_ALLIN_FREQ_DEALER].set(get_freq(FREQ_IDX_ALLIN_DEALER))
    tensor = tensor.at[IDX_AGG_ALLIN_FREQ_BB].set(get_freq(FREQ_IDX_ALLIN_BB))
    tensor = tensor.at[IDX_AGG_ALLIN_CALL_FREQ_DEALER].set(get_freq(FREQ_IDX_ALLIN_CALL_DEALER))
    tensor = tensor.at[IDX_AGG_ALLIN_CALL_FREQ_BB].set(get_freq(FREQ_IDX_ALLIN_CALL_BB))
    tensor = tensor.at[IDX_AGG_WENT_TO_SHOWDOWN_DEALER].set(get_freq(FREQ_IDX_WENT_SHOWDOWN_DEALER))
    tensor = tensor.at[IDX_AGG_WENT_TO_SHOWDOWN_BB].set(get_freq(FREQ_IDX_WENT_SHOWDOWN_BB))
    tensor = tensor.at[IDX_AGG_RERAISE_FREQ_DEALER].set(0.5)
    tensor = tensor.at[IDX_AGG_RERAISE_FREQ_BB].set(0.5)
    tensor = tensor.at[IDX_AGG_RESERVED_1].set(0.5)
    tensor = tensor.at[IDX_AGG_RESERVED_2].set(0.5)
    
    return tensor


@jax.jit
def vec_opponent_stats_to_tensor(stats: VecOpponentStats) -> chex.Array:
    """Convert vectorized opponent stats to (E, 64) tensor for model input."""
    return jax.vmap(lambda fs, avgs, ro: _single_stats_to_tensor(fs, avgs, ro))(
        stats.freq_stats, stats.avg_stats, stats.rounds_observed
    )