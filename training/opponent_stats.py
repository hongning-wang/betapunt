"""Opponent statistics tracking for history encoding.

Memory-optimized version using jnp.stack() for true batched operations.

FIXES APPLIED:
1. ALLIN_CALL trial condition changed from `facing_bet` to `facing_allin`

Note: bet_amount bug is in transformer_train.py, not here.
"""

import jax
import jax.numpy as jnp
import chex

# =============================================================================
# Output tensor indices (64 features)
# =============================================================================

HISTORY_DIM = 64
STACK_SIZE = 400

# PREFLOP (8 features, indices 0-7)
IDX_PREFLOP_VPIP_DEALER = 0
IDX_PREFLOP_VPIP_BB = 1
IDX_PREFLOP_RAISE_FREQ_DEALER = 2
IDX_PREFLOP_RAISE_FREQ_BB = 3
IDX_PREFLOP_FOLD_FREQ_DEALER = 4
IDX_PREFLOP_FOLD_FREQ_BB = 5
IDX_PREFLOP_OPEN_RAISE_SIZE_AVG_DEALER = 6
IDX_PREFLOP_OPEN_RAISE_SIZE_AVG_BB = 7

# PREFLOP EXTRA (2 features, indices 8-9)
IDX_LIMP_FREQ_DEALER = 8
IDX_FOLD_TO_PROBE_FREQ = 9

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
IDX_TURN_DOUBLE_BARREL_FREQ = 34
IDX_BIG_RAISE_FREQ = 35

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
IDX_RIVER_TRIPLE_BARREL_FREQ = 46
IDX_MIN_RAISE_FREQ = 47
IDX_RIVER_OVERBET_FREQ_DEALER = 48
IDX_RIVER_OVERBET_FREQ_BB = 49

# AGGREGATED (14 features, indices 50-63)
IDX_AGG_SHOWDOWN_WINRATE = 50
IDX_AGG_ALLIN_SHOWDOWN_WINRATE = 51
IDX_AGG_AVG_STREETS_SEEN = 52
IDX_AGG_ROUNDS_OBSERVED = 53
IDX_AGG_ALLIN_FREQ_DEALER = 54
IDX_AGG_ALLIN_FREQ_BB = 55
IDX_AGG_ALLIN_CALL_FREQ = 56
IDX_AGG_BIG_CALL_FREQ = 57
IDX_AGG_WENT_TO_SHOWDOWN_DEALER = 58
IDX_AGG_WENT_TO_SHOWDOWN_BB = 59
IDX_AGG_RERAISE_FREQ_DEALER = 60
IDX_AGG_RERAISE_FREQ_BB = 61
IDX_FOLD_TO_CBET_DEALER = 62
IDX_FOLD_TO_CBET_BB = 63

# =============================================================================
# Internal tracking indices - compact (no wasted slots)
# =============================================================================

NUM_FREQ_STATS = 54

FREQ_IDX_VPIP_DEALER = 0
FREQ_IDX_VPIP_BB = 1
FREQ_IDX_PREFLOP_RAISE_DEALER = 2
FREQ_IDX_PREFLOP_RAISE_BB = 3
FREQ_IDX_PREFLOP_FOLD_DEALER = 4
FREQ_IDX_PREFLOP_FOLD_BB = 5
FREQ_IDX_LIMP_DEALER = 6
FREQ_IDX_FOLD_TO_PROBE = 7
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
FREQ_IDX_TURN_DOUBLE_BARREL = 28
FREQ_IDX_BIG_RAISE = 29
FREQ_IDX_RIVER_FOLD_DEALER = 30
FREQ_IDX_RIVER_FOLD_BB = 31
FREQ_IDX_RIVER_CALL_DEALER = 32
FREQ_IDX_RIVER_CALL_BB = 33
FREQ_IDX_RIVER_RAISE_DEALER = 34
FREQ_IDX_RIVER_RAISE_BB = 35
FREQ_IDX_RIVER_BET_DEALER = 36
FREQ_IDX_RIVER_BET_BB = 37
FREQ_IDX_RIVER_TRIPLE_BARREL = 38
FREQ_IDX_MIN_RAISE = 39
FREQ_IDX_RIVER_OVERBET_DEALER = 40
FREQ_IDX_RIVER_OVERBET_BB = 41
FREQ_IDX_SHOWDOWN_WIN = 42
FREQ_IDX_ALLIN_SHOWDOWN_WIN = 43
FREQ_IDX_ALLIN_DEALER = 44
FREQ_IDX_ALLIN_BB = 45
FREQ_IDX_ALLIN_CALL = 46
FREQ_IDX_BIG_CALL = 47
FREQ_IDX_WENT_SHOWDOWN_DEALER = 48
FREQ_IDX_WENT_SHOWDOWN_BB = 49
FREQ_IDX_RERAISE_DEALER = 50
FREQ_IDX_RERAISE_BB = 51
FREQ_IDX_FOLD_TO_CBET_DEALER = 52
FREQ_IDX_FOLD_TO_CBET_BB = 53

NUM_AVG_STATS = 9

AVG_IDX_OPEN_RAISE_SIZE_DEALER = 0
AVG_IDX_OPEN_RAISE_SIZE_BB = 1
AVG_IDX_FLOP_BET_SIZE_DEALER = 2
AVG_IDX_FLOP_BET_SIZE_BB = 3
AVG_IDX_TURN_BET_SIZE_DEALER = 4
AVG_IDX_TURN_BET_SIZE_BB = 5
AVG_IDX_RIVER_BET_SIZE_DEALER = 6
AVG_IDX_RIVER_BET_SIZE_BB = 7
AVG_IDX_STREETS_SEEN = 8

# Action and Phase constants
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


@chex.dataclass(frozen=True)
class VecOpponentStats:
    """Vectorized opponent stats for E parallel environments."""
    freq_stats: chex.Array  # (E, NUM_FREQ_STATS, 2) - [successes, trials]
    avg_stats: chex.Array   # (E, NUM_AVG_STATS, 2) - [sum, count]
    rounds_observed: chex.Array  # (E,)
    preflop_aggressor: chex.Array  # (E,) int32: 0=dealer, 1=bb, -1=none
    opp_bet_flop: chex.Array  # (E,) bool
    opp_bet_turn: chex.Array  # (E,) bool
    opp_checked_this_street: chex.Array  # (E,) bool
    hero_checked_this_street: chex.Array  # (E,) bool
    street_has_raise: chex.Array  # (E,) bool
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
        hero_checked_this_street=jnp.zeros(num_envs, dtype=jnp.bool_),
        street_has_raise=jnp.zeros(num_envs, dtype=jnp.bool_),
        current_street=jnp.zeros(num_envs, dtype=jnp.int32),
        went_allin_this_hand=jnp.zeros(num_envs, dtype=jnp.bool_),
    )


@jax.jit
def vec_reset_round_state(opp_stats: VecOpponentStats, done: chex.Array) -> VecOpponentStats:
    """Reset per-round state on hand completion. Keeps accumulated stats!"""
    return opp_stats.replace(
        preflop_aggressor=jnp.where(done, -1, opp_stats.preflop_aggressor),
        opp_bet_flop=jnp.where(done, False, opp_stats.opp_bet_flop),
        opp_bet_turn=jnp.where(done, False, opp_stats.opp_bet_turn),
        opp_checked_this_street=jnp.where(done, False, opp_stats.opp_checked_this_street),
        hero_checked_this_street=jnp.where(done, False, opp_stats.hero_checked_this_street),
        street_has_raise=jnp.where(done, False, opp_stats.street_has_raise),
        current_street=jnp.where(done, 0, opp_stats.current_street),
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
    """Update opponent statistics. Truly batched using jnp.stack().
    
    Args:
        opp_stats: Current opponent statistics
        phase: Game phase (PHASE_PREFLOP, PHASE_FLOP, etc.)
        action: Action taken (0-12)
        is_opp_turn: Whether it's the opponent's turn (vs hero's turn)
        opp_is_dealer: Whether opponent is the dealer (vs BB)
        to_call: Amount needed to call
        pot_before: Pot size before action
        bet_amount: Actual chips committed by the action-taker this action
    """
    
    update = is_opp_turn
    is_hero_turn = ~is_opp_turn
    opp_is_bb = ~opp_is_dealer
    
    # Action categorizations
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
    
    # Street detection
    is_preflop = phase == PHASE_PREFLOP
    is_flop = phase == PHASE_FLOP
    is_turn = phase == PHASE_TURN
    is_river = phase == PHASE_RIVER
    street_changed = phase != opp_stats.current_street
    
    bet_frac = jnp.where(pot_before > 0, bet_amount / pot_before, 0.0)
    
    # Position + street combinations
    preflop_opp = update & is_preflop & ~is_discard
    preflop_dealer = preflop_opp & opp_is_dealer
    preflop_bb = preflop_opp & opp_is_bb
    
    flop_opp = update & is_flop
    flop_dealer = flop_opp & opp_is_dealer
    flop_bb = flop_opp & opp_is_bb
    
    turn_opp = update & is_turn
    turn_dealer = turn_opp & opp_is_dealer
    turn_bb = turn_opp & opp_is_bb
    
    river_opp = update & is_river
    river_dealer = river_opp & opp_is_dealer
    river_bb = river_opp & opp_is_bb
    
    any_dealer = update & opp_is_dealer
    any_bb = update & opp_is_bb
    
    # Special conditions
    check_raise_opportunity = flop_opp & opp_stats.opp_checked_this_street & facing_bet
    
    opp_was_aggressor = jnp.where(opp_is_dealer,
                                   opp_stats.preflop_aggressor == 0,
                                   opp_stats.preflop_aggressor == 1)
    hero_was_aggressor = jnp.where(opp_is_dealer,
                                    opp_stats.preflop_aggressor == 1,
                                    opp_stats.preflop_aggressor == 0)
    
    is_overbet = (is_bet | is_raise) & (bet_amount > pot_before)
    is_big_call = is_call & (to_call > pot_before * 0.5)
    
    # FIX: facing_allin condition for proper ALLIN_CALL trial
    facing_allin = (to_call + pot_before) == (2 * STACK_SIZE)
    is_allin_call = is_call & facing_allin
    
    is_big_raise = is_raise & (bet_amount > pot_before)
    is_min_raise = is_raise & (bet_amount < pot_before * 0.4)
    
    facing_cbet = flop_opp & facing_bet & hero_was_aggressor
    facing_probe = update & opp_was_aggressor & opp_stats.opp_checked_this_street & facing_bet
    
    reraise_trial_dealer = any_dealer & opp_stats.street_has_raise
    reraise_trial_bb = any_bb & opp_stats.street_has_raise
    
    double_barrel_trial = update & is_turn & opp_stats.opp_bet_flop & ~facing_bet
    triple_barrel_trial = update & is_river & opp_stats.opp_bet_flop & opp_stats.opp_bet_turn & ~facing_bet
    
    false_cond = jnp.zeros_like(update)
    
    # =========================================================================
    # Build trial conditions - Order MUST match FREQ_IDX_* constants!
    # =========================================================================
    trial_list = [
        preflop_dealer,                                    # 0: VPIP_DEALER
        preflop_bb,                                        # 1: VPIP_BB  
        preflop_dealer,                                    # 2: PREFLOP_RAISE_DEALER
        preflop_bb,                                        # 3: PREFLOP_RAISE_BB
        preflop_dealer,                                    # 4: PREFLOP_FOLD_DEALER
        preflop_bb,                                        # 5: PREFLOP_FOLD_BB
        preflop_dealer,                                    # 6: LIMP_DEALER
        facing_probe,                                      # 7: FOLD_TO_PROBE
        flop_dealer,                                       # 8: FLOP_FOLD_DEALER
        flop_bb,                                           # 9: FLOP_FOLD_BB
        flop_dealer,                                       # 10: FLOP_CALL_DEALER
        flop_bb,                                           # 11: FLOP_CALL_BB
        flop_dealer,                                       # 12: FLOP_RAISE_DEALER
        flop_bb,                                           # 13: FLOP_RAISE_BB
        flop_dealer,                                       # 14: FLOP_BET_DEALER
        flop_bb,                                           # 15: FLOP_BET_BB
        check_raise_opportunity,                           # 16: FLOP_CHECK_RAISE
        flop_dealer & opp_was_aggressor,                   # 17: FLOP_CBET_DEALER
        flop_bb & opp_was_aggressor,                       # 18: FLOP_CBET_BB
        flop_bb & (opp_stats.preflop_aggressor == 0),      # 19: FLOP_DONK_BB
        turn_dealer,                                       # 20: TURN_FOLD_DEALER
        turn_bb,                                           # 21: TURN_FOLD_BB
        turn_dealer,                                       # 22: TURN_CALL_DEALER
        turn_bb,                                           # 23: TURN_CALL_BB
        turn_dealer,                                       # 24: TURN_RAISE_DEALER
        turn_bb,                                           # 25: TURN_RAISE_BB
        turn_dealer,                                       # 26: TURN_BET_DEALER
        turn_bb,                                           # 27: TURN_BET_BB
        double_barrel_trial,                               # 28: TURN_DOUBLE_BARREL
        update & is_raise,                                 # 29: BIG_RAISE
        river_dealer,                                      # 30: RIVER_FOLD_DEALER
        river_bb,                                          # 31: RIVER_FOLD_BB
        river_dealer,                                      # 32: RIVER_CALL_DEALER
        river_bb,                                          # 33: RIVER_CALL_BB
        river_dealer,                                      # 34: RIVER_RAISE_DEALER
        river_bb,                                          # 35: RIVER_RAISE_BB
        river_dealer,                                      # 36: RIVER_BET_DEALER
        river_bb,                                          # 37: RIVER_BET_BB
        triple_barrel_trial,                               # 38: RIVER_TRIPLE_BARREL
        update & is_raise,                                 # 39: MIN_RAISE
        river_dealer,                                      # 40: RIVER_OVERBET_DEALER
        river_bb,                                          # 41: RIVER_OVERBET_BB
        false_cond,                                        # 42: SHOWDOWN_WIN
        false_cond,                                        # 43: ALLIN_SHOWDOWN_WIN
        any_dealer,                                        # 44: ALLIN_DEALER
        any_bb,                                            # 45: ALLIN_BB
        update & facing_allin,                             # 46: ALLIN_CALL - FIX: trial when facing all-in
        update & facing_bet,                               # 47: BIG_CALL - trial when facing any bet
        false_cond,                                        # 48: WENT_SHOWDOWN_DEALER
        false_cond,                                        # 49: WENT_SHOWDOWN_BB
        reraise_trial_dealer,                              # 50: RERAISE_DEALER
        reraise_trial_bb,                                  # 51: RERAISE_BB
        facing_cbet & opp_is_dealer,                       # 52: FOLD_TO_CBET_DEALER
        facing_cbet & opp_is_bb,                           # 53: FOLD_TO_CBET_BB
    ]
    
    # Success conditions
    success_list = [
        is_vpip,           # 0: VPIP_DEALER
        is_vpip,           # 1: VPIP_BB
        is_raise,          # 2: PREFLOP_RAISE_DEALER
        is_raise,          # 3: PREFLOP_RAISE_BB
        is_fold,           # 4: PREFLOP_FOLD_DEALER
        is_fold,           # 5: PREFLOP_FOLD_BB
        is_call,           # 6: LIMP_DEALER
        is_fold,           # 7: FOLD_TO_PROBE
        is_fold,           # 8: FLOP_FOLD_DEALER
        is_fold,           # 9: FLOP_FOLD_BB
        is_call,           # 10: FLOP_CALL_DEALER
        is_call,           # 11: FLOP_CALL_BB
        is_raise_vs_bet,   # 12: FLOP_RAISE_DEALER
        is_raise_vs_bet,   # 13: FLOP_RAISE_BB
        is_bet,            # 14: FLOP_BET_DEALER
        is_bet,            # 15: FLOP_BET_BB
        is_raise_vs_bet,   # 16: FLOP_CHECK_RAISE
        is_bet,            # 17: FLOP_CBET_DEALER
        is_bet,            # 18: FLOP_CBET_BB
        is_bet,            # 19: FLOP_DONK_BB
        is_fold,           # 20: TURN_FOLD_DEALER
        is_fold,           # 21: TURN_FOLD_BB
        is_call,           # 22: TURN_CALL_DEALER
        is_call,           # 23: TURN_CALL_BB
        is_raise_vs_bet,   # 24: TURN_RAISE_DEALER
        is_raise_vs_bet,   # 25: TURN_RAISE_BB
        is_bet,            # 26: TURN_BET_DEALER
        is_bet,            # 27: TURN_BET_BB
        is_bet,            # 28: TURN_DOUBLE_BARREL
        is_big_raise,      # 29: BIG_RAISE
        is_fold,           # 30: RIVER_FOLD_DEALER
        is_fold,           # 31: RIVER_FOLD_BB
        is_call,           # 32: RIVER_CALL_DEALER
        is_call,           # 33: RIVER_CALL_BB
        is_raise_vs_bet,   # 34: RIVER_RAISE_DEALER
        is_raise_vs_bet,   # 35: RIVER_RAISE_BB
        is_bet,            # 36: RIVER_BET_DEALER
        is_bet,            # 37: RIVER_BET_BB
        is_bet,            # 38: RIVER_TRIPLE_BARREL
        is_min_raise,      # 39: MIN_RAISE
        is_overbet,        # 40: RIVER_OVERBET_DEALER
        is_overbet,        # 41: RIVER_OVERBET_BB
        false_cond,        # 42: SHOWDOWN_WIN
        false_cond,        # 43: ALLIN_SHOWDOWN_WIN
        is_allin,          # 44: ALLIN_DEALER
        is_allin,          # 45: ALLIN_BB
        is_call,           # 46: ALLIN_CALL - success = called
        is_big_call,       # 47: BIG_CALL
        false_cond,        # 48: WENT_SHOWDOWN_DEALER
        false_cond,        # 49: WENT_SHOWDOWN_BB
        is_raise,          # 50: RERAISE_DEALER
        is_raise,          # 51: RERAISE_BB
        is_fold,           # 52: FOLD_TO_CBET_DEALER
        is_fold,           # 53: FOLD_TO_CBET_BB
    ]
    
    # Stack into (E, NUM_FREQ_STATS)
    freq_trials = jnp.stack(trial_list, axis=1).astype(jnp.float32)
    freq_successes = jnp.stack([t & s for t, s in zip(trial_list, success_list)], axis=1).astype(jnp.float32)
    
    freq = opp_stats.freq_stats
    freq = freq.at[:, :, 0].add(freq_successes)
    freq = freq.at[:, :, 1].add(freq_trials)
    
    # =========================================================================
    # Average stats
    # =========================================================================
    avg_trial_list = [
        preflop_dealer & is_raise,              # 0: OPEN_RAISE_SIZE_DEALER
        preflop_bb & is_raise,                  # 1: OPEN_RAISE_SIZE_BB
        flop_dealer & (is_bet | is_raise),      # 2: FLOP_BET_SIZE_DEALER
        flop_bb & (is_bet | is_raise),          # 3: FLOP_BET_SIZE_BB
        turn_dealer & (is_bet | is_raise),      # 4: TURN_BET_SIZE_DEALER
        turn_bb & (is_bet | is_raise),          # 5: TURN_BET_SIZE_BB
        river_dealer & (is_bet | is_raise),     # 6: RIVER_BET_SIZE_DEALER
        river_bb & (is_bet | is_raise),         # 7: RIVER_BET_SIZE_BB
        false_cond,                             # 8: STREETS_SEEN (updated in showdown)
    ]
    
    avg_counts = jnp.stack(avg_trial_list, axis=1).astype(jnp.float32)
    avg_sums = avg_counts * bet_frac[:, None]
    
    avg = opp_stats.avg_stats
    avg = avg.at[:, :, 0].add(avg_sums)
    avg = avg.at[:, :, 1].add(avg_counts)
    
    # =========================================================================
    # Update per-round state
    # =========================================================================
    opp_checked_reset = jnp.where(street_changed, False, opp_stats.opp_checked_this_street)
    hero_checked_reset = jnp.where(street_changed, False, opp_stats.hero_checked_this_street)
    street_has_raise_reset = jnp.where(street_changed, False, opp_stats.street_has_raise)
    
    new_opp_checked = jnp.where(update & is_check, True, opp_checked_reset)
    new_hero_checked = jnp.where(is_hero_turn & is_check, True, hero_checked_reset)
    new_street_has_raise = jnp.where(is_raise, True, street_has_raise_reset)
    
    # Preflop aggressor tracking (both players)
    new_preflop_agg = opp_stats.preflop_aggressor
    new_preflop_agg = jnp.where(update & is_preflop & is_raise,
                                 jnp.where(opp_is_dealer, 0, 1),
                                 new_preflop_agg)
    new_preflop_agg = jnp.where(is_hero_turn & is_preflop & is_raise,
                                 jnp.where(opp_is_dealer, 1, 0),
                                 new_preflop_agg)
    
    new_opp_bet_flop = jnp.where(update & is_flop & is_bet, True, opp_stats.opp_bet_flop)
    new_opp_bet_turn = jnp.where(update & is_turn & is_bet, True, opp_stats.opp_bet_turn)
    new_went_allin = jnp.where(update & is_allin, True, opp_stats.went_allin_this_hand)
    
    return opp_stats.replace(
        freq_stats=freq,
        avg_stats=avg,
        preflop_aggressor=new_preflop_agg,
        opp_bet_flop=new_opp_bet_flop,
        opp_bet_turn=new_opp_bet_turn,
        opp_checked_this_street=new_opp_checked,
        hero_checked_this_street=new_hero_checked,
        street_has_raise=new_street_has_raise,
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
    opp_is_bb = ~opp_is_dealer
    
    showdown_cond = done & went_to_showdown
    allin_showdown_cond = showdown_cond & opp_stats.went_allin_this_hand
    
    freq = opp_stats.freq_stats
    
    freq = freq.at[:, FREQ_IDX_SHOWDOWN_WIN, 0].add((showdown_cond & opp_won).astype(jnp.float32))
    freq = freq.at[:, FREQ_IDX_SHOWDOWN_WIN, 1].add(showdown_cond.astype(jnp.float32))
    
    freq = freq.at[:, FREQ_IDX_ALLIN_SHOWDOWN_WIN, 0].add((allin_showdown_cond & opp_won).astype(jnp.float32))
    freq = freq.at[:, FREQ_IDX_ALLIN_SHOWDOWN_WIN, 1].add(allin_showdown_cond.astype(jnp.float32))
    
    freq = freq.at[:, FREQ_IDX_WENT_SHOWDOWN_DEALER, 0].add((done & opp_is_dealer & went_to_showdown).astype(jnp.float32))
    freq = freq.at[:, FREQ_IDX_WENT_SHOWDOWN_DEALER, 1].add((done & opp_is_dealer).astype(jnp.float32))
    
    freq = freq.at[:, FREQ_IDX_WENT_SHOWDOWN_BB, 0].add((done & opp_is_bb & went_to_showdown).astype(jnp.float32))
    freq = freq.at[:, FREQ_IDX_WENT_SHOWDOWN_BB, 1].add((done & opp_is_bb).astype(jnp.float32))
    
    # Streets seen mapping: PREFLOP=0->1, FLOP=3->2, TURN=4->3, RIVER=5->4
    phase = opp_stats.current_street
    streets_seen = jnp.where(
        phase == PHASE_PREFLOP, 1.0,
        jnp.where(phase == PHASE_FLOP, 2.0,
            jnp.where(phase == PHASE_TURN, 3.0, 4.0)))
    
    avg = opp_stats.avg_stats
    avg = avg.at[:, AVG_IDX_STREETS_SEEN, 0].add(jnp.where(done, streets_seen, 0.0))
    avg = avg.at[:, AVG_IDX_STREETS_SEEN, 1].add(done.astype(jnp.float32))
    
    return opp_stats.replace(freq_stats=freq, avg_stats=avg)


# =============================================================================
# Conversion to tensor
# =============================================================================

def _single_stats_to_tensor(freq_stats, avg_stats, rounds_observed):
    """Convert single env stats to 64-dim tensor."""
    trials = freq_stats[:, 1]
    successes = freq_stats[:, 0]
    freqs = jnp.where(trials > 0, successes / trials, 0.5)
    
    avg_defaults = jnp.array([2.5, 3.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 2.0], dtype=jnp.float32)
    avg_counts = avg_stats[:, 1]
    avg_sums = avg_stats[:, 0]
    avgs = jnp.where(avg_counts > 0, avg_sums / avg_counts, avg_defaults)
    
    return jnp.array([
        freqs[FREQ_IDX_VPIP_DEALER],               # 0
        freqs[FREQ_IDX_VPIP_BB],                   # 1
        freqs[FREQ_IDX_PREFLOP_RAISE_DEALER],      # 2
        freqs[FREQ_IDX_PREFLOP_RAISE_BB],          # 3
        freqs[FREQ_IDX_PREFLOP_FOLD_DEALER],       # 4
        freqs[FREQ_IDX_PREFLOP_FOLD_BB],           # 5
        avgs[AVG_IDX_OPEN_RAISE_SIZE_DEALER],      # 6
        avgs[AVG_IDX_OPEN_RAISE_SIZE_BB],          # 7
        freqs[FREQ_IDX_LIMP_DEALER],               # 8
        freqs[FREQ_IDX_FOLD_TO_PROBE],             # 9
        freqs[FREQ_IDX_FLOP_FOLD_DEALER],          # 10
        freqs[FREQ_IDX_FLOP_FOLD_BB],              # 11
        freqs[FREQ_IDX_FLOP_CALL_DEALER],          # 12
        freqs[FREQ_IDX_FLOP_CALL_BB],              # 13
        freqs[FREQ_IDX_FLOP_RAISE_DEALER],         # 14
        freqs[FREQ_IDX_FLOP_RAISE_BB],             # 15
        freqs[FREQ_IDX_FLOP_BET_DEALER],           # 16
        freqs[FREQ_IDX_FLOP_BET_BB],               # 17
        avgs[AVG_IDX_FLOP_BET_SIZE_DEALER],        # 18
        avgs[AVG_IDX_FLOP_BET_SIZE_BB],            # 19
        freqs[FREQ_IDX_FLOP_CHECK_RAISE],          # 20
        freqs[FREQ_IDX_FLOP_CBET_DEALER],          # 21
        freqs[FREQ_IDX_FLOP_CBET_BB],              # 22
        freqs[FREQ_IDX_FLOP_DONK_BB],              # 23
        freqs[FREQ_IDX_TURN_FOLD_DEALER],          # 24
        freqs[FREQ_IDX_TURN_FOLD_BB],              # 25
        freqs[FREQ_IDX_TURN_CALL_DEALER],          # 26
        freqs[FREQ_IDX_TURN_CALL_BB],              # 27
        freqs[FREQ_IDX_TURN_RAISE_DEALER],         # 28
        freqs[FREQ_IDX_TURN_RAISE_BB],             # 29
        freqs[FREQ_IDX_TURN_BET_DEALER],           # 30
        freqs[FREQ_IDX_TURN_BET_BB],               # 31
        avgs[AVG_IDX_TURN_BET_SIZE_DEALER],        # 32
        avgs[AVG_IDX_TURN_BET_SIZE_BB],            # 33
        freqs[FREQ_IDX_TURN_DOUBLE_BARREL],        # 34
        freqs[FREQ_IDX_BIG_RAISE],                 # 35
        freqs[FREQ_IDX_RIVER_FOLD_DEALER],         # 36
        freqs[FREQ_IDX_RIVER_FOLD_BB],             # 37
        freqs[FREQ_IDX_RIVER_CALL_DEALER],         # 38
        freqs[FREQ_IDX_RIVER_CALL_BB],             # 39
        freqs[FREQ_IDX_RIVER_RAISE_DEALER],        # 40
        freqs[FREQ_IDX_RIVER_RAISE_BB],            # 41
        freqs[FREQ_IDX_RIVER_BET_DEALER],          # 42
        freqs[FREQ_IDX_RIVER_BET_BB],              # 43
        avgs[AVG_IDX_RIVER_BET_SIZE_DEALER],       # 44
        avgs[AVG_IDX_RIVER_BET_SIZE_BB],           # 45
        freqs[FREQ_IDX_RIVER_TRIPLE_BARREL],       # 46
        freqs[FREQ_IDX_MIN_RAISE],                 # 47
        freqs[FREQ_IDX_RIVER_OVERBET_DEALER],      # 48
        freqs[FREQ_IDX_RIVER_OVERBET_BB],          # 49
        freqs[FREQ_IDX_SHOWDOWN_WIN],              # 50
        freqs[FREQ_IDX_ALLIN_SHOWDOWN_WIN],        # 51
        avgs[AVG_IDX_STREETS_SEEN] / 4.0,          # 52: normalized to [0, 1]
        jnp.log1p(rounds_observed) / 5.0,          # 53: log-scaled rounds
        freqs[FREQ_IDX_ALLIN_DEALER],              # 54
        freqs[FREQ_IDX_ALLIN_BB],                  # 55
        freqs[FREQ_IDX_ALLIN_CALL],                # 56
        freqs[FREQ_IDX_BIG_CALL],                  # 57
        freqs[FREQ_IDX_WENT_SHOWDOWN_DEALER],      # 58
        freqs[FREQ_IDX_WENT_SHOWDOWN_BB],          # 59
        freqs[FREQ_IDX_RERAISE_DEALER],            # 60
        freqs[FREQ_IDX_RERAISE_BB],                # 61
        freqs[FREQ_IDX_FOLD_TO_CBET_DEALER],       # 62
        freqs[FREQ_IDX_FOLD_TO_CBET_BB],           # 63
    ], dtype=jnp.float32)


@jax.jit
def vec_opponent_stats_to_tensor(stats: VecOpponentStats) -> chex.Array:
    """Convert vectorized opponent stats to (E, 64) tensor."""
    return jax.vmap(_single_stats_to_tensor)(
        stats.freq_stats, stats.avg_stats, stats.rounds_observed
    )