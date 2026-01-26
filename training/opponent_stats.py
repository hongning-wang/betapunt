"""Opponent statistics tracking for history encoding.

This module provides the OpponentStats class and helper functions for tracking
64 opponent history features used by the HistoryEncoder in model.py.

Feature List (64 total):

PREFLOP (8):
- vpip_dealer, vpip_bb
- raise_freq_dealer, raise_freq_bb
- fold_freq_dealer, fold_freq_bb
- open_raise_size_avg_dealer, open_raise_size_avg_bb

DISCARD (2):
- discard_high_card_freq_dealer, discard_high_card_freq_bb

FLOP (14):
- fold_freq_dealer, fold_freq_bb
- call_freq_dealer, call_freq_bb
- raise_freq_dealer, raise_freq_bb
- bet_freq_dealer, bet_freq_bb
- avg_bet_size_dealer, avg_bet_size_bb
- check_raise_freq
- cbet_freq_dealer, cbet_freq_bb
- donk_bet_freq_bb

TURN (12):
- fold_freq_dealer, fold_freq_bb
- call_freq_dealer, call_freq_bb
- raise_freq_dealer, raise_freq_bb
- bet_freq_dealer, bet_freq_bb
- avg_bet_size_dealer, avg_bet_size_bb
- double_barrel_freq_dealer, double_barrel_freq_bb

RIVER (14):
- fold_freq_dealer, fold_freq_bb
- call_freq_dealer, call_freq_bb
- raise_freq_dealer, raise_freq_bb
- bet_freq_dealer, bet_freq_bb
- avg_bet_size_dealer, avg_bet_size_bb
- triple_barrel_freq_dealer, triple_barrel_freq_bb
- overbet_freq_dealer, overbet_freq_bb

AGGREGATED (14):
- showdown_winrate
- allin_showdown_winrate
- avg_streets_seen
- rounds_observed
- allin_freq_dealer, allin_freq_bb
- allin_call_freq_dealer, allin_call_freq_bb
- went_to_showdown_dealer, went_to_showdown_bb
- reraise_freq_dealer, reraise_freq_bb
"""

import jax
import jax.numpy as jnp
import chex
from typing import Tuple

# Feature indices for the 64-dimensional history tensor
# This defines the exact ordering of features

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

# Padding to reach 64 (indices 62-63) - reserved for future use
IDX_RESERVED_1 = 62
IDX_RESERVED_2 = 63

HISTORY_DIM = 64


@chex.dataclass(frozen=True)
class OpponentStats:
    """Tracks opponent statistics for history encoding.
    
    Uses (successes, trials) pairs for frequency-based stats.
    Uses (sum, count) pairs for average-based stats.
    
    All arrays are flat to enable efficient JAX operations.
    """
    # Frequency stats: (num_stats, 2) where [:, 0] = successes, [:, 1] = trials
    # We track dealer and bb versions separately
    freq_stats: chex.Array  # (num_freq_stats, 2)
    
    # Average stats: (num_avg_stats, 2) where [:, 0] = sum, [:, 1] = count
    avg_stats: chex.Array  # (num_avg_stats, 2)
    
    # Round-level tracking
    rounds_observed: chex.Array  # scalar
    
    # Per-round state tracking (reset each round)
    # Track who was preflop aggressor for cbet/donk detection
    preflop_aggressor: chex.Array  # 0 = dealer, 1 = bb, -1 = none
    
    # Track if opponent bet on each street (for barrel detection)
    opp_bet_flop: chex.Array  # bool
    opp_bet_turn: chex.Array  # bool
    
    # Track if opponent checked this street (for check-raise detection)
    opp_checked_this_street: chex.Array  # bool
    
    # Track current street's pot size for overbet detection
    pot_at_street_start: chex.Array  # float


# Number of frequency-based statistics we track
# Each has (successes, trials) pair
NUM_FREQ_STATS = 50  # Adjust based on actual count

# Number of average-based statistics we track
# Each has (sum, count) pair
NUM_AVG_STATS = 10  # open_raise_size, avg_bet_size per street, avg_streets_seen


def opponent_stats_init() -> OpponentStats:
    """Initialize empty opponent stats."""
    return OpponentStats(
        freq_stats=jnp.zeros((NUM_FREQ_STATS, 2), dtype=jnp.float32),
        avg_stats=jnp.zeros((NUM_AVG_STATS, 2), dtype=jnp.float32),
        rounds_observed=jnp.float32(0),
        preflop_aggressor=jnp.int32(-1),
        opp_bet_flop=jnp.bool_(False),
        opp_bet_turn=jnp.bool_(False),
        opp_checked_this_street=jnp.bool_(False),
        pot_at_street_start=jnp.float32(0),
    )


def opponent_stats_reset_round(stats: OpponentStats) -> OpponentStats:
    """Reset per-round tracking state (called at start of each round)."""
    return stats.replace(
        preflop_aggressor=jnp.int32(-1),
        opp_bet_flop=jnp.bool_(False),
        opp_bet_turn=jnp.bool_(False),
        opp_checked_this_street=jnp.bool_(False),
        pot_at_street_start=jnp.float32(0),
    )


def _freq(successes: chex.Array, trials: chex.Array) -> chex.Array:
    """Compute frequency with 0.5 default when trials=0."""
    return jnp.where(trials > 0, successes / trials, 0.5)


def _avg(total: chex.Array, count: chex.Array, default: float = 0.0) -> chex.Array:
    """Compute average with default when count=0."""
    return jnp.where(count > 0, total / count, default)


def _log_rounds(count: chex.Array) -> chex.Array:
    """Log-scale rounds observed for better numerical properties."""
    return jnp.log1p(count)


@jax.jit
def opponent_stats_to_tensor(stats: OpponentStats) -> chex.Array:
    """Convert opponent stats to 64-dimensional tensor for model input.
    
    Args:
        stats: OpponentStats object
        
    Returns:
        tensor: (64,) float32 tensor of features
    """
    tensor = jnp.zeros(HISTORY_DIM, dtype=jnp.float32)
    
    # Helper to get frequency from freq_stats array
    def get_freq(idx):
        return _freq(stats.freq_stats[idx, 0], stats.freq_stats[idx, 1])
    
    def get_avg(idx, default=0.0):
        return _avg(stats.avg_stats[idx, 0], stats.avg_stats[idx, 1], default)
    
    # === PREFLOP ===
    tensor = tensor.at[IDX_PREFLOP_VPIP_DEALER].set(get_freq(0))
    tensor = tensor.at[IDX_PREFLOP_VPIP_BB].set(get_freq(1))
    tensor = tensor.at[IDX_PREFLOP_RAISE_FREQ_DEALER].set(get_freq(2))
    tensor = tensor.at[IDX_PREFLOP_RAISE_FREQ_BB].set(get_freq(3))
    tensor = tensor.at[IDX_PREFLOP_FOLD_FREQ_DEALER].set(get_freq(4))
    tensor = tensor.at[IDX_PREFLOP_FOLD_FREQ_BB].set(get_freq(5))
    tensor = tensor.at[IDX_PREFLOP_OPEN_RAISE_SIZE_AVG_DEALER].set(get_avg(0, 2.5))  # default ~2.5x
    tensor = tensor.at[IDX_PREFLOP_OPEN_RAISE_SIZE_AVG_BB].set(get_avg(1, 3.0))
    
    # === DISCARD ===
    tensor = tensor.at[IDX_DISCARD_HIGH_CARD_FREQ_DEALER].set(get_freq(6))
    tensor = tensor.at[IDX_DISCARD_HIGH_CARD_FREQ_BB].set(get_freq(7))
    
    # === FLOP ===
    tensor = tensor.at[IDX_FLOP_FOLD_FREQ_DEALER].set(get_freq(8))
    tensor = tensor.at[IDX_FLOP_FOLD_FREQ_BB].set(get_freq(9))
    tensor = tensor.at[IDX_FLOP_CALL_FREQ_DEALER].set(get_freq(10))
    tensor = tensor.at[IDX_FLOP_CALL_FREQ_BB].set(get_freq(11))
    tensor = tensor.at[IDX_FLOP_RAISE_FREQ_DEALER].set(get_freq(12))
    tensor = tensor.at[IDX_FLOP_RAISE_FREQ_BB].set(get_freq(13))
    tensor = tensor.at[IDX_FLOP_BET_FREQ_DEALER].set(get_freq(14))
    tensor = tensor.at[IDX_FLOP_BET_FREQ_BB].set(get_freq(15))
    tensor = tensor.at[IDX_FLOP_AVG_BET_SIZE_DEALER].set(get_avg(2, 0.5))  # default 0.5 pot
    tensor = tensor.at[IDX_FLOP_AVG_BET_SIZE_BB].set(get_avg(3, 0.5))
    tensor = tensor.at[IDX_FLOP_CHECK_RAISE_FREQ].set(get_freq(16))
    tensor = tensor.at[IDX_FLOP_CBET_FREQ_DEALER].set(get_freq(17))
    tensor = tensor.at[IDX_FLOP_CBET_FREQ_BB].set(get_freq(18))
    tensor = tensor.at[IDX_FLOP_DONK_BET_FREQ_BB].set(get_freq(19))
    
    # === TURN ===
    tensor = tensor.at[IDX_TURN_FOLD_FREQ_DEALER].set(get_freq(20))
    tensor = tensor.at[IDX_TURN_FOLD_FREQ_BB].set(get_freq(21))
    tensor = tensor.at[IDX_TURN_CALL_FREQ_DEALER].set(get_freq(22))
    tensor = tensor.at[IDX_TURN_CALL_FREQ_BB].set(get_freq(23))
    tensor = tensor.at[IDX_TURN_RAISE_FREQ_DEALER].set(get_freq(24))
    tensor = tensor.at[IDX_TURN_RAISE_FREQ_BB].set(get_freq(25))
    tensor = tensor.at[IDX_TURN_BET_FREQ_DEALER].set(get_freq(26))
    tensor = tensor.at[IDX_TURN_BET_FREQ_BB].set(get_freq(27))
    tensor = tensor.at[IDX_TURN_AVG_BET_SIZE_DEALER].set(get_avg(4, 0.5))
    tensor = tensor.at[IDX_TURN_AVG_BET_SIZE_BB].set(get_avg(5, 0.5))
    tensor = tensor.at[IDX_TURN_DOUBLE_BARREL_FREQ_DEALER].set(get_freq(28))
    tensor = tensor.at[IDX_TURN_DOUBLE_BARREL_FREQ_BB].set(get_freq(29))
    
    # === RIVER ===
    tensor = tensor.at[IDX_RIVER_FOLD_FREQ_DEALER].set(get_freq(30))
    tensor = tensor.at[IDX_RIVER_FOLD_FREQ_BB].set(get_freq(31))
    tensor = tensor.at[IDX_RIVER_CALL_FREQ_DEALER].set(get_freq(32))
    tensor = tensor.at[IDX_RIVER_CALL_FREQ_BB].set(get_freq(33))
    tensor = tensor.at[IDX_RIVER_RAISE_FREQ_DEALER].set(get_freq(34))
    tensor = tensor.at[IDX_RIVER_RAISE_FREQ_BB].set(get_freq(35))
    tensor = tensor.at[IDX_RIVER_BET_FREQ_DEALER].set(get_freq(36))
    tensor = tensor.at[IDX_RIVER_BET_FREQ_BB].set(get_freq(37))
    tensor = tensor.at[IDX_RIVER_AVG_BET_SIZE_DEALER].set(get_avg(6, 0.5))
    tensor = tensor.at[IDX_RIVER_AVG_BET_SIZE_BB].set(get_avg(7, 0.5))
    tensor = tensor.at[IDX_RIVER_TRIPLE_BARREL_FREQ_DEALER].set(get_freq(38))
    tensor = tensor.at[IDX_RIVER_TRIPLE_BARREL_FREQ_BB].set(get_freq(39))
    tensor = tensor.at[IDX_RIVER_OVERBET_FREQ_DEALER].set(get_freq(40))
    tensor = tensor.at[IDX_RIVER_OVERBET_FREQ_BB].set(get_freq(41))
    
    # === AGGREGATED ===
    tensor = tensor.at[IDX_AGG_SHOWDOWN_WINRATE].set(get_freq(42))
    tensor = tensor.at[IDX_AGG_ALLIN_SHOWDOWN_WINRATE].set(get_freq(43))
    tensor = tensor.at[IDX_AGG_AVG_STREETS_SEEN].set(get_avg(8, 2.0) / 4.0)  # normalize to [0, 1]
    tensor = tensor.at[IDX_AGG_ROUNDS_OBSERVED].set(_log_rounds(stats.rounds_observed))
    tensor = tensor.at[IDX_AGG_ALLIN_FREQ_DEALER].set(get_freq(44))
    tensor = tensor.at[IDX_AGG_ALLIN_FREQ_BB].set(get_freq(45))
    tensor = tensor.at[IDX_AGG_ALLIN_CALL_FREQ_DEALER].set(get_freq(46))
    tensor = tensor.at[IDX_AGG_ALLIN_CALL_FREQ_BB].set(get_freq(47))
    tensor = tensor.at[IDX_AGG_WENT_TO_SHOWDOWN_DEALER].set(get_freq(48))
    tensor = tensor.at[IDX_AGG_WENT_TO_SHOWDOWN_BB].set(get_freq(49))
    tensor = tensor.at[IDX_AGG_RERAISE_FREQ_DEALER].set(get_freq(50 - 1))  # adjust index
    tensor = tensor.at[IDX_AGG_RERAISE_FREQ_BB].set(get_freq(51 - 1))
    
    return tensor


# =============================================================================
# Frequency stat indices (for freq_stats array)
# =============================================================================
# This maps logical stat names to indices in the freq_stats array

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

# Average stat indices (for avg_stats array)
AVG_IDX_OPEN_RAISE_SIZE_DEALER = 0
AVG_IDX_OPEN_RAISE_SIZE_BB = 1
AVG_IDX_FLOP_BET_SIZE_DEALER = 2
AVG_IDX_FLOP_BET_SIZE_BB = 3
AVG_IDX_TURN_BET_SIZE_DEALER = 4
AVG_IDX_TURN_BET_SIZE_BB = 5
AVG_IDX_RIVER_BET_SIZE_DEALER = 6
AVG_IDX_RIVER_BET_SIZE_BB = 7
AVG_IDX_STREETS_SEEN = 8


# =============================================================================
# Update functions
# =============================================================================

@jax.jit
def _update_freq(stats: OpponentStats, idx: int, success: bool) -> OpponentStats:
    """Update a frequency stat."""
    new_freq = stats.freq_stats.at[idx, 0].add(jnp.float32(success))
    new_freq = new_freq.at[idx, 1].add(1.0)
    return stats.replace(freq_stats=new_freq)


@jax.jit
def _update_avg(stats: OpponentStats, idx: int, value: float) -> OpponentStats:
    """Update an average stat."""
    new_avg = stats.avg_stats.at[idx, 0].add(value)
    new_avg = new_avg.at[idx, 1].add(1.0)
    return stats.replace(avg_stats=new_avg)


def is_high_card(card_index: int) -> bool:
    """Check if card is a high card (A, K, Q, J = ranks 12, 11, 10, 9)."""
    rank = card_index % 13
    return rank >= 9


# =============================================================================
# Vectorized opponent stats for parallel environments
# =============================================================================

@chex.dataclass(frozen=True)
class VecOpponentStats:
    """Vectorized opponent stats for E parallel environments."""
    freq_stats: chex.Array  # (E, NUM_FREQ_STATS, 2)
    avg_stats: chex.Array   # (E, NUM_AVG_STATS, 2)
    rounds_observed: chex.Array  # (E,)
    preflop_aggressor: chex.Array  # (E,)
    opp_bet_flop: chex.Array  # (E,)
    opp_bet_turn: chex.Array  # (E,)
    opp_checked_this_street: chex.Array  # (E,)
    pot_at_street_start: chex.Array  # (E,)


def vec_opponent_stats_init(num_envs: int) -> VecOpponentStats:
    """Initialize vectorized opponent stats for num_envs environments."""
    return VecOpponentStats(
        freq_stats=jnp.zeros((num_envs, NUM_FREQ_STATS, 2), dtype=jnp.float32),
        avg_stats=jnp.zeros((num_envs, NUM_AVG_STATS, 2), dtype=jnp.float32),
        rounds_observed=jnp.zeros(num_envs, dtype=jnp.float32),
        preflop_aggressor=jnp.full(num_envs, -1, dtype=jnp.int32),
        opp_bet_flop=jnp.zeros(num_envs, dtype=jnp.bool_),
        opp_bet_turn=jnp.zeros(num_envs, dtype=jnp.bool_),
        opp_checked_this_street=jnp.zeros(num_envs, dtype=jnp.bool_),
        pot_at_street_start=jnp.zeros(num_envs, dtype=jnp.float32),
    )


@jax.jit
def vec_opponent_stats_to_tensor(stats: VecOpponentStats) -> chex.Array:
    """Convert vectorized opponent stats to (E, 64) tensor."""
    return jax.vmap(lambda fs, avgs, ro: _single_stats_to_tensor(fs, avgs, ro))(
        stats.freq_stats, stats.avg_stats, stats.rounds_observed
    )


def _single_stats_to_tensor(freq_stats, avg_stats, rounds_observed):
    """Convert single env stats arrays to 64-dim tensor."""
    tensor = jnp.zeros(HISTORY_DIM, dtype=jnp.float32)
    
    def get_freq(idx):
        return _freq(freq_stats[idx, 0], freq_stats[idx, 1])
    
    def get_avg(idx, default=0.0):
        return _avg(avg_stats[idx, 0], avg_stats[idx, 1], default)
    
    # === PREFLOP ===
    tensor = tensor.at[IDX_PREFLOP_VPIP_DEALER].set(get_freq(FREQ_IDX_VPIP_DEALER))
    tensor = tensor.at[IDX_PREFLOP_VPIP_BB].set(get_freq(FREQ_IDX_VPIP_BB))
    tensor = tensor.at[IDX_PREFLOP_RAISE_FREQ_DEALER].set(get_freq(FREQ_IDX_PREFLOP_RAISE_DEALER))
    tensor = tensor.at[IDX_PREFLOP_RAISE_FREQ_BB].set(get_freq(FREQ_IDX_PREFLOP_RAISE_BB))
    tensor = tensor.at[IDX_PREFLOP_FOLD_FREQ_DEALER].set(get_freq(FREQ_IDX_PREFLOP_FOLD_DEALER))
    tensor = tensor.at[IDX_PREFLOP_FOLD_FREQ_BB].set(get_freq(FREQ_IDX_PREFLOP_FOLD_BB))
    tensor = tensor.at[IDX_PREFLOP_OPEN_RAISE_SIZE_AVG_DEALER].set(get_avg(AVG_IDX_OPEN_RAISE_SIZE_DEALER, 2.5))
    tensor = tensor.at[IDX_PREFLOP_OPEN_RAISE_SIZE_AVG_BB].set(get_avg(AVG_IDX_OPEN_RAISE_SIZE_BB, 3.0))
    
    # === DISCARD ===
    tensor = tensor.at[IDX_DISCARD_HIGH_CARD_FREQ_DEALER].set(get_freq(FREQ_IDX_DISCARD_HIGH_DEALER))
    tensor = tensor.at[IDX_DISCARD_HIGH_CARD_FREQ_BB].set(get_freq(FREQ_IDX_DISCARD_HIGH_BB))
    
    # === FLOP ===
    tensor = tensor.at[IDX_FLOP_FOLD_FREQ_DEALER].set(get_freq(FREQ_IDX_FLOP_FOLD_DEALER))
    tensor = tensor.at[IDX_FLOP_FOLD_FREQ_BB].set(get_freq(FREQ_IDX_FLOP_FOLD_BB))
    tensor = tensor.at[IDX_FLOP_CALL_FREQ_DEALER].set(get_freq(FREQ_IDX_FLOP_CALL_DEALER))
    tensor = tensor.at[IDX_FLOP_CALL_FREQ_BB].set(get_freq(FREQ_IDX_FLOP_CALL_BB))
    tensor = tensor.at[IDX_FLOP_RAISE_FREQ_DEALER].set(get_freq(FREQ_IDX_FLOP_RAISE_DEALER))
    tensor = tensor.at[IDX_FLOP_RAISE_FREQ_BB].set(get_freq(FREQ_IDX_FLOP_RAISE_BB))
    tensor = tensor.at[IDX_FLOP_BET_FREQ_DEALER].set(get_freq(FREQ_IDX_FLOP_BET_DEALER))
    tensor = tensor.at[IDX_FLOP_BET_FREQ_BB].set(get_freq(FREQ_IDX_FLOP_BET_BB))
    tensor = tensor.at[IDX_FLOP_AVG_BET_SIZE_DEALER].set(get_avg(AVG_IDX_FLOP_BET_SIZE_DEALER, 0.5))
    tensor = tensor.at[IDX_FLOP_AVG_BET_SIZE_BB].set(get_avg(AVG_IDX_FLOP_BET_SIZE_BB, 0.5))
    tensor = tensor.at[IDX_FLOP_CHECK_RAISE_FREQ].set(get_freq(FREQ_IDX_FLOP_CHECK_RAISE))
    tensor = tensor.at[IDX_FLOP_CBET_FREQ_DEALER].set(get_freq(FREQ_IDX_FLOP_CBET_DEALER))
    tensor = tensor.at[IDX_FLOP_CBET_FREQ_BB].set(get_freq(FREQ_IDX_FLOP_CBET_BB))
    tensor = tensor.at[IDX_FLOP_DONK_BET_FREQ_BB].set(get_freq(FREQ_IDX_FLOP_DONK_BB))
    
    # === TURN ===
    tensor = tensor.at[IDX_TURN_FOLD_FREQ_DEALER].set(get_freq(FREQ_IDX_TURN_FOLD_DEALER))
    tensor = tensor.at[IDX_TURN_FOLD_FREQ_BB].set(get_freq(FREQ_IDX_TURN_FOLD_BB))
    tensor = tensor.at[IDX_TURN_CALL_FREQ_DEALER].set(get_freq(FREQ_IDX_TURN_CALL_DEALER))
    tensor = tensor.at[IDX_TURN_CALL_FREQ_BB].set(get_freq(FREQ_IDX_TURN_CALL_BB))
    tensor = tensor.at[IDX_TURN_RAISE_FREQ_DEALER].set(get_freq(FREQ_IDX_TURN_RAISE_DEALER))
    tensor = tensor.at[IDX_TURN_RAISE_FREQ_BB].set(get_freq(FREQ_IDX_TURN_RAISE_BB))
    tensor = tensor.at[IDX_TURN_BET_FREQ_DEALER].set(get_freq(FREQ_IDX_TURN_BET_DEALER))
    tensor = tensor.at[IDX_TURN_BET_FREQ_BB].set(get_freq(FREQ_IDX_TURN_BET_BB))
    tensor = tensor.at[IDX_TURN_AVG_BET_SIZE_DEALER].set(get_avg(AVG_IDX_TURN_BET_SIZE_DEALER, 0.5))
    tensor = tensor.at[IDX_TURN_AVG_BET_SIZE_BB].set(get_avg(AVG_IDX_TURN_BET_SIZE_BB, 0.5))
    tensor = tensor.at[IDX_TURN_DOUBLE_BARREL_FREQ_DEALER].set(get_freq(FREQ_IDX_TURN_DOUBLE_BARREL_DEALER))
    tensor = tensor.at[IDX_TURN_DOUBLE_BARREL_FREQ_BB].set(get_freq(FREQ_IDX_TURN_DOUBLE_BARREL_BB))
    
    # === RIVER ===
    tensor = tensor.at[IDX_RIVER_FOLD_FREQ_DEALER].set(get_freq(FREQ_IDX_RIVER_FOLD_DEALER))
    tensor = tensor.at[IDX_RIVER_FOLD_FREQ_BB].set(get_freq(FREQ_IDX_RIVER_FOLD_BB))
    tensor = tensor.at[IDX_RIVER_CALL_FREQ_DEALER].set(get_freq(FREQ_IDX_RIVER_CALL_DEALER))
    tensor = tensor.at[IDX_RIVER_CALL_FREQ_BB].set(get_freq(FREQ_IDX_RIVER_CALL_BB))
    tensor = tensor.at[IDX_RIVER_RAISE_FREQ_DEALER].set(get_freq(FREQ_IDX_RIVER_RAISE_DEALER))
    tensor = tensor.at[IDX_RIVER_RAISE_FREQ_BB].set(get_freq(FREQ_IDX_RIVER_RAISE_BB))
    tensor = tensor.at[IDX_RIVER_BET_FREQ_DEALER].set(get_freq(FREQ_IDX_RIVER_BET_DEALER))
    tensor = tensor.at[IDX_RIVER_BET_FREQ_BB].set(get_freq(FREQ_IDX_RIVER_BET_BB))
    tensor = tensor.at[IDX_RIVER_AVG_BET_SIZE_DEALER].set(get_avg(AVG_IDX_RIVER_BET_SIZE_DEALER, 0.5))
    tensor = tensor.at[IDX_RIVER_AVG_BET_SIZE_BB].set(get_avg(AVG_IDX_RIVER_BET_SIZE_BB, 0.5))
    tensor = tensor.at[IDX_RIVER_TRIPLE_BARREL_FREQ_DEALER].set(get_freq(FREQ_IDX_RIVER_TRIPLE_BARREL_DEALER))
    tensor = tensor.at[IDX_RIVER_TRIPLE_BARREL_FREQ_BB].set(get_freq(FREQ_IDX_RIVER_TRIPLE_BARREL_BB))
    tensor = tensor.at[IDX_RIVER_OVERBET_FREQ_DEALER].set(get_freq(FREQ_IDX_RIVER_OVERBET_DEALER))
    tensor = tensor.at[IDX_RIVER_OVERBET_FREQ_BB].set(get_freq(FREQ_IDX_RIVER_OVERBET_BB))
    
    # === AGGREGATED ===
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
    
    # reraise_freq - need to add these
    # For now, use placeholder indices
    tensor = tensor.at[IDX_AGG_RERAISE_FREQ_DEALER].set(0.5)  # placeholder
    tensor = tensor.at[IDX_AGG_RERAISE_FREQ_BB].set(0.5)  # placeholder
    
    return tensor