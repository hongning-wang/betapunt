"""Archetype opponents for training diversity.

Rule-based bots that provide diverse opponent styles:
- TIGHT: Folds weak hands, raises/all-in on strong hands
- LAG: Loose-aggressive, raises often with wide range
- PASSIVE: Calls a lot, rarely raises
"""

import jax
import jax.numpy as jnp
import chex

from game import (
    PHASE_PREFLOP,
    PHASE_DISCARD_OOP,
    PHASE_DISCARD_DLR,
    ACT_FOLD,
    ACT_CHECK,
    ACT_CALL,
    ACT_RAISE_MIN,
    ACT_ALLIN,
    ACT_DISCARD_0,
    evaluate_best_hand,
)

# Archetype types
ARCHETYPE_TIGHT = 0   # Only plays premium hands, raises/all-in on strong hands
ARCHETYPE_LAG = 1     # Loose-aggressive: raises often with wide range
ARCHETYPE_PASSIVE = 2 # Calls a lot, rarely raises


@jax.jit
def compute_preflop_strength(hole_cards: chex.Array) -> chex.Array:
    """Compute preflop hand strength from 3 hole cards.

    Returns a strength score 0-100 based on best 2-card combination.
    Premium hands (AA, KK, QQ, JJ, TT, AK, AQ) get high scores.

    Args:
        hole_cards: (3,) int32 card indices

    Returns:
        strength: float32 score 0-100
    """
    # Extract ranks (0-12, where 12=A, 11=K, etc.)
    ranks = hole_cards % 13
    suits = hole_cards // 13

    # Sort ranks descending to find best combination
    sorted_ranks = jnp.sort(ranks)[::-1]  # Descending
    r0, r1, r2 = sorted_ranks[0], sorted_ranks[1], sorted_ranks[2]

    # Check for pairs among all 3 cards
    has_pair_01 = ranks[0] == ranks[1]
    has_pair_02 = ranks[0] == ranks[2]
    has_pair_12 = ranks[1] == ranks[2]
    has_pair = has_pair_01 | has_pair_02 | has_pair_12

    # Find the pair rank if exists
    pair_rank = jnp.where(has_pair_01, ranks[0],
                jnp.where(has_pair_02, ranks[0],
                jnp.where(has_pair_12, ranks[1], jnp.int32(-1))))

    # Check for trips
    has_trips = has_pair_01 & has_pair_02

    # Check for suited combinations
    suited_01 = suits[0] == suits[1]
    suited_02 = suits[0] == suits[2]
    suited_12 = suits[1] == suits[2]

    # Premium pairs: AA=100, KK=95, QQ=90, JJ=85, TT=80, 99=70, 88=65, 77-22=50-30
    # Ace = rank 12, King = rank 11, etc.
    pair_strength = jnp.where(
        has_trips,
        95.0 + pair_rank * 0.3,  # Trips are very strong
        jnp.where(
            has_pair,
            jnp.where(pair_rank >= 12, 100.0,  # AA
            jnp.where(pair_rank >= 11, 95.0,   # KK
            jnp.where(pair_rank >= 10, 90.0,   # QQ
            jnp.where(pair_rank >= 9, 85.0,    # JJ
            jnp.where(pair_rank >= 8, 80.0,    # TT
            jnp.where(pair_rank >= 7, 70.0,    # 99
            jnp.where(pair_rank >= 6, 65.0,    # 88
            jnp.where(pair_rank >= 5, 55.0,    # 77
            jnp.where(pair_rank >= 4, 50.0,    # 66
            jnp.where(pair_rank >= 3, 45.0,    # 55
            jnp.where(pair_rank >= 2, 40.0,    # 44
            jnp.where(pair_rank >= 1, 35.0,    # 33
            30.0)))))))))))),                   # 22
            0.0
        )
    )

    # High card combinations (when no pair)
    # Best case: AK suited = 82, AK offsuit = 78
    # AQ suited = 75, AQ offsuit = 70
    # AJ suited = 68, KQ suited = 65
    has_ace = r0 >= 12
    has_king = (r0 >= 11) | (r1 >= 11)
    has_queen = (r0 >= 10) | (r1 >= 10) | (r2 >= 10)

    # Check if top 2 cards are suited
    top2_suited = jnp.where(
        (r0 == ranks[0]) & (r1 == ranks[1]), suited_01,
        jnp.where((r0 == ranks[0]) & (r1 == ranks[2]), suited_02,
        jnp.where((r0 == ranks[1]) & (r1 == ranks[0]), suited_01,
        jnp.where((r0 == ranks[1]) & (r1 == ranks[2]), suited_12,
        jnp.where((r0 == ranks[2]) & (r1 == ranks[0]), suited_02,
        suited_12)))))
    

    suited_bonus = jnp.where(top2_suited, 4.0, 0.0)

    # AK = 78+, AQ = 70+, AJ = 65+, AT = 60+
    # KQ = 62+, KJ = 58+, QJ = 55+
    high_card_strength = jnp.where(
        has_ace & (r1 >= 11),  # AK
        78.0 + suited_bonus,
        jnp.where(has_ace & (r1 >= 10),  # AQ
        70.0 + suited_bonus,
        jnp.where(has_ace & (r1 >= 9),   # AJ
        65.0 + suited_bonus,
        jnp.where(has_ace & (r1 >= 8),   # AT
        60.0 + suited_bonus,
        jnp.where(has_ace & (r1 >= 7),   # A9
        52.0 + suited_bonus,
        jnp.where((r0 >= 11) & (r1 >= 10),  # KQ
        62.0 + suited_bonus,
        jnp.where((r0 >= 11) & (r1 >= 9),   # KJ
        58.0 + suited_bonus,
        jnp.where((r0 >= 10) & (r1 >= 9),   # QJ
        55.0 + suited_bonus,
        jnp.where((r0 >= 11) & (r1 >= 8),   # KT
        54.0 + suited_bonus,
        jnp.where((r0 >= 10) & (r1 >= 8),   # QT
        50.0 + suited_bonus,
        # Connected cards with suited bonus
        jnp.where(jnp.abs(r0 - r1) == 1,
        40.0 + suited_bonus + r0 * 0.5,
        # Other hands
        25.0 + r0 * 1.5 + r1 * 0.5 + suited_bonus
        )))))))))))
    

    # Return pair strength if has pair, otherwise high card strength
    return jnp.where(has_pair, pair_strength, high_card_strength)


@jax.jit
def compute_postflop_strength(hole_cards: chex.Array, active_mask: chex.Array,
                               board: chex.Array, phase: chex.Array) -> chex.Array:
    """Compute postflop hand strength using the hand evaluator.

    Args:
        hole_cards: (3,) int32 hole cards
        active_mask: (3,) bool mask of active hole cards (2 True after discard)
        board: (6,) int32 board cards
        phase: current game phase

    Returns:
        strength: float32 score 0-100 based on hand rank
    """
    # Determine how many board cards are revealed based on phase
    # PHASE_FLOP=3, PHASE_TURN=4, PHASE_RIVER=5
    num_board = jnp.where(phase >= 5, 5,  # River: 5 cards
               jnp.where(phase >= 4, 4,   # Turn: 4 cards
               jnp.where(phase >= 3, 3,   # Flop: 3 cards
               0)))

    # Create board mask (only revealed cards)
    board_indices = jnp.arange(6)
    board_mask = board_indices < num_board

    # Get active hole cards (after discard, 2 are active)
    # Use top_k to get indices of active cards
    _, active_indices = jax.lax.top_k(active_mask.astype(jnp.int32), k=2)
    active_hole = hole_cards[active_indices]

    # Masked board (use -1 for unrevealed cards - evaluator handles this)
    masked_board = jnp.where(board_mask, board, jnp.int32(-1))

    # Evaluate hand
    score = evaluate_best_hand(active_hole, masked_board)

    # Extract tier (top bits) - tier 0-8
    # 0=high card, 1=pair, 2=two pair, 3=trips, 4=straight, 5=flush, 6=full house, 7=quads, 8=straight flush
    tier = (score >> 20) & 0xF

    # Convert tier to strength (0-100)
    # High card: 10-25, Pair: 25-45, Two pair: 45-60, Trips: 60-75,
    # Straight: 75-80, Flush: 80-85, Full house: 85-92, Quads: 92-97, SF: 97-100
    strength = jnp.where(tier >= 8, 97.0 + (score & 0xFFFFF) / 1048576.0 * 3.0,   # SF
              jnp.where(tier >= 7, 92.0 + (score & 0xFFFFF) / 1048576.0 * 5.0,    # Quads
              jnp.where(tier >= 6, 85.0 + (score & 0xFFFFF) / 1048576.0 * 7.0,    # Full house
              jnp.where(tier >= 5, 80.0 + (score & 0xFFFFF) / 1048576.0 * 5.0,    # Flush
              jnp.where(tier >= 4, 75.0 + (score & 0xFFFFF) / 1048576.0 * 5.0,    # Straight
              jnp.where(tier >= 3, 60.0 + (score & 0xFFFFF) / 1048576.0 * 15.0,   # Trips
              jnp.where(tier >= 2, 45.0 + (score & 0xFFFFF) / 1048576.0 * 15.0,   # Two pair
              jnp.where(tier >= 1, 25.0 + (score & 0xFFFFF) / 1048576.0 * 20.0,   # Pair
              10.0 + (score & 0xFFFFF) / 1048576.0 * 15.0                          # High card
              ))))))))

    return strength


@jax.jit
def tight_get_logits(legal_mask: chex.Array, strength: chex.Array, is_preflop: chex.Array) -> chex.Array:
    """Get action logits for TIGHT archetype.

    TIGHT player behavior:
    - Preflop: Only plays premium hands (strength >= 75), folds otherwise
    - Postflop: Raises/bets with strong hands, checks weak hands, folds to aggression with weak hands

    Args:
        legal_mask: (13,) bool mask of legal actions
        strength: float32 hand strength 0-100
        is_preflop: bool whether we're preflop

    Returns:
        logits: (13,) float32 action logits
    """
    # Actions: [Fold, Check, Call, MinR, 33%, 50%, 75%, 125%, 200%, AllIn, Disc0, Disc1, Disc2]

    # Thresholds - tight player has high standards
    premium_threshold = 75.0  # Preflop premium hands
    strong_threshold = 70.0   # Postflop strong hands (two pair+)
    medium_threshold = 50.0   # Postflop medium hands (top pair+)

    # Can we check? (no bet to call)
    can_check = legal_mask[ACT_CHECK]
    can_fold = legal_mask[ACT_FOLD]

    # Base logits (will be masked by legal_mask)
    base = jnp.zeros(13, dtype=jnp.float32)

    # PREFLOP logic
    preflop_logits = jnp.where(
        strength >= premium_threshold,
        # Strong hand: raise big or all-in
        jnp.array([
            -5.0,   # Fold - never
            -2.0,   # Check - rarely
            0.0,    # Call - sometimes
            1.0,    # Min raise
            1.5,    # 33%
            2.0,    # 50%
            2.5,    # 75%
            3.0,    # 125%
            3.5,    # 200%
            4.0,    # All-in - love it
            0.0, 0.0, 0.0,  # Discard - neutral
        ], dtype=jnp.float32),
        jnp.where(
            strength >= 60.0,
            # Decent hand: call or small raise
            jnp.array([
                -1.0,   # Fold - reluctant
                2.0,    # Check - prefer
                1.0,    # Call - ok
                0.5,    # Min raise - sometimes
                0.0, -0.5, -1.0, -1.5, -2.0,  # Larger raises - avoid
                -3.0,   # All-in - no
                0.0, 0.0, 0.0,
            ], dtype=jnp.float32),
            # Weak hand: fold or check
            jnp.array([
                3.0,    # Fold - strong preference
                4.0,    # Check - if possible
                -2.0,   # Call - avoid
                -4.0, -4.0, -4.0, -4.0, -4.0, -4.0,  # Raises - never
                -5.0,   # All-in - never
                0.0, 0.0, 0.0,
            ], dtype=jnp.float32)
        )
    )

    # POSTFLOP logic
    postflop_logits = jnp.where(
        strength >= strong_threshold,
        # Strong hand (two pair+): bet/raise aggressively
        jnp.array([
            -5.0,   # Fold - never
            -1.0,   # Check - rarely (slowplay sometimes)
            0.5,    # Call - ok but prefer raise
            1.5,    # Min raise
            2.0,    # 33%
            2.5,    # 50%
            3.0,    # 75%
            3.5,    # 125%
            3.5,    # 200%
            4.0,    # All-in
            0.0, 0.0, 0.0,
        ], dtype=jnp.float32),
        jnp.where(
            strength >= medium_threshold,
            # Medium hand (top pair): bet small or check/call
            jnp.array([
                -1.0,   # Fold - reluctant
                2.0,    # Check - often
                1.5,    # Call - ok
                1.0,    # Min raise - sometimes
                0.5,    # 33%
                0.0,    # 50%
                -1.0, -2.0, -2.5,  # Larger raises - avoid
                -3.0,   # All-in - no
                0.0, 0.0, 0.0,
            ], dtype=jnp.float32),
            # Weak hand: check or fold
            jnp.array([
                3.0,    # Fold - yes to bets
                4.0,    # Check - if possible
                -1.0,   # Call - rarely
                -3.0, -3.0, -3.0, -3.0, -3.0, -3.0,  # Raises - never
                -5.0,   # All-in - never
                0.0, 0.0, 0.0,
            ], dtype=jnp.float32)
        )
    )

    # Select based on phase
    logits = jnp.where(is_preflop, preflop_logits, postflop_logits)

    return logits


@jax.jit
def lag_get_logits(legal_mask: chex.Array, strength: chex.Array, is_preflop: chex.Array) -> chex.Array:
    """Get action logits for LAG (Loose-Aggressive) archetype.

    LAG player behavior:
    - Raises frequently with a wide range
    - Bluffs often
    - Puts pressure on opponents
    """
    # LAG raises with everything, strength just affects sizing
    base_aggression = jnp.array([
        0.5,    # Fold - sometimes give up
        -0.5,   # Check - rarely
        0.0,    # Call - not preferred
        1.5,    # Min raise
        2.0,    # 33%
        2.5,    # 50%
        2.0,    # 75%
        1.5,    # 125%
        1.0,    # 200%
        0.8,    # All-in
        0.0, 0.0, 0.0,
    ], dtype=jnp.float32)

    # Adjust based on strength - stronger hands bet bigger
    strength_bonus = (strength - 50.0) / 50.0  # -1 to +1

    # Increase big bet preference with strong hands
    size_adjustment = jnp.array([
        -strength_bonus * 2.0,  # Less likely to fold with strength
        -strength_bonus,        # Less likely to check
        0.0,
        0.0,
        0.0,
        strength_bonus * 0.5,
        strength_bonus * 1.0,
        strength_bonus * 1.5,
        strength_bonus * 2.0,
        strength_bonus * 2.5,   # More likely to go all-in with strength
        0.0, 0.0, 0.0,
    ], dtype=jnp.float32)

    return base_aggression + size_adjustment


@jax.jit
def passive_get_logits(legal_mask: chex.Array, strength: chex.Array, is_preflop: chex.Array) -> chex.Array:
    """Get action logits for PASSIVE archetype.

    PASSIVE player behavior:
    - Calls a lot, rarely raises
    - Only raises with very strong hands
    - Rarely bluffs
    """
    # Strong hand threshold for passive player to raise
    raise_threshold = 80.0

    passive_logits = jnp.where(
        strength >= raise_threshold,
        # Very strong hand - even passive player raises
        jnp.array([
            -3.0,   # Fold - no
            1.0,    # Check - sometimes slowplay
            1.5,    # Call - often
            2.0,    # Min raise
            1.5,    # 33%
            1.0,    # 50%
            0.5,    # 75%
            0.0,    # 125%
            -0.5,   # 200%
            0.5,    # All-in - sometimes
            0.0, 0.0, 0.0,
        ], dtype=jnp.float32),
        jnp.where(
            strength >= 40.0,
            # Medium hand - call/check
            jnp.array([
                0.5,    # Fold - sometimes
                3.0,    # Check - prefer
                2.5,    # Call - very happy to call
                0.0,    # Min raise - rarely
                -1.0, -1.5, -2.0, -2.5, -3.0,
                -4.0,   # All-in - never
                0.0, 0.0, 0.0,
            ], dtype=jnp.float32),
            # Weak hand - fold or check
            jnp.array([
                2.0,    # Fold - yes
                3.0,    # Check - if possible
                0.5,    # Call - sometimes call too much (calling station)
                -2.0, -2.0, -2.0, -2.0, -2.0, -2.0,
                -4.0,
                0.0, 0.0, 0.0,
            ], dtype=jnp.float32)
        )
    )

    return passive_logits


@jax.jit
def archetype_get_logits(
    archetype_type: chex.Array,
    legal_mask: chex.Array,
    hole_cards: chex.Array,
    active_mask: chex.Array,
    board: chex.Array,
    phase: chex.Array,
) -> chex.Array:
    """Get action logits for archetype policy based on hand strength.

    Args:
        archetype_type: int32 scalar (0=TIGHT, 1=LAG, 2=PASSIVE)
        legal_mask: (13,) bool mask of legal actions
        hole_cards: (3,) int32 hole cards
        active_mask: (3,) bool mask of active hole cards
        board: (6,) int32 board cards
        phase: int32 current game phase

    Returns:
        logits: (13,) float32 action logits
    """
    # Determine if preflop (including discard phases)
    is_preflop = phase <= PHASE_DISCARD_DLR

    # Compute hand strength
    preflop_strength = compute_preflop_strength(hole_cards)
    postflop_strength = compute_postflop_strength(hole_cards, active_mask, board, phase)
    strength = jnp.where(is_preflop, preflop_strength, postflop_strength)

    # Get archetype-specific logits
    logits = jax.lax.switch(
        archetype_type,
        [
            lambda: tight_get_logits(legal_mask, strength, is_preflop),
            lambda: lag_get_logits(legal_mask, strength, is_preflop),
            lambda: passive_get_logits(legal_mask, strength, is_preflop),
        ],
    )

    # Apply legal mask
    neg_inf = jnp.float32(-1e9)
    masked_logits = jnp.where(legal_mask, logits, neg_inf)

    return masked_logits
