"""
Two Plus Two Poker Hand Evaluator - Pure JAX Implementation

This is the REAL Two Plus Two algorithm using the ~130MB precomputed lookup table.
Hand evaluation is O(1) - just 5 table lookups for a 5-card hand.

Card Encoding (YOUR format):
    - Suits: clubs=0, diamonds=1, hearts=2, spades=3
    - Ranks: 2=0, 3=1, 4=2, ..., K=11, A=12
    - Card index = suit * 13 + rank
    - So: 2c=0, Ac=12, 2d=13, Ad=25, ..., As=51

The lookup table uses a different encoding internally (1-52, rank-major),
but this is handled transparently via a conversion table.

Usage:
    # Load the lookup table (do this once)
    HR = load_hand_ranks_table("HandRanks.dat")
    
    # Evaluate hands (JAX JIT compatible)
    hand = jnp.array([12, 11, 10, 9, 8])  # Ac Kc Qc Jc Tc
    rank = evaluate_5_cards(hand, HR)      # Returns hand value
    
    # Batch evaluation
    hands = jnp.array([[12, 11, 10, 9, 8], [0, 1, 2, 3, 4]])
    ranks = evaluate_5_cards_batch(hands, HR)
"""

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from pathlib import Path


# =============================================================================
# Card Encoding Conversion
# =============================================================================

def _generate_card_conversion_table() -> jnp.ndarray:
    """
    Generate conversion table from your encoding to Two Plus Two encoding.
    
    Your encoding: suit * 13 + rank (0-51)
        - suits: c=0, d=1, h=2, s=3
        - ranks: 2=0, 3=1, ..., A=12
        
    Two Plus Two encoding: rank * 4 + suit + 1 (1-52)
        - ranks: 2=0, 3=1, ..., A=12 (same)
        - suits: c=0, d=1, h=2, s=3 (same)
        - But stored as: 2c=1, 2d=2, 2h=3, 2s=4, 3c=5, ...
    """
    conversion = np.zeros(52, dtype=np.int32)
    for your_card in range(52):
        suit = your_card // 13  # 0-3
        rank = your_card % 13   # 0-12
        tpt_card = rank * 4 + suit + 1  # 1-52
        conversion[your_card] = tpt_card
    return jnp.array(conversion)


# Pre-compute conversion table
CARD_CONVERSION = _generate_card_conversion_table()


# =============================================================================
# Lookup Table Loading
# =============================================================================

def load_hand_ranks_table(filepath: str = "HandRanks.dat") -> jnp.ndarray:
    """
    Load the Two Plus Two HandRanks.dat lookup table.
    
    The table is ~130MB containing 32,487,834 32-bit integers.
    
    Args:
        filepath: Path to HandRanks.dat file
        
    Returns:
        JAX array containing the lookup table
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(
            f"HandRanks.dat not found at {filepath}. "
            "Download from a poker-evaluator package or generate it."
        )
    
    # Read as numpy first (faster), then convert to JAX
    hr = np.fromfile(filepath, dtype=np.int32)
    print(f"Loaded HandRanks.dat: {len(hr):,} entries ({hr.nbytes / 1024 / 1024:.1f} MB)")
    
    return jnp.array(hr)


def generate_hand_ranks_table() -> jnp.ndarray:
    """
    Generate the Two Plus Two lookup table from scratch.
    
    This is slow (~minutes) and only needs to be done once.
    The generated table can be saved with save_hand_ranks_table().
    """
    raise NotImplementedError(
        "Table generation not implemented. "
        "Please use a pre-generated HandRanks.dat file."
    )


def save_hand_ranks_table(hr: jnp.ndarray, filepath: str = "HandRanks.dat"):
    """Save the lookup table to a file."""
    np.array(hr, dtype=np.int32).tofile(filepath)


# =============================================================================
# Core Evaluation Functions (JAX JIT compatible)
# =============================================================================

@partial(jax.jit, static_argnums=())
def evaluate_5_cards(cards: jnp.ndarray, hr: jnp.ndarray) -> jnp.ndarray:
    """
    Evaluate a 5-card poker hand using the Two Plus Two lookup table.
    
    This is O(1) - just 6 table lookups total.
    
    Args:
        cards: Array of shape (5,) with card indices 0-51 (your encoding)
        hr: The HandRanks lookup table
        
    Returns:
        Hand value (scalar). Higher is better.
        Use get_hand_type() and get_hand_rank() to extract components.
    """
    # Convert from your encoding to Two Plus Two encoding
    c = CARD_CONVERSION[cards]
    
    # Chain lookups starting from index 53
    p = hr[53 + c[0]]
    p = hr[p + c[1]]
    p = hr[p + c[2]]
    p = hr[p + c[3]]
    p = hr[p + c[4]]
    
    # Extra lookup for 5-card hands
    p = hr[p]
    
    return p


@partial(jax.jit, static_argnums=())
def evaluate_6_cards(cards: jnp.ndarray, hr: jnp.ndarray) -> jnp.ndarray:
    """
    Evaluate a 6-card poker hand.
    
    Args:
        cards: Array of shape (6,) with card indices 0-51
        hr: The HandRanks lookup table
        
    Returns:
        Hand value (scalar). Higher is better.
    """
    c = CARD_CONVERSION[cards]
    
    p = hr[53 + c[0]]
    p = hr[p + c[1]]
    p = hr[p + c[2]]
    p = hr[p + c[3]]
    p = hr[p + c[4]]
    p = hr[p + c[5]]
    
    # Extra lookup for 6-card hands
    p = hr[p]
    
    return p


@partial(jax.jit, static_argnums=())
def evaluate_7_cards(cards: jnp.ndarray, hr: jnp.ndarray) -> jnp.ndarray:
    """
    Evaluate a 7-card poker hand.
    
    Args:
        cards: Array of shape (7,) with card indices 0-51
        hr: The HandRanks lookup table
        
    Returns:
        Hand value (scalar). Higher is better.
    """
    c = CARD_CONVERSION[cards]
    
    p = hr[53 + c[0]]
    p = hr[p + c[1]]
    p = hr[p + c[2]]
    p = hr[p + c[3]]
    p = hr[p + c[4]]
    p = hr[p + c[5]]
    p = hr[p + c[6]]
    
    # No extra lookup needed for 7-card hands
    return p


# =============================================================================
# Batched Evaluation (JAX JIT + vmap)
# =============================================================================

@jax.jit
def evaluate_5_cards_batch(hands: jnp.ndarray, hr: jnp.ndarray) -> jnp.ndarray:
    """
    Evaluate multiple 5-card hands.
    
    Args:
        hands: Array of shape (n_hands, 5) with card indices 0-51
        hr: The HandRanks lookup table
        
    Returns:
        Array of shape (n_hands,) with hand values
    """
    return jax.vmap(lambda h: evaluate_5_cards(h, hr))(hands)


@jax.jit
def evaluate_6_cards_batch(hands: jnp.ndarray, hr: jnp.ndarray) -> jnp.ndarray:
    """Evaluate multiple 6-card hands."""
    return jax.vmap(lambda h: evaluate_6_cards(h, hr))(hands)


@jax.jit
def evaluate_7_cards_batch(hands: jnp.ndarray, hr: jnp.ndarray) -> jnp.ndarray:
    """Evaluate multiple 7-card hands."""
    return jax.vmap(lambda h: evaluate_7_cards(h, hr))(hands)


# =============================================================================
# Hand Value Utilities
# =============================================================================

HAND_TYPES = [
    'invalid',
    'high card',
    'one pair', 
    'two pair',
    'three of a kind',
    'straight',
    'flush',
    'full house',
    'four of a kind',
    'straight flush'
]


def get_hand_type(value: int) -> int:
    """
    Extract hand type from hand value.
    
    Returns:
        0: invalid
        1: high card
        2: one pair
        3: two pair
        4: three of a kind
        5: straight
        6: flush
        7: full house
        8: four of a kind
        9: straight flush
    """
    return int(value) >> 12


def get_hand_rank(value: int) -> int:
    """
    Extract hand rank within type from hand value.
    
    Lower rank within same type = better hand.
    """
    return int(value) & 0x0FFF


def get_hand_name(value: int) -> str:
    """Get human-readable hand name."""
    return HAND_TYPES[get_hand_type(value)]


# =============================================================================
# Card Utilities
# =============================================================================

RANKS = "23456789TJQKA"
SUITS = "cdhs"


def card_to_string(card: int) -> str:
    """Convert card index (0-51) to string like 'Ac'."""
    rank = int(card) % 13
    suit = int(card) // 13
    return RANKS[rank] + SUITS[suit]


def string_to_card(s: str) -> int:
    """Convert string like 'Ac' to card index (0-51)."""
    rank = RANKS.index(s[0].upper())
    suit = SUITS.index(s[1].lower())
    return suit * 13 + rank


def hand_to_string(cards) -> str:
    """Convert array of card indices to string."""
    return " ".join(card_to_string(int(c)) for c in cards)


def string_to_hand(s: str) -> jnp.ndarray:
    """Convert string like 'Ac Kc Qc Jc Tc' to card array."""
    return jnp.array([string_to_card(c) for c in s.split()], dtype=jnp.int32)


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Demonstrate the Two Plus Two evaluator."""
    import time
    
    print("=" * 60)
    print("Two Plus Two Poker Evaluator - Real Lookup Table")
    print("=" * 60)
    
    # Load the lookup table
    hr = load_hand_ranks_table("HandRanks.dat")
    
    # Test hands
    test_hands = [
        "Ac Kc Qc Jc Tc",  # Royal flush
        "9h 8h 7h 6h 5h",  # Straight flush
        "Ks Kh Kd Kc 3s",  # Four of a kind
        "Qd Qh Qs 7c 7d",  # Full house
        "2d 5d 8d Jd Ad",  # Flush
        "Th 9c 8d 7s 6h",  # Straight
        "5c 5d 5h 9s Kc",  # Three of a kind
        "Jh Jc 4s 4d As",  # Two pair
        "8h 8c Ks Qd 3c",  # One pair
        "Ac Kd Qh Js 9c",  # High card
    ]
    
    print("\n5-Card Hand Evaluation:")
    print("-" * 60)
    
    for hand_str in test_hands:
        cards = string_to_hand(hand_str)
        value = evaluate_5_cards(cards, hr)
        hand_type = get_hand_type(value)
        hand_rank = get_hand_rank(value)
        print(f"{hand_str:20} -> value={int(value):5}, type={hand_type}, rank={hand_rank:4}, {HAND_TYPES[hand_type]}")
    
    print("\nRaw Index Input (your format):")
    print("-" * 60)
    
    # Royal flush: Ac(12), Kc(11), Qc(10), Jc(9), Tc(8)
    royal = jnp.array([12, 11, 10, 9, 8], dtype=jnp.int32)
    value = evaluate_5_cards(royal, hr)
    print(f"jnp.array([12, 11, 10, 9, 8]) -> value={int(value)}, {get_hand_name(value)}")
    
    # Four aces: Ac(12), Ad(25), Ah(38), As(51), Kc(11)
    quads = jnp.array([12, 25, 38, 51, 11], dtype=jnp.int32)
    value = evaluate_5_cards(quads, hr)
    print(f"jnp.array([12, 25, 38, 51, 11]) -> value={int(value)}, {get_hand_name(value)}")
    
    # Performance test
    print("\nPerformance Test:")
    print("-" * 60)
    
    key = jax.random.PRNGKey(42)
    n_hands = 100_000
    
    # Generate random 5-card hands more efficiently
    key, subkey = jax.random.split(key)
    # Generate indices and reshape - not perfect sampling but fast for demo
    rand_cards = jax.random.randint(subkey, (n_hands, 5), 0, 52)
    hands_5 = rand_cards.astype(jnp.int32)
    
    # Warmup
    _ = evaluate_5_cards_batch(hands_5[:1000], hr).block_until_ready()
    
    start = time.time()
    results = evaluate_5_cards_batch(hands_5, hr)
    results.block_until_ready()
    elapsed = time.time() - start
    
    print(f"Evaluated {n_hands:,} 5-card hands in {elapsed:.3f}s")
    print(f"Rate: {n_hands/elapsed:,.0f} hands/second")
    
    # 7-card hands
    n_7 = 100_000
    key, subkey = jax.random.split(key)
    hands_7 = jax.random.randint(subkey, (n_7, 7), 0, 52).astype(jnp.int32)
    
    _ = evaluate_7_cards_batch(hands_7[:1000], hr).block_until_ready()
    
    start = time.time()
    results = evaluate_7_cards_batch(hands_7, hr)
    results.block_until_ready()
    elapsed = time.time() - start
    
    print(f"Evaluated {n_7:,} 7-card hands in {elapsed:.3f}s")
    print(f"Rate: {n_7/elapsed:,.0f} hands/second")
    
    print("\n" + "=" * 60)
    print("Demo complete!")


if __name__ == "__main__":
    demo()
