from __future__ import annotations

import random
from typing import List, Tuple

import jax.numpy as jnp
import pkrbot

from game import evaluate_best_hand


_RANKS = "23456789TJQKA"
_SUITS = "cdhs" 


def _idx_to_card_str(idx: int) -> str:
    rank = idx % 13
    suit = idx // 13
    return _RANKS[rank] + _SUITS[suit]


def _cmp(a: int, b: int) -> int:
    return (a > b) - (a < b)


def test_evaluate_best_hand_against_engine_evaluator(
    num_trials: int = 10_000,
    seed: int = 0,
    max_failures_to_show: int = 5,
) -> None:
    """
    Stress-test `training.game.evaluate_best_hand` against the engine's evaluator.

    In `engine.py`, showdown uses:
        pkrbot.evaluate(board + hand)
    and compares the resulting scores directly (higher is better).

    This test compares only the *outcome* (win/loss/tie) for two players given the
    same 6-card board and 2-card holes, which is robust even if the two evaluators
    use different absolute scoring scales.
    """

    rng = random.Random(seed)

    # Pre-build all 52 card objects once for speed.
    cards: List[pkrbot.Card] = [pkrbot.Card(_idx_to_card_str(i)) for i in range(52)]

    # Warm-up JIT compile (evaluate_best_hand is jitted).
    _ = evaluate_best_hand(
        jnp.array([0, 1], dtype=jnp.int32),
        jnp.array([2, 3, 4, 5, 6, 7], dtype=jnp.int32),
    ).item()

    failures: List[Tuple[List[int], List[int], List[int], int, int]] = []

    for _t in range(num_trials):
        deal = rng.sample(range(52), 10)
        board = deal[0:6]
        p0 = deal[6:8]
        p1 = deal[8:10]

        # Engine evaluator (pkrbot) — use 8 cards: 6 board + 2 hole.
        e0 = pkrbot.evaluate([cards[i] for i in (board + p0)])
        e1 = pkrbot.evaluate([cards[i] for i in (board + p1)])
        engine_outcome = _cmp(e0, e1)

        # JAX evaluator — expects ints 0..51.
        j0 = int(
            evaluate_best_hand(
                jnp.array(p0, dtype=jnp.int32),
                jnp.array(board, dtype=jnp.int32),
            ).item()
        )
        j1 = int(
            evaluate_best_hand(
                jnp.array(p1, dtype=jnp.int32),
                jnp.array(board, dtype=jnp.int32),
            ).item()
        )
        jax_outcome = _cmp(j0, j1)

        if engine_outcome != jax_outcome:
            failures.append((board, p0, p1, engine_outcome, jax_outcome))
            if len(failures) >= max_failures_to_show:
                break

    if failures:
        lines = [
            f"Found {len(failures)} mismatch(es) (showing up to {max_failures_to_show}).",
            "Format: board | p0 | p1 | engine_outcome | jax_outcome",
        ]
        for board, p0, p1, eo, jo in failures:
            lines.append(
                f"{[_idx_to_card_str(i) for i in board]} | "
                f"{[_idx_to_card_str(i) for i in p0]} | "
                f"{[_idx_to_card_str(i) for i in p1]} | "
                f"{eo:+d} | {jo:+d}"
            )
        raise AssertionError("\n".join(lines))


if __name__ == "__main__":
    # Run a quick local check without a test runner.
    test_evaluate_best_hand_against_engine_evaluator(num_trials=100_000, seed=0)
    print("OK: evaluate_best_hand matches engine evaluator on randomized trials.")
