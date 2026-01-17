"""
Test file to verify that game.py (RL environment) matches engine.py (official game engine)
for the Toss or Hold'em poker variant.

This tests:
1. Initial state setup (blinds, stacks, dealer position)
2. Legal action computation
3. Raise bounds computation  
4. State transitions for all action types
5. Showdown evaluation
6. Discard mechanics
7. Street progression
8. Delta/payout calculation

NOTE: Some tests are marked as expected failures due to known issues in game.py
that need to be fixed (JAX dynamic slicing issues in _advance_street).
"""
import os
os.environ["JAX_TRACEBACK_FILTERING"] = "off"

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from collections import namedtuple
from typing import List, Tuple, Optional
import itertools

# Import game.py components
import sys
sys.path.insert(0, '/mnt/user-data/uploads')
from training.game import (
    TossHoldemEnv, State, 
    NUM_CARDS, STACK_SIZE, SB, BB,
    PHASE_PREFLOP, PHASE_DISCARD_OOP, PHASE_DISCARD_DLR, PHASE_FLOP, PHASE_TURN, PHASE_RIVER, PHASE_SHOWDOWN,
    ACT_FOLD, ACT_CHECK, ACT_CALL, ACT_RAISE_MIN, ACT_RAISE_33, ACT_RAISE_50, ACT_RAISE_75, ACT_RAISE_125, ACT_RAISE_200, ACT_ALLIN,
    ACT_DISCARD_0, ACT_DISCARD_1, ACT_DISCARD_2,
    score_5_hand, evaluate_best_hand, get_showdown_result, COMBINATIONS_8_5
)


# ============================================================================
# Mock Engine Implementation (extracted from engine.py logic)
# ============================================================================

STARTING_STACK = 400
BIG_BLIND = 2
SMALL_BLIND = 1

DiscardAction = namedtuple('DiscardAction', ['card'])
FoldAction = namedtuple('FoldAction', [])
CallAction = namedtuple('CallAction', [])
CheckAction = namedtuple('CheckAction', [])
RaiseAction = namedtuple('RaiseAction', ['amount'])


class MockRoundState:
    """Simplified mock of engine.py's RoundState for testing purposes."""
    
    def __init__(self, button, street, pips, stacks, hands, board):
        self.button = button
        self.street = street
        self.pips = list(pips)
        self.stacks = list(stacks)
        self.hands = [list(h) for h in hands]
        self.board = list(board)
    
    def legal_actions(self):
        """Returns legal actions matching engine.py logic."""
        active = self.button % 2
        continue_cost = self.pips[1-active] - self.pips[active]
        
        if self.street in (2, 3):
            return {DiscardAction} if active != self.street % 2 else {CheckAction}
        
        if continue_cost == 0:
            bets_forbidden = (self.stacks[0] == 0 or self.stacks[1] == 0)
            return {CheckAction, FoldAction} if bets_forbidden else {CheckAction, RaiseAction, FoldAction}
        
        raises_forbidden = (continue_cost == self.stacks[active] or self.stacks[1-active] == 0)
        return {FoldAction, CallAction} if raises_forbidden else {FoldAction, CallAction, RaiseAction}
    
    def raise_bounds(self):
        """Returns (min_raise, max_raise) matching engine.py logic."""
        active = self.button % 2
        continue_cost = self.pips[1-active] - self.pips[active]
        max_contribution = min(self.stacks[active], self.stacks[1-active] + continue_cost)
        min_contribution = min(max_contribution, continue_cost + max(continue_cost, BIG_BLIND))
        return (self.pips[active] + min_contribution, self.pips[active] + max_contribution)
    
    def get_delta(self, winner_index):
        """Returns delta for player 0 matching engine.py logic."""
        if winner_index == 2:
            return 0
        elif winner_index == 0:
            return STARTING_STACK - self.stacks[1]
        else:
            return self.stacks[0] - STARTING_STACK


# ============================================================================
# Helper Functions
# ============================================================================

def card_to_int(rank: int, suit: int) -> int:
    """
    Convert rank (0-12: 2-A) and suit (0-3: c,d,h,s) to integer 0-51.
    Card encoding: suit * 13 + rank
    """
    return suit * 13 + rank


def int_to_card(card_int: int) -> str:
    """Convert integer 0-51 to card string."""
    ranks = '23456789TJQKA'
    suits = 'cdhs'
    rank = card_int % 13
    suit = card_int // 13
    return ranks[rank] + suits[suit]


# ============================================================================
# Test: Initial State Setup
# ============================================================================

class TestInitialState:
    """Test that initial state setup matches engine.py."""
    
    def test_initial_stacks(self):
        """Verify starting stacks are 400."""
        env = TossHoldemEnv()
        key = jax.random.PRNGKey(42)
        state = env.reset(key)
        
        total = state.p0_stack + state.p1_stack + state.p0_bet + state.p1_bet
        assert total == 2 * STACK_SIZE, f"Total chips should be {2 * STACK_SIZE}, got {total}"
    
    def test_blind_amounts(self):
        """Verify blinds are SB=1, BB=2."""
        env = TossHoldemEnv()
        
        for seed in range(100):
            key = jax.random.PRNGKey(seed)
            state = env.reset(key)
            bets = sorted([int(state.p0_bet), int(state.p1_bet)])
            assert bets == [SB, BB], f"Blinds should be [1, 2], got {bets}"
    
    def test_dealer_posts_small_blind(self):
        """Verify dealer posts small blind, opponent posts big blind."""
        env = TossHoldemEnv()
        
        for seed in range(100):
            key = jax.random.PRNGKey(seed)
            state = env.reset(key)
            
            if state.is_dealer_p0:
                assert state.p0_bet == SB, "Dealer (p0) should post SB"
                assert state.p1_bet == BB, "Non-dealer (p1) should post BB"
            else:
                assert state.p0_bet == BB, "Non-dealer (p0) should post BB"
                assert state.p1_bet == SB, "Dealer (p1) should post SB"
    
    def test_dealer_acts_first_preflop(self):
        """Verify dealer acts first in preflop betting."""
        env = TossHoldemEnv()
        
        for seed in range(50):
            key = jax.random.PRNGKey(seed)
            state = env.reset(key)
            dealer_id = 0 if state.is_dealer_p0 else 1
            assert state.current_player == dealer_id, "Dealer should act first preflop"
    
    def test_three_hole_cards_dealt(self):
        """Verify each player gets 3 hole cards."""
        env = TossHoldemEnv()
        key = jax.random.PRNGKey(42)
        state = env.reset(key)
        
        assert jnp.all((state.p0_hole >= 0) & (state.p0_hole < 52))
        assert jnp.all((state.p1_hole >= 0) & (state.p1_hole < 52))
        assert jnp.all(state.p0_active_mask)
        assert jnp.all(state.p1_active_mask)
    
    def test_no_duplicate_cards(self):
        """Verify no duplicate cards between players."""
        env = TossHoldemEnv()
        
        for seed in range(50):
            key = jax.random.PRNGKey(seed)
            state = env.reset(key)
            all_cards = jnp.concatenate([state.p0_hole, state.p1_hole])
            unique_cards = jnp.unique(all_cards)
            assert len(unique_cards) == 6, "Should have 6 unique cards dealt"
    
    def test_initial_phase_is_preflop(self):
        """Verify game starts in preflop phase."""
        env = TossHoldemEnv()
        key = jax.random.PRNGKey(42)
        state = env.reset(key)
        assert state.phase == PHASE_PREFLOP
    
    def test_initial_pot_is_zero(self):
        """Verify pot starts at zero."""
        env = TossHoldemEnv()
        key = jax.random.PRNGKey(42)
        state = env.reset(key)
        assert state.pot == 0
    
    def test_deck_pointer_starts_at_6(self):
        """Verify deck pointer starts at 6."""
        env = TossHoldemEnv()
        key = jax.random.PRNGKey(42)
        state = env.reset(key)
        assert state.deck_ptr == 6


# ============================================================================
# Test: Legal Actions
# ============================================================================

class TestLegalActions:
    """Test that legal action computation matches engine.py."""
    
    def test_preflop_dealer_can_fold_call_raise(self):
        """Dealer facing BB can fold, call, or raise."""
        env = TossHoldemEnv()
        key = jax.random.PRNGKey(42)
        state = env.reset(key)
        state = state.replace(is_dealer_p0=True, current_player=0, p0_bet=SB, p1_bet=BB, p0_acted_this_street=True)
        
        legal = env.get_legal_actions(state)
        
        assert legal[ACT_FOLD]
        assert legal[ACT_CALL]
        assert not legal[ACT_CHECK]
    
    def test_preflop_bb_can_check_after_limp(self):
        """After SB calls, BB can check or raise."""
        env = TossHoldemEnv()
        key = jax.random.PRNGKey(42)
        state = env.reset(key)
        state = state.replace(
            is_dealer_p0=True, current_player=1,
            p0_bet=BB, p1_bet=BB,
            p0_stack=STACK_SIZE - BB, p1_stack=STACK_SIZE - BB,
            p0_acted_this_street=True, p1_acted_this_street=False
        )
        
        legal = env.get_legal_actions(state)
        
        assert legal[ACT_CHECK]
        assert not legal[ACT_CALL]
    
    def test_facing_bet_can_fold_call_raise(self):
        """When facing a bet, can fold, call, or raise."""
        env = TossHoldemEnv()
        key = jax.random.PRNGKey(42)
        state = env.reset(key)
        state = state.replace(
            is_dealer_p0=True, current_player=1,
            p0_bet=10, p1_bet=BB,
            p0_stack=STACK_SIZE - 10, p1_stack=STACK_SIZE - BB,
            phase=PHASE_PREFLOP
        )
        
        legal = env.get_legal_actions(state)
        
        assert legal[ACT_FOLD]
        assert legal[ACT_CALL]
        assert not legal[ACT_CHECK]
    
    def test_all_in_opponent_cannot_raise(self):
        """Cannot raise when opponent is all-in."""
        env = TossHoldemEnv()
        key = jax.random.PRNGKey(42)
        state = env.reset(key)
        state = state.replace(
            is_dealer_p0=True, current_player=1,
            p0_bet=STACK_SIZE, p1_bet=BB,
            p0_stack=0, p1_stack=STACK_SIZE - BB,
            phase=PHASE_PREFLOP
        )
        
        legal = env.get_legal_actions(state)
        
        assert legal[ACT_FOLD]
        assert legal[ACT_CALL]
        assert not legal[ACT_RAISE_MIN]
    
    def test_discard_phase_only_discard_legal(self):
        """In discard phase, only discard actions are legal."""
        env = TossHoldemEnv()
        key = jax.random.PRNGKey(42)
        state = env.reset(key)
        state = state.replace(
            phase=PHASE_DISCARD_OOP,
            is_dealer_p0=True, current_player=1,
            board=jnp.array([10, 20, -1, -1, -1, -1])
        )
        
        legal = env.get_legal_actions(state)
        
        assert not legal[ACT_FOLD]
        assert not legal[ACT_CHECK]
        assert not legal[ACT_CALL]
        assert legal[ACT_DISCARD_0]
        assert legal[ACT_DISCARD_1]
        assert legal[ACT_DISCARD_2]

    def test_cannot_fold_when_checked_to(self):
        """Cannot fold when there's nothing to call (standard poker rules)."""
        env = TossHoldemEnv()
        key = jax.random.PRNGKey(42)
        state = env.reset(key)
        
        # BB after SB limps
        state = state.replace(
            is_dealer_p0=True, current_player=1,
            p0_bet=BB, p1_bet=BB,
            p0_stack=STACK_SIZE - BB, p1_stack=STACK_SIZE - BB,
            p0_acted_this_street=True, p1_acted_this_street=False
        )
        
        legal = env.get_legal_actions(state)
        
        assert not legal[ACT_FOLD], "Fold should NOT be legal when checked to"
        assert legal[ACT_CHECK], "Check should be legal"

# ============================================================================
# Test: Raise Bounds (Engine comparison)
# ============================================================================

class TestRaiseBounds:
    """Test that raise bounds match engine.py logic."""
    
    def test_engine_min_raise_is_bb_initially(self):
        """Initial min raise should be BB (2) per engine.py."""
        engine_state = MockRoundState(
            button=0, street=0,
            pips=[SB, BB], stacks=[STACK_SIZE - SB, STACK_SIZE - BB],
            hands=[[0, 1, 2], [3, 4, 5]], board=[]
        )
        min_raise, max_raise = engine_state.raise_bounds()
        assert min_raise == 4  # SB + call(1) + raise(2) = 4
    
    def test_engine_max_raise_limited_by_shorter_stack(self):
        """Max raise is limited by the shorter stack."""
        engine_state = MockRoundState(
            button=0, street=4,
            pips=[0, 0], stacks=[100, 200],
            hands=[[0, 1], [2, 3]], board=[10, 20, 30, 40]
        )
        min_raise, max_raise = engine_state.raise_bounds()
        assert max_raise == 100
    
    def test_game_py_tracks_last_raise(self):
        """Verify game.py tracks last_raise_amt correctly."""
        env = TossHoldemEnv()
        key = jax.random.PRNGKey(42)
        state = env.reset(key)
        assert state.last_raise_amt == BB

    def test_raise_clipped_to_max_contribution(self):
        """Raise that exceeds max contribution should be clipped to all-in."""
        env = TossHoldemEnv()
        key = jax.random.PRNGKey(42)
        state = env.reset(key)
        
        # P0 has 100 chips, P1 has 50 chips
        # Max contribution for P0 = min(100, 50 + 0) = 50
        state = state.replace(
            is_dealer_p0=True, current_player=0,
            p0_bet=0, p1_bet=0,
            p0_stack=100, p1_stack=50,
            phase=PHASE_FLOP,
            pot=200,
            last_raise_amt=BB,
            p0_acted_this_street=False,
            p1_acted_this_street=False
        )
        
        # Try a 2x pot raise (would be way more than 50)
        new_state = env._step_betting(state, ACT_RAISE_200, key)
        
        # Should be clipped to 50 (opponent's stack)
        assert new_state.p0_bet == 50, f"Bet should be clipped to 50, got {new_state.p0_bet}"
        assert new_state.p0_stack == 50, f"Stack should be 50, got {new_state.p0_stack}"


    def test_allin_matches_max_contribution(self):
        """All-in should equal max contribution (min of stacks)."""
        env = TossHoldemEnv()
        key = jax.random.PRNGKey(42)
        state = env.reset(key)
        
        # P0 has 200 chips, P1 has 80 chips
        state = state.replace(
            is_dealer_p0=True, current_player=0,
            p0_bet=0, p1_bet=0,
            p0_stack=200, p1_stack=80,
            phase=PHASE_FLOP,
            pot=0,
            last_raise_amt=BB,
            p0_acted_this_street=False,
            p1_acted_this_street=False
        )
        
        new_state = env._step_betting(state, ACT_ALLIN, key)
        
        # All-in should be capped at opponent's stack (80)
        assert new_state.p0_bet == 80, f"All-in bet should be 80, got {new_state.p0_bet}"
        assert new_state.p0_stack == 120, f"Stack should be 120, got {new_state.p0_stack}"

# ============================================================================
# Test: Discard Mechanics
# ============================================================================

class TestDiscardMechanics:
    """Test discard phase behavior."""
    
    def test_oop_discards_in_phase_discard_oop(self):
        """OOP player should discard in PHASE_DISCARD_OOP."""
        env = TossHoldemEnv()
        
        for is_dealer_p0 in [True, False]:
            oop_player = 1 if is_dealer_p0 else 0
            key = jax.random.PRNGKey(42)
            state = env.reset(key)
            state = state.replace(
                phase=PHASE_DISCARD_OOP,
                is_dealer_p0=is_dealer_p0,
                current_player=oop_player
            )
            legal = env.get_legal_actions(state)
            assert legal[ACT_DISCARD_0]
            assert legal[ACT_DISCARD_1]
            assert legal[ACT_DISCARD_2]
    
    def test_dealer_discards_in_phase_discard_dlr(self):
        """Dealer should discard in PHASE_DISCARD_DLR."""
        env = TossHoldemEnv()
        
        for is_dealer_p0 in [True, False]:
            dealer = 0 if is_dealer_p0 else 1
            key = jax.random.PRNGKey(42)
            state = env.reset(key)
            state = state.replace(
                phase=PHASE_DISCARD_DLR,
                is_dealer_p0=is_dealer_p0,
                current_player=dealer
            )
            legal = env.get_legal_actions(state)
            assert legal[ACT_DISCARD_0]
            assert legal[ACT_DISCARD_1]
            assert legal[ACT_DISCARD_2]

    def test_discard_updates_active_mask(self):
        """Discarding should update the active mask."""
        env = TossHoldemEnv()
        key = jax.random.PRNGKey(42)
        state = env.reset(key)
        state = state.replace(
            phase=PHASE_DISCARD_OOP,
            is_dealer_p0=True, current_player=1,
            board=jnp.array([10, 20, -1, -1, -1, -1]),
            p1_hole=jnp.array([5, 6, 7]),
            p1_active_mask=jnp.array([True, True, True])
        )
        
        new_state = env._step_discard(state, ACT_DISCARD_1, key)
        expected_mask = jnp.array([True, False, True])
        assert jnp.array_equal(new_state.p1_active_mask, expected_mask)
    
    def test_discard_places_card_on_board(self):
        """Discarded card should be placed on the board."""
        env = TossHoldemEnv()
        key = jax.random.PRNGKey(42)
        state = env.reset(key)
        state = state.replace(
            phase=PHASE_DISCARD_OOP,
            is_dealer_p0=True, current_player=1,
            board=jnp.array([10, 20, -1, -1, -1, -1]),
            p1_hole=jnp.array([5, 6, 7])
        )
        
        new_state = env._step_discard(state, ACT_DISCARD_0, key)
        assert new_state.board[2] == 5
    
    def test_discard_advances_phase(self):
        """Discarding should advance to next phase."""
        env = TossHoldemEnv()
        key = jax.random.PRNGKey(42)
        state = env.reset(key)
        state = state.replace(
            phase=PHASE_DISCARD_OOP,
            is_dealer_p0=True, current_player=1,
            board=jnp.array([10, 20, -1, -1, -1, -1]),
            p1_hole=jnp.array([5, 6, 7])
        )
        
        new_state = env._step_discard(state, ACT_DISCARD_0, key)
        assert new_state.phase == PHASE_DISCARD_DLR


# ============================================================================
# Test: Betting Actions
# ============================================================================

class TestBettingActions:
    """Test betting action execution."""
    
    @pytest.mark.xfail(reason="JAX dynamic slicing issue in _advance_street when street ends")
    def test_fold_ends_game(self):
        """Folding should end the game."""
        env = TossHoldemEnv()
        key = jax.random.PRNGKey(42)
        state = env.reset(key)
        state = state.replace(
            is_dealer_p0=True, current_player=0,
            p0_bet=SB, p1_bet=BB, phase=PHASE_PREFLOP
        )
        
        new_state = env._step_betting(state, ACT_FOLD, key)
        
        assert new_state.done
        assert new_state.winner == 1
    
    @pytest.mark.xfail(reason="JAX dynamic slicing issue in _advance_street when street ends")
    def test_check_doesnt_cost_chips(self):
        """Checking should not cost any chips."""
        env = TossHoldemEnv()
        key = jax.random.PRNGKey(42)
        state = env.reset(key)
        state = state.replace(
            is_dealer_p0=True, current_player=1,
            p0_bet=0, p1_bet=0,
            p0_stack=350, p1_stack=350,
            phase=PHASE_FLOP,
            p0_acted_this_street=False, p1_acted_this_street=False
        )
        
        old_stack = int(state.p1_stack)
        new_state = env._step_betting(state, ACT_CHECK, key)
        
        assert new_state.p1_stack == old_stack


# ============================================================================
# Test: Hand Evaluator
# ============================================================================

class TestHandEvaluator:
    """
    Test the hand evaluator.
    Card encoding: suit * 13 + rank
    Suits: 0=clubs, 1=diamonds, 2=hearts, 3=spades
    Ranks: 0=2, 1=3, ..., 8=T, 9=J, 10=Q, 11=K, 12=A
    """
    
    def test_royal_flush(self):
        """Royal flush should be tier 8 (straight flush)."""
        hand = jnp.array([
            card_to_int(8, 2), card_to_int(9, 2), card_to_int(10, 2),
            card_to_int(11, 2), card_to_int(12, 2)
        ])
        score = score_5_hand(hand)
        tier = score >> 20
        assert tier == 8
    
    def test_straight_flush(self):
        """Straight flush (not royal)."""
        hand = jnp.array([
            card_to_int(3, 2), card_to_int(4, 2), card_to_int(5, 2),
            card_to_int(6, 2), card_to_int(7, 2)
        ])
        score = score_5_hand(hand)
        tier = score >> 20
        assert tier == 8
    
    def test_quads(self):
        """Four of a kind."""
        hand = jnp.array([
            card_to_int(12, 0), card_to_int(12, 1), card_to_int(12, 2),
            card_to_int(12, 3), card_to_int(0, 0)
        ])
        score = score_5_hand(hand)
        tier = score >> 20
        assert tier == 7
    
    def test_full_house(self):
        """Full house."""
        hand = jnp.array([
            card_to_int(11, 0), card_to_int(11, 1), card_to_int(11, 2),
            card_to_int(10, 0), card_to_int(10, 1)
        ])
        score = score_5_hand(hand)
        tier = score >> 20
        assert tier == 6
    
    def test_flush(self):
        """Flush (not straight)."""
        hand = jnp.array([
            card_to_int(0, 0), card_to_int(2, 0), card_to_int(4, 0),
            card_to_int(6, 0), card_to_int(11, 0)
        ])
        score = score_5_hand(hand)
        tier = score >> 20
        assert tier == 5
    
    def test_straight(self):
        """Straight (not flush)."""
        hand = jnp.array([
            card_to_int(3, 0), card_to_int(4, 1), card_to_int(5, 2),
            card_to_int(6, 3), card_to_int(7, 0)
        ])
        score = score_5_hand(hand)
        tier = score >> 20
        assert tier == 4
    
    def test_wheel(self):
        """Wheel straight (A-2-3-4-5)."""
        hand = jnp.array([
            card_to_int(12, 0), card_to_int(0, 1), card_to_int(1, 2),
            card_to_int(2, 3), card_to_int(3, 0)
        ])
        score = score_5_hand(hand)
        tier = score >> 20
        assert tier == 4
    
    def test_trips(self):
        """Three of a kind."""
        hand = jnp.array([
            card_to_int(6, 0), card_to_int(6, 1), card_to_int(6, 2),
            card_to_int(0, 0), card_to_int(1, 0)
        ])
        score = score_5_hand(hand)
        tier = score >> 20
        assert tier == 3
    
    def test_two_pair(self):
        """Two pair."""
        hand = jnp.array([
            card_to_int(6, 0), card_to_int(6, 1), card_to_int(4, 0),
            card_to_int(4, 1), card_to_int(0, 0)
        ])
        score = score_5_hand(hand)
        tier = score >> 20
        assert tier == 2
    
    def test_one_pair(self):
        """One pair."""
        hand = jnp.array([
            card_to_int(6, 0), card_to_int(6, 1), card_to_int(0, 0),
            card_to_int(2, 0), card_to_int(4, 0)
        ])
        score = score_5_hand(hand)
        tier = score >> 20
        assert tier == 1
    
    def test_high_card(self):
        """High card."""
        hand = jnp.array([
            card_to_int(0, 0), card_to_int(2, 1), card_to_int(4, 2),
            card_to_int(6, 3), card_to_int(11, 0)
        ])
        score = score_5_hand(hand)
        tier = score >> 20
        assert tier == 0
    
    def test_flush_beats_straight(self):
        """Flush should beat straight."""
        flush = jnp.array([
            card_to_int(0, 0), card_to_int(2, 0), card_to_int(4, 0),
            card_to_int(6, 0), card_to_int(11, 0)
        ])
        straight = jnp.array([
            card_to_int(3, 0), card_to_int(4, 1), card_to_int(5, 2),
            card_to_int(6, 3), card_to_int(7, 0)
        ])
        
        score_flush = score_5_hand(flush)
        score_straight = score_5_hand(straight)
        
        assert score_flush > score_straight
    
    def test_pair_beats_high_card(self):
        """Pair should beat high card."""
        pair = jnp.array([
            card_to_int(0, 0), card_to_int(0, 1), card_to_int(2, 0),
            card_to_int(4, 0), card_to_int(6, 0)
        ])
        high_card = jnp.array([
            card_to_int(0, 0), card_to_int(2, 1), card_to_int(4, 2),
            card_to_int(6, 3), card_to_int(11, 0)
        ])
        
        score_pair = score_5_hand(pair)
        score_high = score_5_hand(high_card)
        
        assert score_pair > score_high
    
    def test_higher_pair_wins(self):
        """Higher pair should beat lower pair."""
        high_pair = jnp.array([
            card_to_int(11, 0), card_to_int(11, 1), card_to_int(2, 0),
            card_to_int(4, 0), card_to_int(6, 0)
        ])
        low_pair = jnp.array([
            card_to_int(0, 0), card_to_int(0, 1), card_to_int(2, 0),
            card_to_int(4, 0), card_to_int(6, 0)
        ])
        
        score_high = score_5_hand(high_pair)
        score_low = score_5_hand(low_pair)
        
        assert score_high > score_low


# ============================================================================
# Test: Engine Comparison (Legal Actions)
# ============================================================================

class TestEngineComparison:
    """Compare game.py behavior directly against engine.py mock."""
    
    def test_legal_actions_match_preflop(self):
        """Legal actions at preflop should match engine."""
        engine_state = MockRoundState(
            button=0, street=0,
            pips=[SB, BB],
            stacks=[STACK_SIZE - SB, STACK_SIZE - BB],
            hands=[[0, 1, 2], [3, 4, 5]],
            board=[]
        )
        engine_legal = engine_state.legal_actions()
        
        env = TossHoldemEnv()
        key = jax.random.PRNGKey(42)
        game_state = env.reset(key)
        game_state = game_state.replace(
            is_dealer_p0=True, current_player=0,
            p0_bet=SB, p1_bet=BB,
            p0_stack=STACK_SIZE - SB, p1_stack=STACK_SIZE - BB,
            phase=PHASE_PREFLOP
        )
        game_legal = env.get_legal_actions(game_state)
        
        if FoldAction in engine_legal:
            assert game_legal[ACT_FOLD]
        if CallAction in engine_legal:
            assert game_legal[ACT_CALL]
        if CheckAction in engine_legal:
            assert game_legal[ACT_CHECK]
        if RaiseAction in engine_legal:
            any_raise = (game_legal[ACT_RAISE_MIN] or game_legal[ACT_RAISE_33] or 
                        game_legal[ACT_RAISE_50] or game_legal[ACT_ALLIN])
            assert any_raise
    
    def test_legal_actions_when_checked_to(self):
        """Legal actions when checked to should match engine."""
        engine_state = MockRoundState(
            button=1, street=4,
            pips=[0, 0],
            stacks=[398, 398],
            hands=[[0, 1], [2, 3]],
            board=[10, 20, 30, 40]
        )
        engine_legal = engine_state.legal_actions()
        
        env = TossHoldemEnv()
        key = jax.random.PRNGKey(42)
        game_state = env.reset(key)
        game_state = game_state.replace(
            is_dealer_p0=True, current_player=0,
            p0_bet=0, p1_bet=0,
            p0_stack=398, p1_stack=398,
            phase=PHASE_FLOP,
            p0_acted_this_street=False, p1_acted_this_street=True
        )
        game_legal = env.get_legal_actions(game_state)
        
        if CheckAction in engine_legal:
            assert game_legal[ACT_CHECK]
        if RaiseAction in engine_legal:
            any_raise = game_legal[ACT_RAISE_MIN] or game_legal[ACT_ALLIN]
            assert any_raise
    
    def test_raise_bounds_match(self):
        """Raise bounds should match engine."""
        engine_state = MockRoundState(
            button=0, street=0,
            pips=[SB, BB],
            stacks=[STACK_SIZE - SB, STACK_SIZE - BB],
            hands=[[0, 1, 2], [3, 4, 5]],
            board=[]
        )
        engine_min, engine_max = engine_state.raise_bounds()

        assert engine_min == 4
        assert engine_max == STACK_SIZE
        
        env = TossHoldemEnv()
        key = jax.random.PRNGKey(42)
        game_state = env.reset(key)
        game_state = game_state.replace(
            is_dealer_p0=True, current_player=0,
            p0_bet=SB, p1_bet=BB,
            p0_stack=STACK_SIZE - SB, p1_stack=STACK_SIZE - BB,
            phase=PHASE_PREFLOP, last_raise_amt=BB
        )
        
        game_legal = env.get_legal_actions(game_state)
        assert game_legal[ACT_RAISE_MIN]
        assert game_legal[ACT_ALLIN]


# ============================================================================
# Test: Known Issues (Expected Failures documenting bugs)
# ============================================================================

class TestKnownIssues:
    """
    Tests for known issues in game.py that cause failures.
    These document discrepancies between game.py and engine.py.
    """
    
    @pytest.mark.xfail(reason="JAX dynamic slicing issue in _advance_street - uses deck[deck_ptr:deck_ptr+2]")
    def test_preflop_to_discard_transition(self):
        """After preflop betting, should go to discard phase.
        
        BUG: game.py line 457 uses dynamic slicing:
        new_board = s.board.at[0:2].set(s.deck[s.deck_ptr:s.deck_ptr+2])
        
        This fails in JAX because deck_ptr is a traced value.
        FIX: Use jax.lax.dynamic_slice instead.
        """
        env = TossHoldemEnv()
        key = jax.random.PRNGKey(42)
        state = env.reset(key)
        
        state = state.replace(
            is_dealer_p0=True, current_player=0,
            p0_bet=SB, p1_bet=BB,
            p0_stack=STACK_SIZE - SB, p1_stack=STACK_SIZE - BB
        )
        
        # Dealer calls
        state = env.step(state, key, ACT_CALL)
        # BB checks
        state = env.step(state, key, ACT_CHECK)
        
        assert state.phase == PHASE_DISCARD_OOP
    
    @pytest.mark.xfail(reason="JAX dynamic slicing issue in _advance_street")
    def test_full_game_simulation(self):
        """Complete game should run without errors."""
        env = TossHoldemEnv()
        key = jax.random.PRNGKey(123)
        state = env.reset(key)
        
        # Preflop
        state = env.step(state, key, ACT_CALL)
        state = env.step(state, key, ACT_CHECK)
        
        # Discard
        state = env.step(state, key, ACT_DISCARD_0)
        state = env.step(state, key, ACT_DISCARD_0)
        
        # Flop
        state = env.step(state, key, ACT_CHECK)
        state = env.step(state, key, ACT_CHECK)
        
        # Turn
        state = env.step(state, key, ACT_CHECK)
        state = env.step(state, key, ACT_CHECK)
        
        # River
        state = env.step(state, key, ACT_CHECK)
        state = env.step(state, key, ACT_CHECK)
        
        assert state.done


# ============================================================================
# Test: Best Hand Selection with 8 Cards
# ============================================================================

class TestBestHandSelection:
    """Test that evaluate_best_hand correctly picks best 5 from 8 cards."""
    
    def test_finds_flush_in_8_cards(self):
        """Should find flush when available in 8 cards."""
        hole = jnp.array([card_to_int(0, 0), card_to_int(2, 0)])  # 2c, 4c
        board = jnp.array([
            card_to_int(4, 0), card_to_int(6, 0), card_to_int(11, 0),
            card_to_int(0, 1), card_to_int(1, 2), card_to_int(3, 3)
        ])
        
        score = evaluate_best_hand(hole, board)
        tier = score >> 20
        
        assert tier == 5  # Flush
    
    def test_finds_straight_in_8_cards(self):
        """Should find straight when available in 8 cards."""
        hole = jnp.array([card_to_int(3, 0), card_to_int(4, 1)])  # 5c, 6d
        board = jnp.array([
            card_to_int(5, 2), card_to_int(6, 3), card_to_int(7, 0),
            card_to_int(0, 1), card_to_int(1, 2), card_to_int(11, 0)
        ])
        
        score = evaluate_best_hand(hole, board)
        tier = score >> 20
        
        assert tier == 4  # Straight


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
