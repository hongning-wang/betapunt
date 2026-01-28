"""
Poker bot using trained transformer model for Toss-or-Hold'em.
Includes opponent statistics tracking for history encoding.
"""
from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction, DiscardAction
from skeleton.states import GameState, TerminalState, RoundState
from skeleton.states import NUM_ROUNDS, STARTING_STACK, BIG_BLIND, SMALL_BLIND
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot

import os
import random
import traceback
import math
from typing import List, Optional, Tuple, Dict

import torch
import torch.nn.functional as F

# Import our model
from torch_transformer_model import (
    TossHoldemTransformerConfig,
    TossHoldemTransformerTorch,
    load_exported_transformer_model,
    apply_legal_action_mask,
)


# =============================================================================
# Constants (matching training/model.py)
# =============================================================================

# Model action indices
ACT_FOLD = 0
ACT_CHECK = 1
ACT_CALL = 2
ACT_RAISE_MIN = 3
ACT_RAISE_33 = 4
ACT_RAISE_50 = 5
ACT_RAISE_75 = 6
ACT_RAISE_125 = 7
ACT_RAISE_200 = 8
ACT_ALLIN = 9
ACT_DISCARD_0 = 10
ACT_DISCARD_1 = 11
ACT_DISCARD_2 = 12

# Tensor dimensions
CARD_SEQ_LEN = 9
ACTION_SEQ_LEN = 32
CARD_INPUT_DIM = 2
ACTION_INPUT_DIM = 8  # Updated: now includes is_dealer
HISTORY_INPUT_DIM = 64
NUM_ACTIONS = 13
NUM_CARD_TYPES = 6

# Card type IDs (semantic, matching training/model.py)
CARD_TYPE_HEROPRIV = 0
CARD_TYPE_FLOP = 1
CARD_TYPE_HERODISCARD = 2
CARD_TYPE_VILLAINDISCARD = 3
CARD_TYPE_TURN = 4
CARD_TYPE_RIVER = 5

# Paths
BOT_DIR = os.path.dirname(__file__)
DEFAULT_MODEL_PATH = os.path.join(BOT_DIR, "model.pt")
DEFAULT_CHECKPOINT_DIR = os.path.join(BOT_DIR, "step_14000")

# Street IDs for action history
STREET_PREFLOP = 0
STREET_DISCARD = 1
STREET_FLOP = 2
STREET_TURN = 3
STREET_RIVER = 4

# Phase constants (matching opponent_stats.py)
PHASE_PREFLOP = 0
PHASE_FLOP = 3
PHASE_TURN = 4
PHASE_RIVER = 5


# =============================================================================
# Card Conversion Utilities
# =============================================================================

RANK_MAP = {
    '2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6,
    '9': 7, 'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12
}
SUIT_CHARS = {'s', 'h', 'd', 'c'}


def parse_card(card_str: str) -> Tuple[int, str]:
    """Parse card string into (rank, suit_char)."""
    if len(card_str) != 2:
        return -1, ''
    rank_char = card_str[0]
    suit_char = card_str[1].lower()
    if rank_char not in RANK_MAP or suit_char not in SUIT_CHARS:
        return -1, ''
    return RANK_MAP[rank_char], suit_char


def canonize_and_sort_cards(
    hero_cards: List[str],
    flop_cards: List[str],
    hero_discard: Optional[str],
    villain_discard: Optional[str],
    turn_card: Optional[str],
    river_card: Optional[str],
) -> Tuple[List[int], List[int], int, int, int, int, List[int]]:
    """Canonize suits and sort cards by rank."""
    suit_canon = {}

    def get_canon_suit(suit_char: str) -> int:
        if suit_char not in suit_canon:
            suit_canon[suit_char] = len(suit_canon)
        return suit_canon[suit_char]

    def card_to_index(card_str: Optional[str]) -> int:
        if not card_str:
            return -1
        rank, suit_char = parse_card(card_str)
        if rank < 0:
            return -1
        canon_suit = get_canon_suit(suit_char)
        return canon_suit * 13 + rank

    # Process hero cards first, then sort by rank
    hero_parsed = []
    for orig_pos, card_str in enumerate(hero_cards[:3]):
        rank, suit_char = parse_card(card_str)
        if rank >= 0:
            canon_suit = get_canon_suit(suit_char)
            hero_parsed.append((rank, canon_suit, canon_suit * 13 + rank, orig_pos))
        else:
            hero_parsed.append((-1, -1, -1, orig_pos))

    hero_parsed_valid = [(r, s, idx, orig) for r, s, idx, orig in hero_parsed if r >= 0]
    hero_parsed_invalid = [(r, s, idx, orig) for r, s, idx, orig in hero_parsed if r < 0]
    hero_parsed_valid.sort(key=lambda x: x[0])
    hero_sorted = hero_parsed_valid + hero_parsed_invalid
    hero_indices = [idx for _, _, idx, _ in hero_sorted]
    hero_sort_map = [orig for _, _, _, orig in hero_sorted]
    while len(hero_indices) < 3:
        hero_indices.append(-1)
    while len(hero_sort_map) < 3:
        hero_sort_map.append(-1)

    # Process flop cards, then sort by rank
    flop_parsed = []
    for card_str in flop_cards[:2]:
        rank, suit_char = parse_card(card_str)
        if rank >= 0:
            canon_suit = get_canon_suit(suit_char)
            flop_parsed.append((rank, canon_suit * 13 + rank))
        else:
            flop_parsed.append((-1, -1))

    flop_parsed_valid = [(r, idx) for r, idx in flop_parsed if r >= 0]
    flop_parsed_invalid = [(r, idx) for r, idx in flop_parsed if r < 0]
    flop_parsed_valid.sort(key=lambda x: x[0])
    flop_sorted = flop_parsed_valid + flop_parsed_invalid
    flop_indices = [idx for _, idx in flop_sorted]
    while len(flop_indices) < 2:
        flop_indices.append(-1)

    hero_discard_idx = card_to_index(hero_discard)
    villain_discard_idx = card_to_index(villain_discard)
    turn_idx = card_to_index(turn_card)
    river_idx = card_to_index(river_card)

    return hero_indices, flop_indices, hero_discard_idx, villain_discard_idx, turn_idx, river_idx, hero_sort_map


def skeleton_street_to_model_street(street: int) -> int:
    """Convert skeleton street to model street ID."""
    if street == 0:
        return STREET_PREFLOP
    elif street in (2, 3):
        return STREET_DISCARD
    elif street == 4:
        return STREET_FLOP
    elif street == 5:
        return STREET_TURN
    elif street == 6:
        return STREET_RIVER
    else:
        return STREET_PREFLOP


def skeleton_street_to_phase(street: int) -> int:
    """Convert skeleton street to phase constant."""
    if street == 0:
        return PHASE_PREFLOP
    elif street in (2, 3):
        return PHASE_PREFLOP  # Discard is still "preflop" for stats
    elif street == 4:
        return PHASE_FLOP
    elif street == 5:
        return PHASE_TURN
    elif street == 6:
        return PHASE_RIVER
    else:
        return PHASE_PREFLOP


# =============================================================================
# Opponent Statistics Tracker
# =============================================================================

class OpponentStatsTracker:
    """Tracks opponent statistics for history encoding.
    
    Simplified version of opponent_stats.py for runtime use.
    Tracks frequencies as (successes, trials) pairs.
    """
    
    # Number of frequency stats to track
    NUM_FREQ_STATS = 54
    NUM_AVG_STATS = 9
    
    def __init__(self):
        self.reset_all()
    
    def reset_all(self):
        """Reset all statistics (new opponent)."""
        # Frequency stats: [successes, trials]
        self.freq_stats = [[0.0, 0.0] for _ in range(self.NUM_FREQ_STATS)]
        # Average stats: [sum, count]
        self.avg_stats = [[0.0, 0.0] for _ in range(self.NUM_AVG_STATS)]
        self.rounds_observed = 0
        
        # Per-round state
        self.reset_round_state()
    
    def reset_round_state(self):
        """Reset per-round state (new hand)."""
        self.preflop_aggressor = -1  # 0=dealer, 1=bb, -1=none
        self.opp_bet_flop = False
        self.opp_bet_turn = False
        self.opp_checked_this_street = False
        self.hero_checked_this_street = False
        self.street_has_raise = False
        self.current_street = PHASE_PREFLOP
        self.went_allin_this_hand = False
    
    def on_round_end(self, went_to_showdown: bool, opp_won: bool, opp_is_dealer: bool):
        """Called when a hand ends."""
        # Update showdown stats
        if went_to_showdown:
            # SHOWDOWN_WIN (idx 42)
            self.freq_stats[42][1] += 1  # trial
            if opp_won:
                self.freq_stats[42][0] += 1  # success
            
            # ALLIN_SHOWDOWN_WIN (idx 43)
            if self.went_allin_this_hand:
                self.freq_stats[43][1] += 1
                if opp_won:
                    self.freq_stats[43][0] += 1
        
        # WENT_SHOWDOWN_DEALER/BB (idx 48, 49)
        if opp_is_dealer:
            self.freq_stats[48][1] += 1
            if went_to_showdown:
                self.freq_stats[48][0] += 1
        else:
            self.freq_stats[49][1] += 1
            if went_to_showdown:
                self.freq_stats[49][0] += 1
        
        # AVG_STREETS_SEEN (idx 8)
        streets_seen = {PHASE_PREFLOP: 1, PHASE_FLOP: 2, PHASE_TURN: 3, PHASE_RIVER: 4}.get(self.current_street, 1)
        self.avg_stats[8][0] += streets_seen
        self.avg_stats[8][1] += 1
        
        self.rounds_observed += 1
        self.reset_round_state()
    
    def update(
        self,
        phase: int,
        action: int,
        is_opp_turn: bool,
        opp_is_dealer: bool,
        to_call: float,
        pot_before: float,
        bet_amount: float,
    ):
        """Update statistics based on an action."""
        if not is_opp_turn:
            # Track hero's actions for some stats
            is_check = (action == ACT_CHECK)
            is_raise = (ACT_RAISE_MIN <= action <= ACT_ALLIN)
            
            if is_check:
                self.hero_checked_this_street = True
            if is_raise:
                self.street_has_raise = True
                if phase == PHASE_PREFLOP:
                    self.preflop_aggressor = 0 if opp_is_dealer else 1  # hero is aggressor
            return
        
        # Opponent's turn - update their stats
        is_fold = (action == ACT_FOLD)
        is_check = (action == ACT_CHECK)
        is_call = (action == ACT_CALL)
        is_raise = (ACT_RAISE_MIN <= action <= ACT_ALLIN)
        is_allin = (action == ACT_ALLIN)
        is_discard = (ACT_DISCARD_0 <= action <= ACT_DISCARD_2)
        
        is_vpip = is_call or is_raise
        facing_bet = to_call > 0
        is_bet = is_raise and not facing_bet
        is_raise_vs_bet = is_raise and facing_bet
        
        is_preflop = (phase == PHASE_PREFLOP)
        is_flop = (phase == PHASE_FLOP)
        is_turn = (phase == PHASE_TURN)
        is_river = (phase == PHASE_RIVER)
        
        opp_is_bb = not opp_is_dealer
        
        bet_frac = bet_amount / pot_before if pot_before > 0 else 0.0
        
        # Compute derived conditions
        opp_was_aggressor = (self.preflop_aggressor == 0) if opp_is_dealer else (self.preflop_aggressor == 1)
        hero_was_aggressor = not opp_was_aggressor and self.preflop_aggressor >= 0
        
        is_overbet = (is_bet or is_raise) and (bet_amount > pot_before)
        is_big_call = is_call and (to_call > pot_before * 0.5)
        is_allin_call = is_call and (to_call > pot_before)
        is_big_raise = is_raise and (bet_amount > pot_before)
        is_min_raise = is_raise and (bet_amount < pot_before * 0.4)
        
        # === Update frequency stats ===
        # Index assignments match opponent_stats.py
        
        # PREFLOP stats (0-7)
        if is_preflop and not is_discard:
            if opp_is_dealer:
                self.freq_stats[0][1] += 1  # VPIP_DEALER trial
                if is_vpip:
                    self.freq_stats[0][0] += 1
                self.freq_stats[2][1] += 1  # PREFLOP_RAISE_DEALER trial
                if is_raise:
                    self.freq_stats[2][0] += 1
                self.freq_stats[4][1] += 1  # PREFLOP_FOLD_DEALER trial
                if is_fold:
                    self.freq_stats[4][0] += 1
                self.freq_stats[6][1] += 1  # LIMP_DEALER trial
                if is_call:
                    self.freq_stats[6][0] += 1
            else:  # BB
                self.freq_stats[1][1] += 1  # VPIP_BB trial
                if is_vpip:
                    self.freq_stats[1][0] += 1
                self.freq_stats[3][1] += 1  # PREFLOP_RAISE_BB trial
                if is_raise:
                    self.freq_stats[3][0] += 1
                self.freq_stats[5][1] += 1  # PREFLOP_FOLD_BB trial
                if is_fold:
                    self.freq_stats[5][0] += 1
        
        # FOLD_TO_PROBE (idx 7) - opponent faces our probe bet
        facing_probe = opp_was_aggressor and self.opp_checked_this_street and facing_bet
        if facing_probe:
            self.freq_stats[7][1] += 1
            if is_fold:
                self.freq_stats[7][0] += 1
        
        # FLOP stats (8-19)
        if is_flop:
            if opp_is_dealer:
                self.freq_stats[8][1] += 1  # FLOP_FOLD_DEALER
                if is_fold:
                    self.freq_stats[8][0] += 1
                self.freq_stats[10][1] += 1  # FLOP_CALL_DEALER
                if is_call:
                    self.freq_stats[10][0] += 1
                self.freq_stats[12][1] += 1  # FLOP_RAISE_DEALER
                if is_raise_vs_bet:
                    self.freq_stats[12][0] += 1
                self.freq_stats[14][1] += 1  # FLOP_BET_DEALER
                if is_bet:
                    self.freq_stats[14][0] += 1
            else:
                self.freq_stats[9][1] += 1  # FLOP_FOLD_BB
                if is_fold:
                    self.freq_stats[9][0] += 1
                self.freq_stats[11][1] += 1  # FLOP_CALL_BB
                if is_call:
                    self.freq_stats[11][0] += 1
                self.freq_stats[13][1] += 1  # FLOP_RAISE_BB
                if is_raise_vs_bet:
                    self.freq_stats[13][0] += 1
                self.freq_stats[15][1] += 1  # FLOP_BET_BB
                if is_bet:
                    self.freq_stats[15][0] += 1
            
            # CHECK_RAISE (idx 16)
            if self.opp_checked_this_street and facing_bet:
                self.freq_stats[16][1] += 1
                if is_raise_vs_bet:
                    self.freq_stats[16][0] += 1
            
            # CBET (idx 17, 18)
            if opp_was_aggressor:
                if opp_is_dealer:
                    self.freq_stats[17][1] += 1
                    if is_bet:
                        self.freq_stats[17][0] += 1
                else:
                    self.freq_stats[18][1] += 1
                    if is_bet:
                        self.freq_stats[18][0] += 1
            
            # DONK_BET_BB (idx 19)
            if opp_is_bb and self.preflop_aggressor == 0:  # dealer was aggressor
                self.freq_stats[19][1] += 1
                if is_bet:
                    self.freq_stats[19][0] += 1
        
        # TURN stats (20-28)
        if is_turn:
            if opp_is_dealer:
                self.freq_stats[20][1] += 1  # TURN_FOLD_DEALER
                if is_fold:
                    self.freq_stats[20][0] += 1
                self.freq_stats[22][1] += 1  # TURN_CALL_DEALER
                if is_call:
                    self.freq_stats[22][0] += 1
                self.freq_stats[24][1] += 1  # TURN_RAISE_DEALER
                if is_raise_vs_bet:
                    self.freq_stats[24][0] += 1
                self.freq_stats[26][1] += 1  # TURN_BET_DEALER
                if is_bet:
                    self.freq_stats[26][0] += 1
            else:
                self.freq_stats[21][1] += 1  # TURN_FOLD_BB
                if is_fold:
                    self.freq_stats[21][0] += 1
                self.freq_stats[23][1] += 1  # TURN_CALL_BB
                if is_call:
                    self.freq_stats[23][0] += 1
                self.freq_stats[25][1] += 1  # TURN_RAISE_BB
                if is_raise_vs_bet:
                    self.freq_stats[25][0] += 1
                self.freq_stats[27][1] += 1  # TURN_BET_BB
                if is_bet:
                    self.freq_stats[27][0] += 1
            
            # DOUBLE_BARREL (idx 28) - merged, no position split
            if self.opp_bet_flop and not facing_bet:
                self.freq_stats[28][1] += 1
                if is_bet:
                    self.freq_stats[28][0] += 1
        
        # BIG_RAISE (idx 29)
        if is_raise:
            self.freq_stats[29][1] += 1
            if is_big_raise:
                self.freq_stats[29][0] += 1
        
        # RIVER stats (30-41)
        if is_river:
            if opp_is_dealer:
                self.freq_stats[30][1] += 1  # RIVER_FOLD_DEALER
                if is_fold:
                    self.freq_stats[30][0] += 1
                self.freq_stats[32][1] += 1  # RIVER_CALL_DEALER
                if is_call:
                    self.freq_stats[32][0] += 1
                self.freq_stats[34][1] += 1  # RIVER_RAISE_DEALER
                if is_raise_vs_bet:
                    self.freq_stats[34][0] += 1
                self.freq_stats[36][1] += 1  # RIVER_BET_DEALER
                if is_bet:
                    self.freq_stats[36][0] += 1
                self.freq_stats[40][1] += 1  # RIVER_OVERBET_DEALER
                if is_overbet:
                    self.freq_stats[40][0] += 1
            else:
                self.freq_stats[31][1] += 1  # RIVER_FOLD_BB
                if is_fold:
                    self.freq_stats[31][0] += 1
                self.freq_stats[33][1] += 1  # RIVER_CALL_BB
                if is_call:
                    self.freq_stats[33][0] += 1
                self.freq_stats[35][1] += 1  # RIVER_RAISE_BB
                if is_raise_vs_bet:
                    self.freq_stats[35][0] += 1
                self.freq_stats[37][1] += 1  # RIVER_BET_BB
                if is_bet:
                    self.freq_stats[37][0] += 1
                self.freq_stats[41][1] += 1  # RIVER_OVERBET_BB
                if is_overbet:
                    self.freq_stats[41][0] += 1
            
            # TRIPLE_BARREL (idx 38) - merged
            if self.opp_bet_flop and self.opp_bet_turn and not facing_bet:
                self.freq_stats[38][1] += 1
                if is_bet:
                    self.freq_stats[38][0] += 1
        
        # MIN_RAISE (idx 39)
        if is_raise:
            self.freq_stats[39][1] += 1
            if is_min_raise:
                self.freq_stats[39][0] += 1
        
        # ALLIN stats (44, 45)
        if opp_is_dealer:
            self.freq_stats[44][1] += 1  # ALLIN_DEALER
            if is_allin:
                self.freq_stats[44][0] += 1
        else:
            self.freq_stats[45][1] += 1  # ALLIN_BB
            if is_allin:
                self.freq_stats[45][0] += 1
        
        # ALLIN_CALL (idx 46) - calling a bet > pot
        if facing_bet:
            self.freq_stats[46][1] += 1
            if is_allin_call:
                self.freq_stats[46][0] += 1
        
        # BIG_CALL (idx 47) - calls > 50% pot
        if facing_bet:
            self.freq_stats[47][1] += 1
            if is_big_call:
                self.freq_stats[47][0] += 1
        
        # RERAISE (idx 50, 51)
        if self.street_has_raise:
            if opp_is_dealer:
                self.freq_stats[50][1] += 1
                if is_raise:
                    self.freq_stats[50][0] += 1
            else:
                self.freq_stats[51][1] += 1
                if is_raise:
                    self.freq_stats[51][0] += 1
        
        # FOLD_TO_CBET (idx 52, 53)
        facing_cbet = is_flop and facing_bet and hero_was_aggressor
        if facing_cbet:
            if opp_is_dealer:
                self.freq_stats[52][1] += 1
                if is_fold:
                    self.freq_stats[52][0] += 1
            else:
                self.freq_stats[53][1] += 1
                if is_fold:
                    self.freq_stats[53][0] += 1
        
        # === Update average stats ===
        if is_preflop and is_raise:
            if opp_is_dealer:
                self.avg_stats[0][0] += bet_frac  # OPEN_RAISE_SIZE_DEALER
                self.avg_stats[0][1] += 1
            else:
                self.avg_stats[1][0] += bet_frac  # OPEN_RAISE_SIZE_BB
                self.avg_stats[1][1] += 1
        
        if is_flop and (is_bet or is_raise):
            if opp_is_dealer:
                self.avg_stats[2][0] += bet_frac
                self.avg_stats[2][1] += 1
            else:
                self.avg_stats[3][0] += bet_frac
                self.avg_stats[3][1] += 1
        
        if is_turn and (is_bet or is_raise):
            if opp_is_dealer:
                self.avg_stats[4][0] += bet_frac
                self.avg_stats[4][1] += 1
            else:
                self.avg_stats[5][0] += bet_frac
                self.avg_stats[5][1] += 1
        
        if is_river and (is_bet or is_raise):
            if opp_is_dealer:
                self.avg_stats[6][0] += bet_frac
                self.avg_stats[6][1] += 1
            else:
                self.avg_stats[7][0] += bet_frac
                self.avg_stats[7][1] += 1
        
        # === Update per-round state ===
        if is_check:
            self.opp_checked_this_street = True
        if is_raise:
            self.street_has_raise = True
            if is_preflop:
                self.preflop_aggressor = 0 if opp_is_dealer else 1
        if is_flop and is_bet:
            self.opp_bet_flop = True
        if is_turn and is_bet:
            self.opp_bet_turn = True
        if is_allin:
            self.went_allin_this_hand = True
        
        self.current_street = phase
    
    def on_street_change(self):
        """Called when street changes."""
        self.opp_checked_this_street = False
        self.hero_checked_this_street = False
        self.street_has_raise = False
    
    def to_tensor(self) -> torch.Tensor:
        """Convert statistics to 64-dim tensor."""
        # Compute frequencies
        def freq(idx):
            s, t = self.freq_stats[idx]
            return s / t if t > 0 else 0.5
        
        def avg(idx, default):
            s, c = self.avg_stats[idx]
            return s / c if c > 0 else default
        
        # Build tensor (order matches opponent_stats.py)
        tensor = torch.tensor([
            freq(0),   # 0: VPIP_DEALER
            freq(1),   # 1: VPIP_BB
            freq(2),   # 2: PREFLOP_RAISE_DEALER
            freq(3),   # 3: PREFLOP_RAISE_BB
            freq(4),   # 4: PREFLOP_FOLD_DEALER
            freq(5),   # 5: PREFLOP_FOLD_BB
            avg(0, 2.5),  # 6: OPEN_RAISE_SIZE_DEALER
            avg(1, 3.0),  # 7: OPEN_RAISE_SIZE_BB
            freq(6),   # 8: LIMP_DEALER
            freq(7),   # 9: FOLD_TO_PROBE
            freq(8),   # 10: FLOP_FOLD_DEALER
            freq(9),   # 11: FLOP_FOLD_BB
            freq(10),  # 12: FLOP_CALL_DEALER
            freq(11),  # 13: FLOP_CALL_BB
            freq(12),  # 14: FLOP_RAISE_DEALER
            freq(13),  # 15: FLOP_RAISE_BB
            freq(14),  # 16: FLOP_BET_DEALER
            freq(15),  # 17: FLOP_BET_BB
            avg(2, 0.5),  # 18: FLOP_AVG_BET_SIZE_DEALER
            avg(3, 0.5),  # 19: FLOP_AVG_BET_SIZE_BB
            freq(16),  # 20: FLOP_CHECK_RAISE
            freq(17),  # 21: FLOP_CBET_DEALER
            freq(18),  # 22: FLOP_CBET_BB
            freq(19),  # 23: FLOP_DONK_BB
            freq(20),  # 24: TURN_FOLD_DEALER
            freq(21),  # 25: TURN_FOLD_BB
            freq(22),  # 26: TURN_CALL_DEALER
            freq(23),  # 27: TURN_CALL_BB
            freq(24),  # 28: TURN_RAISE_DEALER
            freq(25),  # 29: TURN_RAISE_BB
            freq(26),  # 30: TURN_BET_DEALER
            freq(27),  # 31: TURN_BET_BB
            avg(4, 0.5),  # 32: TURN_AVG_BET_SIZE_DEALER
            avg(5, 0.5),  # 33: TURN_AVG_BET_SIZE_BB
            freq(28),  # 34: TURN_DOUBLE_BARREL
            freq(29),  # 35: BIG_RAISE
            freq(30),  # 36: RIVER_FOLD_DEALER
            freq(31),  # 37: RIVER_FOLD_BB
            freq(32),  # 38: RIVER_CALL_DEALER
            freq(33),  # 39: RIVER_CALL_BB
            freq(34),  # 40: RIVER_RAISE_DEALER
            freq(35),  # 41: RIVER_RAISE_BB
            freq(36),  # 42: RIVER_BET_DEALER
            freq(37),  # 43: RIVER_BET_BB
            avg(6, 0.5),  # 44: RIVER_AVG_BET_SIZE_DEALER
            avg(7, 0.5),  # 45: RIVER_AVG_BET_SIZE_BB
            freq(38),  # 46: RIVER_TRIPLE_BARREL
            freq(39),  # 47: MIN_RAISE
            freq(40),  # 48: RIVER_OVERBET_DEALER
            freq(41),  # 49: RIVER_OVERBET_BB
            freq(42),  # 50: SHOWDOWN_WIN
            freq(43),  # 51: ALLIN_SHOWDOWN_WIN
            avg(8, 2.0) / 4.0,  # 52: AVG_STREETS_SEEN (normalized)
            math.log1p(self.rounds_observed) / 5.0,  # 53: ROUNDS_OBSERVED (log-scaled)
            freq(44),  # 54: ALLIN_DEALER
            freq(45),  # 55: ALLIN_BB
            freq(46),  # 56: ALLIN_CALL
            freq(47),  # 57: BIG_CALL
            freq(48),  # 58: WENT_SHOWDOWN_DEALER
            freq(49),  # 59: WENT_SHOWDOWN_BB
            freq(50),  # 60: RERAISE_DEALER
            freq(51),  # 61: RERAISE_BB
            freq(52),  # 62: FOLD_TO_CBET_DEALER
            freq(53),  # 63: FOLD_TO_CBET_BB
        ], dtype=torch.float32)
        
        return tensor


# =============================================================================
# State Encoder
# =============================================================================

class StateEncoder:
    """Encodes game state into tensors for the model."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset state for a new round."""
        self.action_history = []
        self.action_ptr = 0
        self.my_discarded_idx = -1
        self.opp_discarded_idx = -1
        self.is_dealer = False
        self.my_hole_cards = []
        self.board_cards = []
        self.hero_sort_map = []
        self.last_street = -1  # Track street changes
    
    def record_action(
        self,
        actor_id: int,
        event_type: int,
        street: int,
        to_call: float,
        pot_committed: float,
        eff_stack: float,
        is_dealer: bool,  # NEW: whether the acting player is dealer
    ):
        """Record an action in the history."""
        self.action_history.append({
            'actor_id': actor_id,
            'event_type': event_type,
            'street_id': street,
            'time_id': self.action_ptr,
            'to_call': to_call,
            'pot_committed': pot_committed,
            'eff_stack': eff_stack,
            'is_dealer': 1.0 if is_dealer else 0.0,  # NEW
        })
        self.action_ptr += 1
    
    def encode_cards(
        self,
        my_cards: List[str],
        board: List[str],
        is_dealer: bool,
        discarded_idx: int,
    ) -> torch.Tensor:
        """Encode cards into (9, 2) tensor with canonized suits."""
        card_tensor = torch.full((CARD_SEQ_LEN, CARD_INPUT_DIM), -1, dtype=torch.long)

        flop_cards = board[:2] if len(board) >= 2 else []

        hero_discard = None
        villain_discard = None
        if len(board) >= 4:
            hero_discard_board_idx = 3 if is_dealer else 2
            villain_discard_board_idx = 2 if is_dealer else 3
            hero_discard = board[hero_discard_board_idx] if len(board) > hero_discard_board_idx else None
            villain_discard = board[villain_discard_board_idx] if len(board) > villain_discard_board_idx else None

        turn_card = board[4] if len(board) >= 5 else None
        river_card = board[5] if len(board) >= 6 else None

        if discarded_idx >= 0:
            hero_cards_for_canon = [c for i, c in enumerate(my_cards[:3]) if i != discarded_idx]
        else:
            hero_cards_for_canon = list(my_cards[:3])

        hero_indices, flop_indices, hero_discard_card_idx, villain_discard_card_idx, turn_idx, river_idx, hero_sort_map = \
            canonize_and_sort_cards(
                hero_cards_for_canon,
                flop_cards,
                hero_discard,
                villain_discard,
                turn_card,
                river_card,
            )

        self.hero_sort_map = hero_sort_map

        for i, card_idx in enumerate(hero_indices[:3]):
            if card_idx >= 0:
                card_tensor[i, 0] = card_idx
                card_tensor[i, 1] = CARD_TYPE_HEROPRIV

        for i, card_idx in enumerate(flop_indices[:2]):
            if card_idx >= 0:
                card_tensor[3 + i, 0] = card_idx
                card_tensor[3 + i, 1] = CARD_TYPE_FLOP

        if hero_discard_card_idx >= 0:
            card_tensor[5, 0] = hero_discard_card_idx
            card_tensor[5, 1] = CARD_TYPE_HERODISCARD

        if villain_discard_card_idx >= 0:
            card_tensor[6, 0] = villain_discard_card_idx
            card_tensor[6, 1] = CARD_TYPE_VILLAINDISCARD

        if turn_idx >= 0:
            card_tensor[7, 0] = turn_idx
            card_tensor[7, 1] = CARD_TYPE_TURN

        if river_idx >= 0:
            card_tensor[8, 0] = river_idx
            card_tensor[8, 1] = CARD_TYPE_RIVER

        return card_tensor
    
    def encode_actions(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode action history into (32, 8) tensor with mask."""
        action_tensor = torch.zeros((ACTION_SEQ_LEN, ACTION_INPUT_DIM), dtype=torch.float32)
        
        n_actions = min(len(self.action_history), ACTION_SEQ_LEN)
        
        for i, action in enumerate(self.action_history[-ACTION_SEQ_LEN:]):
            idx = ACTION_SEQ_LEN - n_actions + i
            action_tensor[idx, 0] = action['actor_id']
            action_tensor[idx, 1] = action['event_type']
            action_tensor[idx, 2] = action['street_id']
            action_tensor[idx, 3] = action['time_id']
            action_tensor[idx, 4] = action['to_call']
            action_tensor[idx, 5] = action['pot_committed']
            action_tensor[idx, 6] = action['eff_stack']
            action_tensor[idx, 7] = action['is_dealer']  # NEW
        
        action_mask = torch.zeros(ACTION_SEQ_LEN, dtype=torch.bool)
        if n_actions > 0:
            action_mask[-n_actions:] = True
        
        return action_tensor, action_mask


# =============================================================================
# Main Bot
# =============================================================================

class Player(Bot):
    """Transformer-based poker bot with opponent modeling."""

    def __init__(self):
        self.model = None
        self.device = "cpu"
        self.encoder = StateEncoder()
        self.opp_stats = OpponentStatsTracker()  # NEW
        self._last_round_state = None
        self._last_street = -1  # Track street for opponent stats
        self._load_model()

    def _load_model(self):
        """Load the trained transformer model."""
        try:
            if os.path.exists(DEFAULT_MODEL_PATH):
                self.model = load_exported_transformer_model(DEFAULT_MODEL_PATH, self.device)
                print(f"Loaded exported model from {DEFAULT_MODEL_PATH}")
            else:
                print(f"No model found at {DEFAULT_MODEL_PATH}")
                self.model = None
        except Exception as e:
            print(f"Error loading model: {e}")
            traceback.print_exc()
            self.model = None

    def handle_new_round(self, game_state: GameState, round_state: RoundState, active: int):
        """Called at the start of a new round."""
        self.encoder.reset()
        self.encoder.is_dealer = (active == 0)
        self.encoder.my_hole_cards = list(round_state.hands[active])
        self._last_round_state = round_state
        self._last_street = round_state.street
        
        # Reset per-round opponent stats (but keep accumulated stats)
        self.opp_stats.reset_round_state()

    def handle_round_over(self, game_state: GameState, terminal_state: TerminalState, active: int):
        """Called at the end of a round."""
        # Determine if went to showdown and who won
        prev_state = terminal_state.previous_state
        last_action_was_fold = False
        if prev_state is not None:
            # Check if the last action was a fold
            continue_cost = prev_state.pips[1 - (prev_state.button % 2)] - prev_state.pips[prev_state.button % 2]
            if continue_cost > 0:
                # There was a bet - check if opponent folded
                opp_idx = 1 - active
                opp_pip_before = prev_state.pips[opp_idx]
                # If opponent's pip didn't change after facing a bet, they folded
                # Actually, in terminal state, we need to check deltas
                my_delta = terminal_state.deltas[active]
                # If we won without showdown (opponent folded), my_delta > 0
                last_action_was_fold = (my_delta != 0) and (len(prev_state.board) < 6)
        
        went_to_showdown = not last_action_was_fold and len(prev_state.board) >= 6 if prev_state else False
        opp_won = terminal_state.deltas[1 - active] > 0
        opp_is_dealer = (1 - active) == 0  # Opponent is player 0 = dealer
        
        self.opp_stats.on_round_end(went_to_showdown, opp_won, opp_is_dealer)
        
        self._last_round_state = None

    def get_action(self, game_state: GameState, round_state: RoundState, active: int):
        """Get action for the current game state."""
        legal_actions = round_state.legal_actions()

        if isinstance(round_state, TerminalState):
            return CheckAction() if CheckAction in legal_actions else FoldAction()

        # Sync action history
        self._sync_action_history(round_state, active)

        # Handle discard phase
        if round_state.street in (2, 3):
            return self._get_discard_action(round_state, legal_actions, active)

        # Handle betting phases
        return self._get_betting_action(round_state, legal_actions, active)

    def _get_discard_action(self, round_state, legal_actions, active):
        """Get action during discard phase."""
        if DiscardAction not in legal_actions:
            return CheckAction() if CheckAction in legal_actions else CallAction()

        if self.model is None:
            return self._fallback_action(round_state, legal_actions)

        try:
            card_tensor = self.encoder.encode_cards(
                round_state.hands[active],
                round_state.board,
                self.encoder.is_dealer,
                -1,  # Haven't discarded yet
            )
            action_tensor, action_mask = self.encoder.encode_actions()
            history_tensor = self.opp_stats.to_tensor()  # NEW

            card_b = card_tensor.unsqueeze(0).to(self.device)
            action_b = action_tensor.unsqueeze(0).to(self.device)
            mask_b = action_mask.unsqueeze(0).to(self.device)
            history_b = history_tensor.unsqueeze(0).to(self.device)  # NEW

            with torch.no_grad():
                logits, _ = self.model(card_b, action_b, mask_b, history_b)

            legal_mask = self._compute_legal_mask(round_state, active)
            legal_mask_t = torch.tensor(legal_mask, dtype=torch.bool).unsqueeze(0)
            masked_logits = apply_legal_action_mask(logits, legal_mask_t)
            
            action_idx = torch.argmax(masked_logits, dim=-1).item()

            if action_idx in (ACT_DISCARD_0, ACT_DISCARD_1, ACT_DISCARD_2):
                sorted_idx = action_idx - ACT_DISCARD_0
                if sorted_idx < len(self.encoder.hero_sort_map):
                    original_idx = self.encoder.hero_sort_map[sorted_idx]
                    if 0 <= original_idx < len(round_state.hands[active]):
                        self.encoder.my_discarded_idx = original_idx
                        self._record_my_action(round_state, active, action_idx)
                        self._advance_last_state(round_state, DiscardAction(original_idx))
                        return DiscardAction(original_idx)

            # Fallback: discard first card
            self.encoder.my_discarded_idx = 0
            self._record_my_action(round_state, active, ACT_DISCARD_0)
            self._advance_last_state(round_state, DiscardAction(0))
            return DiscardAction(0)

        except Exception as e:
            print(f"Error in discard: {e}")
            traceback.print_exc()
            return DiscardAction(0)

    def _get_betting_action(self, round_state, legal_actions, active):
        """Get action during betting phase."""
        if self.model is None:
            return self._fallback_action(round_state, legal_actions)

        try:
            card_tensor = self.encoder.encode_cards(
                round_state.hands[active],
                round_state.board,
                self.encoder.is_dealer,
                self.encoder.my_discarded_idx,
            )
            action_tensor, action_mask = self.encoder.encode_actions()
            history_tensor = self.opp_stats.to_tensor()  # NEW

            card_b = card_tensor.unsqueeze(0).to(self.device)
            action_b = action_tensor.unsqueeze(0).to(self.device)
            mask_b = action_mask.unsqueeze(0).to(self.device)
            history_b = history_tensor.unsqueeze(0).to(self.device)  # NEW

            with torch.no_grad():
                logits, _ = self.model(card_b, action_b, mask_b, history_b)

            legal_mask = self._compute_legal_mask(round_state, active)
            legal_mask_t = torch.tensor(legal_mask, dtype=torch.bool).unsqueeze(0)
            masked_logits = apply_legal_action_mask(logits, legal_mask_t)
            
            action_idx = torch.argmax(masked_logits, dim=-1).item()
            action = self._model_action_to_skeleton(action_idx, round_state, legal_actions, active)
            
            self._record_my_action(round_state, active, action_idx)
            self._advance_last_state(round_state, action)
            return action

        except Exception as e:
            print(f"Error in betting: {e}")
            traceback.print_exc()
            return self._fallback_action(round_state, legal_actions)

    def _raise_bounds_training(self, round_state, active):
        """Get raise bounds matching the training environment."""
        my_pip = round_state.pips[active]
        opp_pip = round_state.pips[1 - active]
        my_stack = round_state.stacks[active]
        opp_stack = round_state.stacks[1 - active]

        continue_cost = opp_pip - my_pip
        last_raise_amt = self._last_raise_amt(round_state)

        min_raise_to = opp_pip + last_raise_amt
        if my_stack + my_pip <= min_raise_to:
            min_raise_to = my_stack + my_pip

        max_raise_to = my_stack + my_pip
        opp_total = opp_stack + opp_pip
        if max_raise_to > opp_total:
            max_raise_to = opp_total

        return (max(min_raise_to, opp_pip + 1), max_raise_to)

    def _last_raise_amt(self, round_state) -> int:
        """Get the size of the last raise."""
        if round_state.street == 0:
            return BIG_BLIND
        
        cursor = round_state
        while cursor is not None:
            prev = cursor.previous_state
            if prev is None:
                break
            if prev.street != round_state.street:
                break
            actor = prev.button % 2
            if cursor.pips[actor] > prev.pips[actor]:
                raise_amt = cursor.pips[actor] - prev.pips[1 - actor]
                if raise_amt > 0:
                    return raise_amt
            cursor = prev
        return BIG_BLIND

    def _compute_legal_mask(self, round_state, active):
        """Compute legal action mask."""
        mask = [False] * NUM_ACTIONS
        legal = round_state.legal_actions()

        if round_state.street in (2, 3):
            if DiscardAction in legal:
                mask[ACT_DISCARD_0] = True
                mask[ACT_DISCARD_1] = True
                mask[ACT_DISCARD_2] = True
            if CheckAction in legal:
                mask[ACT_CHECK] = True
            return mask

        my_pip = round_state.pips[active]
        opp_pip = round_state.pips[1 - active]
        cur_stack = round_state.stacks[active]
        opp_stack = round_state.stacks[1 - active]

        to_call = max(0, opp_pip - my_pip)
        total_committed = (STARTING_STACK * 2) - (round_state.stacks[0] + round_state.stacks[1])
        pot_base = total_committed + to_call
        eff_cap = min(cur_stack, opp_stack + to_call)

        if FoldAction in legal and to_call > 0:
            mask[ACT_FOLD] = True

        if CheckAction in legal:
            mask[ACT_CHECK] = True

        if CallAction in legal:
            mask[ACT_CALL] = True

        if RaiseAction in legal:
            min_raise, max_raise = self._raise_bounds_training(round_state, active)
            last_raise_amt = self._last_raise_amt(round_state)

            if pot_base > 0:
                min_raise_amt = to_call + last_raise_amt
                if my_pip + min_raise_amt <= max_raise:
                    mask[ACT_RAISE_MIN] = True
                
                for frac, action_idx in [(1/3, ACT_RAISE_33), (1/2, ACT_RAISE_50), (3/4, ACT_RAISE_75),
                                         (5/4, ACT_RAISE_125), (2, ACT_RAISE_200)]:
                    raise_amount = my_pip + to_call + int(pot_base * frac)
                    if min_raise <= raise_amount <= max_raise:
                        mask[action_idx] = True
                
                can_bet_when_checked = (to_call == 0) and (cur_stack > 0) and (opp_stack > 0)
                can_all_in = (to_call > 0) or can_bet_when_checked
                if can_all_in and eff_cap > 0:
                    mask[ACT_ALLIN] = True
        
        return mask

    def _model_action_to_skeleton(self, action_idx, round_state, legal_actions, active):
        """Convert model action index to skeleton action."""
        if action_idx == ACT_FOLD:
            to_call = max(0, round_state.pips[1 - active] - round_state.pips[active])
            if FoldAction in legal_actions and to_call > 0:
                return FoldAction()
            return CheckAction() if CheckAction in legal_actions else CallAction()
        
        elif action_idx == ACT_CHECK:
            if CheckAction in legal_actions:
                return CheckAction()
            return CallAction() if CallAction in legal_actions else FoldAction()
        
        elif action_idx == ACT_CALL:
            if CallAction in legal_actions:
                return CallAction()
            return CheckAction() if CheckAction in legal_actions else FoldAction()
        
        elif action_idx in (ACT_RAISE_MIN, ACT_RAISE_33, ACT_RAISE_50, ACT_RAISE_75, ACT_RAISE_125, ACT_RAISE_200, ACT_ALLIN):
            if RaiseAction not in legal_actions:
                return CallAction() if CallAction in legal_actions else CheckAction()

            amount = self._compute_raise_amount(round_state, active, action_idx)
            return RaiseAction(amount)

        if CheckAction in legal_actions:
            return CheckAction()
        return CallAction() if CallAction in legal_actions else FoldAction()
    
    def _record_my_action(self, round_state, active, action_idx):
        """Record our action in the history."""
        self._record_action(round_state, active, 0, action_idx, is_hero=True)

    def _record_action(self, round_state, active, actor_id, action_idx, is_hero=False):
        """Record any action (self or opponent) in the history."""
        my_pip = round_state.pips[active]
        opp_pip = round_state.pips[1 - active]
        my_stack = round_state.stacks[active]
        opp_stack = round_state.stacks[1 - active]

        to_call = max(0, opp_pip - my_pip)
        total_committed = (STARTING_STACK * 2) - (round_state.stacks[0] + round_state.stacks[1])
        pot_base = total_committed + to_call
        eff_stack = min(my_stack, opp_stack + to_call)

        street = skeleton_street_to_model_street(round_state.street)
        
        # Determine if the acting player is dealer
        # actor_id=0 means hero acted, actor_id=1 means opponent acted
        if actor_id == 0:  # Hero
            acting_is_dealer = self.encoder.is_dealer
        else:  # Opponent
            acting_is_dealer = not self.encoder.is_dealer

        self.encoder.record_action(
            actor_id=actor_id,
            event_type=action_idx,
            street=street,
            to_call=float(to_call),
            pot_committed=float(pot_base),
            eff_stack=float(eff_stack),
            is_dealer=acting_is_dealer,  # NEW
        )
        
        # Update opponent stats if opponent acted
        if not is_hero:
            phase = skeleton_street_to_phase(round_state.street)
            
            # Compute bet_amount (simplified - use to_call for calls, estimate for raises)
            if action_idx == ACT_CALL:
                bet_amount = to_call
            elif ACT_RAISE_MIN <= action_idx <= ACT_ALLIN:
                bet_amount = self._compute_raise_amount(round_state, active, action_idx) - my_pip
            else:
                bet_amount = 0
            
            self.opp_stats.update(
                phase=phase,
                action=action_idx,
                is_opp_turn=True,
                opp_is_dealer=acting_is_dealer,
                to_call=float(to_call),
                pot_before=float(pot_base),
                bet_amount=float(bet_amount),
            )

    def _compute_raise_amount(self, round_state, active, action_idx):
        """Compute raise amount for a given action index."""
        min_raise, max_raise = self._raise_bounds_training(round_state, active)
        to_call = max(0, round_state.pips[1 - active] - round_state.pips[active])
        total_committed = (STARTING_STACK * 2) - (round_state.stacks[0] + round_state.stacks[1])
        pot_base = total_committed + to_call
        last_raise_amt = self._last_raise_amt(round_state)

        if action_idx == ACT_RAISE_MIN:
            amount = round_state.pips[active] + to_call + last_raise_amt
        elif action_idx == ACT_RAISE_33:
            amount = round_state.pips[active] + to_call + (pot_base // 3)
        elif action_idx == ACT_RAISE_50:
            amount = round_state.pips[active] + to_call + (pot_base // 2)
        elif action_idx == ACT_RAISE_75:
            amount = round_state.pips[active] + to_call + ((pot_base * 3) // 4)
        elif action_idx == ACT_RAISE_125:
            amount = round_state.pips[active] + to_call + ((pot_base * 5) // 4)
        elif action_idx == ACT_RAISE_200:
            amount = round_state.pips[active] + to_call + (pot_base * 2)
        elif action_idx == ACT_ALLIN:
            amount = max_raise
        else:
            amount = min_raise

        return int(max(min_raise, min(amount, max_raise)))

    def _sync_action_history(self, round_state, active):
        """Record any actions since the last observed state."""
        # Check for street change to update opponent stats
        current_street = round_state.street
        if self._last_street >= 0 and current_street != self._last_street:
            self.opp_stats.on_street_change()
        self._last_street = current_street
        
        if self._last_round_state is None:
            self._last_round_state = round_state
            return

        states = []
        cursor = round_state
        while cursor is not None and cursor is not self._last_round_state:
            states.append(cursor)
            cursor = cursor.previous_state

        if cursor is None:
            self.encoder.reset()
            self.encoder.is_dealer = (active == 0)
            self.encoder.my_hole_cards = list(round_state.hands[active])
            self._last_round_state = round_state
            return

        for state in reversed(states):
            prev = state.previous_state
            if prev is None:
                continue
            action_idx = self._infer_action_index(prev, state, active)
            actor = prev.button % 2
            actor_id = 0 if actor == active else 1
            if action_idx is None:
                continue
            if action_idx in (ACT_DISCARD_0, ACT_DISCARD_1, ACT_DISCARD_2) and actor == active:
                self.encoder.my_discarded_idx = action_idx - ACT_DISCARD_0
            self._record_action(prev, active, actor_id, action_idx, is_hero=(actor == active))

        self._last_round_state = round_state

    def _infer_action_index(self, prev_state, next_state, active):
        """Infer model action index from a state transition."""
        if isinstance(next_state, TerminalState):
            continue_cost = prev_state.pips[1 - (prev_state.button % 2)] - prev_state.pips[prev_state.button % 2]
            if continue_cost > 0:
                return ACT_FOLD
            return None

        actor = prev_state.button % 2
        prev_hands = prev_state.hands[actor]
        next_hands = next_state.hands[actor]

        if len(next_state.board) > len(prev_state.board):
            if len(prev_hands) == len(next_hands) + 1:
                discarded_card = next(set(prev_hands) - set(next_hands), None)
                if discarded_card is not None and discarded_card in prev_hands:
                    discard_idx = prev_hands.index(discarded_card)
                    return ACT_DISCARD_0 + discard_idx

        if prev_state.street in (2, 3) and actor == (prev_state.street % 2):
            return None

        continue_cost = prev_state.pips[1 - actor] - prev_state.pips[actor]
        if next_state.street == prev_state.street and next_state.pips[actor] > prev_state.pips[actor]:
            if next_state.pips[actor] > next_state.pips[1 - actor]:
                amount = next_state.pips[actor]
                return self._map_raise_amount_to_action_idx(prev_state, actor, amount)
            if continue_cost > 0:
                return ACT_CALL

        if continue_cost > 0:
            return ACT_CALL
        return ACT_CHECK

    def _map_raise_amount_to_action_idx(self, round_state, active, amount):
        """Map an observed raise amount to the closest model action index."""
        candidates = [
            ACT_RAISE_MIN, ACT_RAISE_33, ACT_RAISE_50, ACT_RAISE_75,
            ACT_RAISE_125, ACT_RAISE_200, ACT_ALLIN,
        ]
        amounts = {idx: self._compute_raise_amount(round_state, active, idx) for idx in candidates}
        for idx in candidates:
            if amounts[idx] == amount:
                return idx
        best = min(candidates, key=lambda idx: (abs(amounts[idx] - amount), idx))
        return best

    def _advance_last_state(self, round_state, action):
        """Advance internal pointer to keep action sync consistent."""
        self._last_round_state = round_state
    
    def _fallback_action(self, round_state, legal_actions):
        """Fallback action when model is not available."""
        raise RuntimeError(
            "Model unavailable; refusing to take fallback action. "
            "Check model load/export logs for the root cause."
        )


if __name__ == '__main__':
    run_bot(Player(), parse_args())