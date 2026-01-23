"""
Poker bot using trained transformer model for Toss-or-Hold'em.
"""
from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction, DiscardAction
from skeleton.states import GameState, TerminalState, RoundState
from skeleton.states import NUM_ROUNDS, STARTING_STACK, BIG_BLIND, SMALL_BLIND
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot

import os
import random
from typing import List, Optional, Tuple

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
# Constants (matching game.py and model.py)
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
ACTION_SEQ_LEN = 40
CARD_INPUT_DIM = 2
ACTION_INPUT_DIM = 7
NUM_ACTIONS = 13

# Street IDs for action history
STREET_PREFLOP = 0
STREET_DISCARD = 1
STREET_FLOP = 2
STREET_TURN = 3
STREET_RIVER = 4


# =============================================================================
# Card Conversion Utilities
# =============================================================================

def card_string_to_index(card_str: str) -> int:
    """Convert card string like 'As' to index 0-51.
    
    Encoding: suit * 13 + rank
    Suits: s=0, h=1, d=2, c=3 (spades, hearts, diamonds, clubs)
    Ranks: 2=0, 3=1, ..., A=12
    """
    if len(card_str) != 2:
        return -1
    
    rank_char = card_str[0]
    suit_char = card_str[1].lower()
    
    rank_map = {
        '2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6,
        '9': 7, 'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12
    }
    suit_map = {'s': 0, 'h': 1, 'd': 2, 'c': 3}
    
    if rank_char not in rank_map or suit_char not in suit_map:
        return -1
    
    rank = rank_map[rank_char]
    suit = suit_map[suit_char]
    return suit * 13 + rank


def skeleton_street_to_model_street(street: int) -> int:
    """Convert skeleton street to model street ID.
    
    Skeleton streets: 0=preflop, 2=discard?, 3=flop, 4=turn, 5=river
    Model streets: 0=preflop, 1=discard, 2=flop, 3=turn, 4=river
    """
    if street == 0:
        return STREET_PREFLOP
    elif street in (2, 3):  # Discard phases
        return STREET_DISCARD
    elif street == 3:
        return STREET_FLOP
    elif street == 4:
        return STREET_TURN
    elif street == 5:
        return STREET_RIVER
    else:
        return STREET_PREFLOP


# =============================================================================
# State Encoder
# =============================================================================

class StateEncoder:
    """Encodes game state into tensors for the model."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset state for a new round."""
        self.action_history = []  # List of action dicts
        self.action_ptr = 0
        self.my_discarded_idx = -1
        self.opp_discarded_idx = -1
        self.is_dealer = False
        self.my_hole_cards = []
        self.board_cards = []
    
    def record_action(
        self,
        actor_id: int,
        event_type: int,
        street: int,
        to_call: float,
        pot_committed: float,
        eff_stack: float,
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
        })
        self.action_ptr += 1
    
    def encode_cards(
        self,
        my_cards: List[str],
        board: List[str],
        is_dealer: bool,
    ) -> torch.Tensor:
        """Encode cards into (9, 2) tensor.
        
        Card positions:
          0: heropriv0 (type_id=0)
          1: heropriv1 (type_id=1)
          2: heropriv2 (type_id=2)
          3: flop0 (type_id=3)
          4: flop1 (type_id=4)
          5: herodiscard (type_id=5)
          6: villaindiscard (type_id=6)
          7: turn (type_id=7)
          8: river (type_id=8)
        """
        card_tensor = torch.full((CARD_SEQ_LEN, CARD_INPUT_DIM), -1, dtype=torch.long)
        
        # Hero's hole cards (positions 0-2)
        for i, card_str in enumerate(my_cards[:3]):
            card_idx = card_string_to_index(card_str)
            if card_idx >= 0:
                card_tensor[i, 0] = card_idx
                card_tensor[i, 1] = i  # type_id = position
        
        # Board cards
        # In Toss-or-Hold'em:
        # - board[0:2] = flop cards
        # - board[2] = OOP discard
        # - board[3] = dealer discard
        # - board[4] = turn
        # - board[5] = river
        
        # Flop cards (positions 3-4)
        if len(board) >= 2:
            for i, card_str in enumerate(board[:2]):
                card_idx = card_string_to_index(card_str)
                if card_idx >= 0:
                    card_tensor[3 + i, 0] = card_idx
                    card_tensor[3 + i, 1] = 3 + i  # type_id = 3 or 4
        
        # Discards (positions 5-6)
        # Hero discard at position 5, villain at position 6
        if len(board) >= 4:
            # board[2] = OOP discard, board[3] = dealer discard
            hero_discard_board_idx = 3 if is_dealer else 2
            villain_discard_board_idx = 2 if is_dealer else 3
            
            hero_discard = board[hero_discard_board_idx] if len(board) > hero_discard_board_idx else None
            villain_discard = board[villain_discard_board_idx] if len(board) > villain_discard_board_idx else None
            
            if hero_discard:
                card_idx = card_string_to_index(hero_discard)
                if card_idx >= 0:
                    card_tensor[5, 0] = card_idx
                    card_tensor[5, 1] = 5
            
            if villain_discard:
                card_idx = card_string_to_index(villain_discard)
                if card_idx >= 0:
                    card_tensor[6, 0] = card_idx
                    card_tensor[6, 1] = 6
        
        # Turn (position 7)
        if len(board) >= 5:
            card_idx = card_string_to_index(board[4])
            if card_idx >= 0:
                card_tensor[7, 0] = card_idx
                card_tensor[7, 1] = 7
        
        # River (position 8)
        if len(board) >= 6:
            card_idx = card_string_to_index(board[5])
            if card_idx >= 0:
                card_tensor[8, 0] = card_idx
                card_tensor[8, 1] = 8
        
        return card_tensor
    
    def encode_actions(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode action history into (40, 7) tensor with mask."""
        action_tensor = torch.zeros((ACTION_SEQ_LEN, ACTION_INPUT_DIM), dtype=torch.float32)
        
        n_actions = min(len(self.action_history), ACTION_SEQ_LEN)
        
        # Fill from the end (most recent actions at the end)
        for i, action in enumerate(self.action_history[-ACTION_SEQ_LEN:]):
            idx = ACTION_SEQ_LEN - n_actions + i
            action_tensor[idx, 0] = action['actor_id']
            action_tensor[idx, 1] = action['event_type']
            action_tensor[idx, 2] = action['street_id']
            action_tensor[idx, 3] = action['time_id']
            action_tensor[idx, 4] = action['to_call']
            action_tensor[idx, 5] = action['pot_committed']
            action_tensor[idx, 6] = action['eff_stack']
        
        # Mask: valid actions are at the end
        action_mask = torch.zeros(ACTION_SEQ_LEN, dtype=torch.bool)
        if n_actions > 0:
            action_mask[-n_actions:] = True
        
        return action_tensor, action_mask


# =============================================================================
# Main Bot
# =============================================================================

class Player(Bot):
    """Poker bot using trained transformer model."""
    
    def __init__(self):
        """Initialize the bot."""
        self.model = None
        self.encoder = StateEncoder()
        self.device = "cpu"
        
        # Try to load model
        model_path = os.environ.get("MODEL_PATH", "model.pt")
        if os.path.exists(model_path):
            try:
                self.model = load_exported_transformer_model(model_path, device=self.device)
                print(f"Loaded model from {model_path}")
            except Exception as e:
                print(f"Warning: Could not load model: {e}")
                self.model = None
        else:
            print(f"Warning: Model file not found at {model_path}")
    
    def handle_new_round(self, game_state, round_state, active):
        """Called when a new round starts."""
        self.encoder.reset()
        
        # Determine if we're the dealer (button)
        # In skeleton: active=0 is small blind (dealer), active=1 is big blind
        self.encoder.is_dealer = (active == 0)
        
        # Store our hole cards
        self.encoder.my_hole_cards = list(round_state.hands[active])
    
    def handle_round_over(self, game_state, terminal_state, active):
        """Called when a round ends."""
        pass
    
    def get_action(self, game_state, round_state, active):
        """Get action from the model."""
        legal_actions = round_state.legal_actions()
        street = round_state.street
        
        # Handle discard phase
        if DiscardAction in legal_actions:
            return self._get_discard_action(game_state, round_state, active, legal_actions)
        
        # Handle betting phase
        return self._get_betting_action(game_state, round_state, active, legal_actions)
    
    def _get_discard_action(self, game_state, round_state, active, legal_actions):
        """Get discard action."""
        my_cards = round_state.hands[active]
        
        if self.model is None:
            # Fallback: discard lowest card
            lowest_idx = min(range(len(my_cards)), key=lambda i: card_string_to_index(my_cards[i]) % 13)
            return DiscardAction(lowest_idx)
        
        # Build tensors
        card_tensor = self.encoder.encode_cards(
            my_cards,
            list(round_state.deck),  # board cards
            self.encoder.is_dealer,
        ).unsqueeze(0)
        
        action_tensor, action_mask = self.encoder.encode_actions()
        action_tensor = action_tensor.unsqueeze(0)
        action_mask = action_mask.unsqueeze(0)
        
        # Build legal mask for discard actions
        legal_mask = torch.zeros(NUM_ACTIONS, dtype=torch.bool)
        legal_mask[ACT_DISCARD_0] = len(my_cards) > 0
        legal_mask[ACT_DISCARD_1] = len(my_cards) > 1
        legal_mask[ACT_DISCARD_2] = len(my_cards) > 2
        legal_mask = legal_mask.unsqueeze(0)
        
        # Get model prediction
        with torch.no_grad():
            logits, _ = self.model(card_tensor, action_tensor, action_mask)
            masked_logits = apply_legal_action_mask(logits, legal_mask)
            probs = F.softmax(masked_logits, dim=-1)
            action_idx = torch.argmax(probs, dim=-1).item()
        
        # Map to discard action
        if action_idx == ACT_DISCARD_0:
            discard_idx = 0
        elif action_idx == ACT_DISCARD_1:
            discard_idx = 1
        elif action_idx == ACT_DISCARD_2:
            discard_idx = 2
        else:
            # Fallback
            discard_idx = 0
        
        # Record the action
        self._record_my_action(round_state, active, action_idx)
        
        return DiscardAction(discard_idx)
    
    def _get_betting_action(self, game_state, round_state, active, legal_actions):
        """Get betting action."""
        my_cards = round_state.hands[active]
        
        if self.model is None:
            return self._fallback_action(round_state, legal_actions)
        
        # Build tensors
        card_tensor = self.encoder.encode_cards(
            my_cards,
            list(round_state.deck),
            self.encoder.is_dealer,
        ).unsqueeze(0)
        
        action_tensor, action_mask = self.encoder.encode_actions()
        action_tensor = action_tensor.unsqueeze(0)
        action_mask = action_mask.unsqueeze(0)
        
        # Build legal mask
        legal_mask = self._build_legal_mask(round_state, legal_actions)
        legal_mask = legal_mask.unsqueeze(0)
        
        # Get model prediction
        with torch.no_grad():
            logits, _ = self.model(card_tensor, action_tensor, action_mask)
            masked_logits = apply_legal_action_mask(logits, legal_mask)
            probs = F.softmax(masked_logits, dim=-1)
            action_idx = torch.argmax(probs, dim=-1).item()
        
        # Convert to skeleton action
        skeleton_action = self._model_action_to_skeleton(action_idx, round_state, legal_actions)
        
        # Record the action
        self._record_my_action(round_state, active, action_idx)
        
        return skeleton_action
    
    def _build_legal_mask(self, round_state, legal_actions) -> torch.Tensor:
        """Build legal action mask for betting."""
        mask = torch.zeros(NUM_ACTIONS, dtype=torch.bool)
        
        # Basic actions
        if FoldAction in legal_actions:
            mask[ACT_FOLD] = True
        if CheckAction in legal_actions:
            mask[ACT_CHECK] = True
        if CallAction in legal_actions:
            mask[ACT_CALL] = True
        
        # Raise actions
        if RaiseAction in legal_actions:
            min_raise, max_raise = round_state.raise_bounds()
            pot = round_state.pips[0] + round_state.pips[1]  # Total in pot
            
            # Calculate raise amounts for each action
            # ACT_RAISE_MIN = min raise
            if min_raise <= max_raise:
                mask[ACT_RAISE_MIN] = True
            
            # Pot-based raises
            raise_33 = int(pot * 0.33)
            raise_50 = int(pot * 0.5)
            raise_75 = int(pot * 0.75)
            raise_125 = int(pot * 1.25)
            raise_200 = int(pot * 2.0)
            
            if raise_33 >= min_raise and raise_33 <= max_raise:
                mask[ACT_RAISE_33] = True
            if raise_50 >= min_raise and raise_50 <= max_raise:
                mask[ACT_RAISE_50] = True
            if raise_75 >= min_raise and raise_75 <= max_raise:
                mask[ACT_RAISE_75] = True
            if raise_125 >= min_raise and raise_125 <= max_raise:
                mask[ACT_RAISE_125] = True
            if raise_200 >= min_raise and raise_200 <= max_raise:
                mask[ACT_RAISE_200] = True
            
            # All-in is always legal if raise is legal
            mask[ACT_ALLIN] = True
        
        return mask
    
    def _model_action_to_skeleton(self, action_idx: int, round_state, legal_actions):
        """Convert model action index to skeleton action."""
        if action_idx == ACT_FOLD:
            if FoldAction in legal_actions:
                return FoldAction()
            return CallAction() if CallAction in legal_actions else CheckAction()
        
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
            
            min_raise, max_raise = round_state.raise_bounds()
            pot = round_state.pips[0] + round_state.pips[1]
            
            # Calculate target raise amount
            if action_idx == ACT_RAISE_MIN:
                amount = min_raise
            elif action_idx == ACT_RAISE_33:
                amount = int(pot * 0.33)
            elif action_idx == ACT_RAISE_50:
                amount = int(pot * 0.5)
            elif action_idx == ACT_RAISE_75:
                amount = int(pot * 0.75)
            elif action_idx == ACT_RAISE_125:
                amount = int(pot * 1.25)
            elif action_idx == ACT_RAISE_200:
                amount = int(pot * 2.0)
            elif action_idx == ACT_ALLIN:
                amount = max_raise
            else:
                amount = min_raise
            
            # Clamp to valid range
            amount = max(min_raise, min(amount, max_raise))
            return RaiseAction(amount)
        
        # Fallback
        if CheckAction in legal_actions:
            return CheckAction()
        elif CallAction in legal_actions:
            return CallAction()
        else:
            return FoldAction()
    
    def _record_my_action(self, round_state, active, action_idx: int):
        """Record our action in the history."""
        # Calculate betting context
        my_pip = round_state.pips[active]
        opp_pip = round_state.pips[1 - active]
        my_stack = round_state.stacks[active]
        opp_stack = round_state.stacks[1 - active]
        
        to_call = max(0, opp_pip - my_pip)
        pot_committed = my_pip + opp_pip
        eff_stack = min(my_stack, opp_stack + to_call)
        
        street = skeleton_street_to_model_street(round_state.street)
        
        self.encoder.record_action(
            actor_id=0,  # 0 = self (ego-relative)
            event_type=action_idx,
            street=street,
            to_call=float(to_call),
            pot_committed=float(pot_committed),
            eff_stack=float(eff_stack),
        )
    
    def _fallback_action(self, round_state, legal_actions):
        """Fallback action when model is not available."""
        if CheckAction in legal_actions:
            return CheckAction()
        if CallAction in legal_actions:
            if random.random() < 0.25:
                return FoldAction()
            return CallAction()
        if RaiseAction in legal_actions:
            min_raise, max_raise = round_state.raise_bounds()
            return RaiseAction(min_raise)
        return FoldAction()


if __name__ == '__main__':
    run_bot(Player(), parse_args())