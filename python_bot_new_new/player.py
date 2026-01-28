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
import traceback
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
ACTION_SEQ_LEN = 32
CARD_INPUT_DIM = 2
ACTION_INPUT_DIM = 7
NUM_ACTIONS = 13

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


def _canonize_suits(card_indices: List[int]) -> List[int]:
    """Canonize suits based on order of first appearance.

    Maps suits so that the first suit seen becomes 0, second becomes 1, etc.
    This makes the model invariant to suit relabeling.
    """
    suit_mapping = [-1, -1, -1, -1]  # Original suit -> canonical suit
    next_canonical = 0

    # First pass: build mapping based on order of appearance
    for card_idx in card_indices:
        if card_idx < 0:
            continue
        suit = card_idx // 13
        if suit_mapping[suit] < 0:
            suit_mapping[suit] = next_canonical
            next_canonical += 1

    # Second pass: apply mapping
    result = []
    for card_idx in card_indices:
        if card_idx < 0:
            result.append(-1)
        else:
            rank = card_idx % 13
            suit = card_idx // 13
            canonical_suit = suit_mapping[suit]
            result.append(canonical_suit * 13 + rank)

    return result


def skeleton_street_to_model_street(street: int) -> int:
    """Convert skeleton street to model street ID.

    Skeleton streets: 0=preflop, 2=discard(OOP), 3=discard(dealer), 4=flop, 5=turn, 6=river
    Model streets: 0=preflop, 1=discard, 2=flop, 3=turn, 4=river
    """
    if street == 0:
        return STREET_PREFLOP
    elif street in (2, 3):  # Both discard phases
        return STREET_DISCARD
    elif street == 4:
        return STREET_FLOP
    elif street == 5:
        return STREET_TURN
    elif street == 6:
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
        discarded_idx: int,
    ) -> torch.Tensor:
        """Encode cards into (9, 2) tensor.

        Matches training preprocessing:
        1. Hero hole cards sorted by rank (lowest first), excluding discarded
        2. Flop cards sorted by rank (lowest first)
        3. All suits canonicalized by order of first appearance
        4. Semantic type_ids: 0=hero_private, 1=flop, 2=hero_discard,
           3=villain_discard, 4=turn, 5=river

        Card positions:
          0-2: hero hole cards (type_id=0)
          3-4: flop (type_id=1)
          5: hero discard (type_id=2)
          6: villain discard (type_id=3)
          7: turn (type_id=4)
          8: river (type_id=5)
        """
        # Convert hero cards to indices and track active status
        hero_indices = []
        hero_active = []
        for i, card_str in enumerate(my_cards[:3]):
            card_idx = card_string_to_index(card_str)
            is_active = (i != discarded_idx) and (card_idx >= 0)
            hero_indices.append(card_idx if card_idx >= 0 else -1)
            hero_active.append(is_active)

        # Sort hero cards by rank (with suit as tiebreaker), inactive cards last
        def hero_sort_key(idx):
            card_idx = hero_indices[idx]
            is_active = hero_active[idx]
            if not is_active or card_idx < 0:
                return 999
            rank = card_idx % 13
            suit = card_idx // 13
            return rank * 4 + suit

        hero_sorted_order = sorted(range(3), key=hero_sort_key)
        hero_sorted = [hero_indices[i] if hero_active[i] else -1 for i in hero_sorted_order]

        # Get flop cards and sort by rank
        flop_indices = []
        if len(board) >= 2:
            for card_str in board[:2]:
                card_idx = card_string_to_index(card_str)
                flop_indices.append(card_idx if card_idx >= 0 else -1)
        else:
            flop_indices = [-1, -1]

        def flop_sort_key(idx):
            card_idx = flop_indices[idx]
            if card_idx < 0:
                return 999
            rank = card_idx % 13
            suit = card_idx // 13
            return rank * 4 + suit

        flop_sorted_order = sorted(range(len(flop_indices)), key=flop_sort_key)
        flop_sorted = [flop_indices[i] for i in flop_sorted_order]
        while len(flop_sorted) < 2:
            flop_sorted.append(-1)

        # Get discard cards (hero at position 5, villain at position 6)
        # board[2] = OOP discard, board[3] = dealer discard
        hero_discard_idx = -1
        villain_discard_idx = -1
        if len(board) >= 4:
            hero_board_idx = 3 if is_dealer else 2
            villain_board_idx = 2 if is_dealer else 3
            if len(board) > hero_board_idx and board[hero_board_idx]:
                hero_discard_idx = card_string_to_index(board[hero_board_idx])
            if len(board) > villain_board_idx and board[villain_board_idx]:
                villain_discard_idx = card_string_to_index(board[villain_board_idx])

        # Get turn and river
        turn_idx = card_string_to_index(board[4]) if len(board) >= 5 else -1
        river_idx = card_string_to_index(board[5]) if len(board) >= 6 else -1

        # Build raw card indices array (before suit canonicalization)
        card_indices_raw = [
            hero_sorted[0],
            hero_sorted[1],
            hero_sorted[2],
            flop_sorted[0],
            flop_sorted[1],
            hero_discard_idx,
            villain_discard_idx,
            turn_idx,
            river_idx,
        ]

        # Canonize suits based on order of first appearance
        card_indices = _canonize_suits(card_indices_raw)

        # Semantic type_ids: 0=hero_private, 1=flop, 2=hero_discard,
        #                    3=villain_discard, 4=turn, 5=river
        type_ids_raw = [0, 0, 0, 1, 1, 2, 3, 4, 5]

        # Build tensor
        card_tensor = torch.full((CARD_SEQ_LEN, CARD_INPUT_DIM), -1, dtype=torch.long)
        for i in range(CARD_SEQ_LEN):
            if card_indices[i] >= 0:
                card_tensor[i, 0] = card_indices[i]
                card_tensor[i, 1] = type_ids_raw[i]

        return card_tensor
    
    def encode_actions(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode action history into (32, 7) tensor with mask."""
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
        self._last_round_state = None

        self._load_model()

    def _load_model(self) -> None:
        model_path = os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH)
        checkpoint_dir = os.environ.get("CHECKPOINT_DIR", DEFAULT_CHECKPOINT_DIR)

        if os.path.isdir(model_path):
            checkpoint_dir = model_path
            model_path = DEFAULT_MODEL_PATH

        if not os.path.exists(model_path) and os.path.isdir(checkpoint_dir):
            exported = self._export_checkpoint(checkpoint_dir, model_path)
            if not exported:
                print(f"Warning: Model export failed from {checkpoint_dir}")

        if os.path.exists(model_path):
            try:
                self.model = load_exported_transformer_model(model_path, device=self.device)
                print(f"Loaded model from {model_path}")
            except Exception as e:
                print(f"Warning: Could not load model: {e}")
                traceback.print_exc()
                self.model = None
        else:
            print(f"Warning: Model file not found at {model_path}")

    def _last_raise_amt(self, round_state) -> int:
        """Infer last raise increment within the current street."""
        last_raise = BIG_BLIND
        state = round_state
        while state.previous_state is not None and state.previous_state.street == state.street:
            prev = state.previous_state
            active = prev.button % 2
            prev_opp_pip = prev.pips[1 - active]
            new_active_pip = state.pips[active]
            if new_active_pip > prev_opp_pip:
                last_raise = new_active_pip - prev_opp_pip
            state = prev
        return int(last_raise)

    def _raise_bounds_training(self, round_state, active: int) -> Tuple[int, int]:
        """Compute raise bounds using training env rules."""
        to_call = max(0, round_state.pips[1 - active] - round_state.pips[active])
        cur_stack = round_state.stacks[active]
        opp_stack = round_state.stacks[1 - active]
        last_raise_amt = self._last_raise_amt(round_state)
        max_contribution = min(cur_stack, opp_stack + to_call)
        min_contribution = min(max_contribution, to_call + max(to_call, last_raise_amt))
        min_raise = round_state.pips[active] + min_contribution
        max_raise = round_state.pips[active] + max_contribution
        return int(min_raise), int(max_raise)

    def _export_checkpoint(self, checkpoint_dir: str, output_path: str) -> bool:
        """Export a JAX/Flax checkpoint to a Torch model file."""
        try:
            from export_checkpoint_to_torch import (
                restore_checkpoint_to_cpu,
                load_flax_params_into_torch,
            )
            from torch_transformer_model import (
                TossHoldemTransformerConfig,
                TossHoldemTransformerTorch,
            )
        except Exception as exc:
            print(f"Warning: Could not import exporter dependencies: {exc}")
            traceback.print_exc()
            return False

        try:
            print(f"Exporting checkpoint from {checkpoint_dir} to {output_path}...")
            restored = restore_checkpoint_to_cpu(checkpoint_dir)
            params = restored["params"] if isinstance(restored, dict) and "params" in restored else restored

            cfg = TossHoldemTransformerConfig()
            model = TossHoldemTransformerTorch(cfg)
            load_flax_params_into_torch(model, params)
            model.eval()

            payload = {"config": cfg.__dict__, "state_dict": model.state_dict()}
            torch.save(payload, output_path)
            print("Export complete.")
            return True
        except Exception as exc:
            print(f"Warning: Failed to export checkpoint: {exc}")
            traceback.print_exc()
            return False
    
    def handle_new_round(self, game_state, round_state, active):
        """Called when a new round starts."""
        self.encoder.reset()
        
        # Determine if we're the dealer (button)
        # In skeleton: active=0 is small blind (dealer), active=1 is big blind
        self.encoder.is_dealer = (active == 0)
        
        # Store our hole cards
        self.encoder.my_hole_cards = list(round_state.hands[active])
        self._last_round_state = round_state
    
    def handle_round_over(self, game_state, terminal_state, active):
        """Called when a round ends."""
        self._last_round_state = None
    
    def get_action(self, game_state, round_state, active):
        """Get action from the model."""
        self._sync_action_history(round_state, active)
        legal_actions = round_state.legal_actions()
        street = round_state.street
        
        # Handle discard phase
        if DiscardAction in legal_actions:
            return self._get_discard_action(game_state, round_state, active, legal_actions)
        if street in (2, 3):
            # Discard rounds: the non-discarding player should only check.
            return CheckAction()
        
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
            self.encoder.my_hole_cards,
            list(round_state.board),
            self.encoder.is_dealer,
            self.encoder.my_discarded_idx,
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
            action_idx = torch.argmax(probs[0]).item()

        # Debug logging
        print(f"[DISCARD] logits: {logits[0].tolist()}")
        print(f"[DISCARD] masked_logits: {masked_logits[0].tolist()}")
        print(f"[DISCARD] probs: {probs[0].tolist()}")
        print(f"[DISCARD] chosen: {action_idx}")

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

        self.encoder.my_discarded_idx = discard_idx
        
        # Record the action
        self._record_my_action(round_state, active, action_idx)
        self._advance_last_state(round_state, DiscardAction(discard_idx))
        
        return DiscardAction(discard_idx)
    
    def _get_betting_action(self, game_state, round_state, active, legal_actions):
        """Get betting action."""
        my_cards = round_state.hands[active]
        
        if self.model is None:
            return self._fallback_action(round_state, legal_actions)
        
        to_call = max(0, round_state.pips[1 - active] - round_state.pips[active])
        if to_call == 0 and FoldAction in legal_actions:
            # Never fold for free; treat fold as illegal when checking is possible.
            legal_actions = set(legal_actions)
            legal_actions.discard(FoldAction)

        # Build tensors
        card_tensor = self.encoder.encode_cards(
            self.encoder.my_hole_cards,
            list(round_state.board),
            self.encoder.is_dealer,
            self.encoder.my_discarded_idx,
        ).unsqueeze(0)
        
        action_tensor, action_mask = self.encoder.encode_actions()
        action_tensor = action_tensor.unsqueeze(0)
        action_mask = action_mask.unsqueeze(0)
        
        # Build legal mask
        legal_mask = self._build_legal_mask(round_state, legal_actions, active)
        legal_mask = legal_mask.unsqueeze(0)
        
        # Get model prediction
        with torch.no_grad():
            logits, _ = self.model(card_tensor, action_tensor, action_mask)
            masked_logits = apply_legal_action_mask(logits, legal_mask)
            probs = F.softmax(masked_logits, dim=-1)
            action_idx = torch.argmax(probs[0]).item()

        # Debug logging
        action_names = ['FOLD', 'CHECK', 'CALL', 'RAISE_MIN', 'RAISE_33', 'RAISE_50', 'RAISE_75', 'RAISE_125', 'RAISE_200', 'ALLIN', 'DISC_0', 'DISC_1', 'DISC_2']
        print(f"[BET] street={round_state.street} legal_mask={legal_mask[0].tolist()}")
        print(f"[BET] logits: {[f'{action_names[i]}:{v:.3f}' for i, v in enumerate(logits[0].tolist())]}")
        print(f"[BET] probs:  {[f'{action_names[i]}:{v:.3f}' for i, v in enumerate(probs[0].tolist())]}")
        print(f"[BET] chosen: {action_names[action_idx]} ({action_idx})")

        # Convert to skeleton action
        skeleton_action = self._model_action_to_skeleton(action_idx, round_state, legal_actions, active)
        
        # Record the action
        self._record_my_action(round_state, active, action_idx)
        self._advance_last_state(round_state, skeleton_action)
        
        return skeleton_action
    
    def _build_legal_mask(self, round_state, legal_actions, active: int) -> torch.Tensor:
        """Build legal action mask for betting."""
        mask = torch.zeros(NUM_ACTIONS, dtype=torch.bool)
        to_call = max(0, round_state.pips[1 - active] - round_state.pips[active])
        total_committed = (STARTING_STACK * 2) - (round_state.stacks[0] + round_state.stacks[1])
        pot_base = total_committed + to_call
        cur_stack = round_state.stacks[active]
        opp_stack = round_state.stacks[1 - active]
        eff_cap = min(cur_stack, opp_stack + to_call)
        last_raise_amt = self._last_raise_amt(round_state)
        
        # Basic actions
        if FoldAction in legal_actions and to_call > 0:
            mask[ACT_FOLD] = True
        if CheckAction in legal_actions:
            mask[ACT_CHECK] = True
        if CallAction in legal_actions:
            mask[ACT_CALL] = True
        
        # Raise actions
        if RaiseAction in legal_actions:
            min_raise, max_raise = self._raise_bounds_training(round_state, active)
            
            # Calculate raise amounts for each action
            # ACT_RAISE_MIN = min raise
            if min_raise <= max_raise:
                mask[ACT_RAISE_MIN] = True
            
            # Pot-based raises
            inc_33 = pot_base // 3
            inc_50 = pot_base // 2
            inc_75 = (pot_base * 3) // 4
            inc_125 = (pot_base * 5) // 4
            inc_200 = pot_base * 2

            raise_33 = round_state.pips[active] + to_call + inc_33
            raise_50 = round_state.pips[active] + to_call + inc_50
            raise_75 = round_state.pips[active] + to_call + inc_75
            raise_125 = round_state.pips[active] + to_call + inc_125
            raise_200 = round_state.pips[active] + to_call + inc_200
            
            if inc_33 >= last_raise_amt and min_raise <= raise_33 <= max_raise:
                mask[ACT_RAISE_33] = True
            if inc_50 >= last_raise_amt and min_raise <= raise_50 <= max_raise:
                mask[ACT_RAISE_50] = True
            if inc_75 >= last_raise_amt and min_raise <= raise_75 <= max_raise:
                mask[ACT_RAISE_75] = True
            if inc_125 >= last_raise_amt and min_raise <= raise_125 <= max_raise:
                mask[ACT_RAISE_125] = True
            if inc_200 >= last_raise_amt and min_raise <= raise_200 <= max_raise:
                mask[ACT_RAISE_200] = True
            
            # All-in is always legal if raise is legal
            can_bet_when_checked = (to_call == 0) and (cur_stack > 0) and (opp_stack > 0)
            can_all_in = (to_call > 0) or can_bet_when_checked
            if can_all_in and eff_cap > 0:
                mask[ACT_ALLIN] = True
        
        return mask
    
    def _model_action_to_skeleton(self, action_idx: int, round_state, legal_actions, active: int):
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

        # Fallback for unexpected indices (e.g., discard actions during betting)
        if CheckAction in legal_actions:
            return CheckAction()
        return CallAction() if CallAction in legal_actions else FoldAction()
        
    
    def _record_my_action(self, round_state, active, action_idx: int):
        """Record our action in the history."""
        self._record_action(round_state, active, 0, action_idx)

    def _record_action(self, round_state, active, actor_id: int, action_idx: int):
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

        self.encoder.record_action(
            actor_id=actor_id,
            event_type=action_idx,
            street=street,
            to_call=float(to_call),
            pot_committed=float(pot_base),
            eff_stack=float(eff_stack),
        )

    def _compute_raise_amount(self, round_state, active: int, action_idx: int) -> int:
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
            self._record_action(prev, active, actor_id, action_idx)

        self._last_round_state = round_state

    def _infer_action_index(self, prev_state, next_state, active) -> Optional[int]:
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

        # Skip check actions during discard phases (streets 2, 3)
        # Training environment doesn't have these - only the discarding player acts
        # Street 2: player 1 (OOP) discards, player 0 checks
        # Street 3: player 0 (dealer) discards, player 1 checks
        # Skip if actor is the non-discarding player (actor == street % 2)
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

    def _map_raise_amount_to_action_idx(self, round_state, active: int, amount: int) -> int:
        """Map an observed raise amount to the closest model action index."""
        candidates = [
            ACT_RAISE_MIN,
            ACT_RAISE_33,
            ACT_RAISE_50,
            ACT_RAISE_75,
            ACT_RAISE_125,
            ACT_RAISE_200,
            ACT_ALLIN,
        ]
        amounts = {idx: self._compute_raise_amount(round_state, active, idx) for idx in candidates}
        for idx in candidates:
            if amounts[idx] == amount:
                return idx
        best = min(candidates, key=lambda idx: (abs(amounts[idx] - amount), idx))
        return best

    def _advance_last_state(self, round_state, action):
        """Advance internal pointer to keep action sync consistent.

        NOTE: We do NOT call proceed() here because it mutates shared state
        (hands/board lists). The runner will process the engine's response
        and update state correctly. We just mark current state as seen.
        """
        self._last_round_state = round_state
    
    def _fallback_action(self, round_state, legal_actions):
        """Fallback action when model is not available."""
        raise RuntimeError(
            "Model unavailable; refusing to take fallback action. "
            "Check model load/export logs for the root cause."
        )

if __name__ == '__main__':
    run_bot(Player(), parse_args())