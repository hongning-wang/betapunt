import jax
import jax.numpy as jnp
import chex
import itertools

# --- 1. CONFIGURATION & CONSTANTS ---
NUM_CARDS = 52
STACK_SIZE = 400
SB = 1
BB = 2
MAX_ROUNDS = 1000

# Phases
PHASE_PREFLOP = 0
PHASE_DISCARD_OOP = 1
PHASE_DISCARD_DLR = 2
PHASE_FLOP = 3  
PHASE_TURN = 4
PHASE_RIVER = 5
PHASE_SHOWDOWN = 6

# Actions (13 Actions - Split Check/Call)
ACT_FOLD = 0
ACT_CHECK = 1       # Explicit Check (valid only if to_call == 0)
ACT_CALL = 2        # Explicit Call (valid only if to_call > 0)
ACT_RAISE_MIN = 3   # Match last raise
ACT_RAISE_33 = 4    # 1/3 Pot
ACT_RAISE_50 = 5    # 1/2 Pot
ACT_RAISE_75 = 6    # 3/4 Pot
ACT_RAISE_125 = 7   # 5/4 Pot
ACT_RAISE_200 = 8   # 2x Pot
ACT_ALLIN = 9
ACT_DISCARD_0 = 10
ACT_DISCARD_1 = 11
ACT_DISCARD_2 = 12

@chex.dataclass(frozen=True)
class State:
    # Card Data
    deck: chex.Array          # (52,)
    deck_ptr: int             # Index of next card
    p0_hole: chex.Array       # (3,)
    p1_hole: chex.Array       # (3,)
    board: chex.Array         # (6,)
    
    # Masks
    p0_active_mask: chex.Array # (3,) bool
    p1_active_mask: chex.Array # (3,) bool

    # Betting State
    pot: int                  # Chips from PREVIOUS streets
    p0_stack: int
    p1_stack: int
    p0_bet: int               # Chips committed THIS street
    p1_bet: int
    last_raise_amt: int       # The size of the last raise (or BB if no raises yet)
    
    # Flow
    current_player: int       # 0 or 1
    is_dealer_p0: bool
    phase: int
    p0_acted_this_street: bool
    p1_acted_this_street: bool
    
    # Meta
    done: bool
    winner: int               # 0=P0, 1=P1, 2=Tie, -1=None

# --- 2. HAND EVALUATOR ---
TIER_SF, TIER_QA, TIER_FH, TIER_FL = 8, 7, 6, 5
TIER_ST, TIER_TR, TIER_2P, TIER_1P, TIER_HC = 4, 3, 2, 1, 0
WHEEL_PATTERN = jnp.array([12, 3, 2, 1, 0]) # A, 5, 4, 3, 2

def _build_straight_table():
    import numpy as np
    tbl = np.full((1 << 13,), -1, dtype=np.int8)

    # Standard 5-long runs: high from 4..12 corresponds to ranks [high, high-1, ..., high-4]
    for high in range(4, 13):
        mask = 0
        for r in range(high - 4, high + 1):
            mask |= (1 << r)
        tbl[mask] = np.int8(high)

    # Wheel: A,2,3,4,5 i.e. ranks {12,0,1,2,3}
    wheel_mask = (1 << 12) | (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3)
    tbl[wheel_mask] = np.int8(3)  # treat as 5-high straight

    return jnp.asarray(tbl)

STRAIGHT_HIGH_FROM_MASK = _build_straight_table()

@jax.jit
def score_5_hand(hand):
    """Fast 5-card evaluator: bitmask straight lookup + top_k rank ordering.

    Returns packed int32 score: (tier << 20) | (r0<<16) | ... | r4
    where r0..r4 are tie-break ranks (higher better).
    """
    ranks = (hand % 13).astype(jnp.int32)   # (5,)
    suits = (hand // 13).astype(jnp.int32) # (5,)

    # Flush check
    flush = jnp.all(suits == suits[0])

    # Build 13-bit rank presence mask
    rank_bits = (jnp.int32(1) << ranks)  # (5,)
    rank_mask = jnp.bitwise_or.reduce(rank_bits)  # scalar int32

    # Straight check via lookup table
    straight_high = STRAIGHT_HIGH_FROM_MASK[rank_mask].astype(jnp.int32)
    strt = straight_high >= 0

    # Rank histogram (13,) using scatter-add (still fine; 13 is tiny)
    rc = jnp.zeros((13,), dtype=jnp.int32).at[ranks].add(1)
    mc = jnp.max(rc)                 # 4,3,2,1
    pc = jnp.sum(rc == 2)            # number of pairs

    # Tier selection (same logic you had) :contentReference[oaicite:2]{index=2}
    tier = jnp.select(
        [
            strt & flush,           # Straight Flush
            mc == 4,                # Quads
            (mc == 3) & (pc >= 1),  # Full House
            flush,                  # Flush
            strt,                   # Straight
            mc == 3,                # Trips
            pc >= 2,                # Two Pair
            pc == 1,                # One Pair
        ],
        [8, 7, 6, 5, 4, 3, 2, 1],
        default=0
    ).astype(jnp.int32)

    # Tie-break ranks:
    # - For straights: just need (high, then zeros)
    # - Otherwise, order ranks by (count desc, rank desc) using top_k on a 13-vector.
    #
    # Key idea: top_k is much cheaper than argsort for TPU.
    keys = rc * 32 + jnp.arange(13, dtype=jnp.int32)  # (13,), count dominates
    _, ordered = jax.lax.top_k(keys, k=5)             # ranks ordered by freq then rank

    # For straights, override to a canonical list: [high, high-1, ..., high-4] with wheel fix
    def _straight_ranks(high):
        # Wheel case: high==3 should map to [3,2,1,0,12]
        is_wheel = high == 3
        std = jnp.array([high, high-1, high-2, high-3, high-4], dtype=jnp.int32)
        wheel = jnp.array([3, 2, 1, 0, 12], dtype=jnp.int32)
        return jnp.where(is_wheel, wheel, std)

    ranks_straight = _straight_ranks(straight_high)
    # For flush/high-card, duplicates can exist, but in 5 cards sorting 5 items is cheap.
    ranks_sorted5 = jnp.sort(ranks)[::-1].astype(jnp.int32)

    # Choose final tie-break list by category
    # - Straight / straight flush: straight sequence
    # - Flush / high card: sorted ranks of the 5 cards
    # - Pair+ hands: ordered by (count, rank) from top_k(keys)
    use_sorted5 = (tier == 5) | (tier == 0)  # flush or high-card
    use_straight = (tier == 4) | (tier == 8) # straight or straight flush

    final = jnp.where(use_sorted5, ranks_sorted5, ordered)
    final = jnp.where(use_straight, ranks_straight, final)

    return (tier << 20) | (final[0] << 16) | (final[1] << 12) | (final[2] << 8) | (final[3] << 4) | final[4]


COMBINATIONS_8_5 = jnp.array(list(itertools.combinations(range(8), 5)), dtype=jnp.int32)

@jax.jit
def evaluate_best_hand(hole_cards, board_cards):
    pool = jnp.concatenate([hole_cards, board_cards])
    candidate_hands = pool[COMBINATIONS_8_5]
    scores = jax.vmap(score_5_hand)(candidate_hands)
    return jnp.max(scores)

@jax.jit
def get_showdown_result(state):
    # Pick the 2 active indices using top_k over a 3-length mask
    _, p0_idx = jax.lax.top_k(state.p0_active_mask.astype(jnp.int32), k=2)
    _, p1_idx = jax.lax.top_k(state.p1_active_mask.astype(jnp.int32), k=2)

    p0_final = state.p0_hole[p0_idx]
    p1_final = state.p1_hole[p1_idx]

    score_p0 = evaluate_best_hand(p0_final, state.board)
    score_p1 = evaluate_best_hand(p1_final, state.board)

    return jnp.sign(score_p0 - score_p1).astype(jnp.float32)

# --- 3. ENVIRONMENT CLASS ---

class TossHoldemEnv:
    def __init__(self):
        pass

    def reset(self, key: chex.PRNGKey) -> State:
        k1, k2, k3 = jax.random.split(key, 3)
        deck = jax.random.permutation(k1, jnp.arange(NUM_CARDS))
        is_dealer_p0 = jax.random.bernoulli(k2).astype(bool)
        
        p0_stack, p1_stack = STACK_SIZE, STACK_SIZE
        p0_bet = jnp.where(is_dealer_p0, SB, BB)
        p1_bet = jnp.where(is_dealer_p0, BB, SB)
        
        p0_stack -= p0_bet
        p1_stack -= p1_bet
        curr_p = jnp.where(is_dealer_p0, 0, 1)
        
        return State(
            deck=deck, deck_ptr=6,
            p0_hole=deck[0:3], p1_hole=deck[3:6],
            board=jnp.full(6, -1, dtype=jnp.int32),
            p0_active_mask=jnp.ones(3, dtype=jnp.bool_),
            p1_active_mask=jnp.ones(3, dtype=jnp.bool_),
            pot=0,
            p0_stack=p0_stack, p1_stack=p1_stack,
            p0_bet=p0_bet, p1_bet=p1_bet,
            last_raise_amt=BB, 
            current_player=curr_p,
            is_dealer_p0=is_dealer_p0,
            phase=PHASE_PREFLOP,
            p0_acted_this_street=False,
            p1_acted_this_street=False,
            done=False, winner=-1
        )

    def get_legal_actions(self, state: State) -> chex.Array:
        is_discard = (state.phase == PHASE_DISCARD_OOP) | (state.phase == PHASE_DISCARD_DLR)
        
        def _discard_mask():
            is_p0 = state.current_player == 0
            mask = jnp.where(is_p0, state.p0_active_mask, state.p1_active_mask)
            # Actions 0-9 (Betting) masked out, 10-12 (Discard) based on active cards
            return jnp.concatenate([jnp.zeros(10, dtype=bool), mask])

        def _betting_mask():
            is_p0 = state.current_player == 0
            cur_stack = jnp.where(is_p0, state.p0_stack, state.p1_stack)
            cur_bet = jnp.where(is_p0, state.p0_bet, state.p1_bet)
            opp_bet = jnp.where(is_p0, state.p1_bet, state.p0_bet)
            opp_stack = jnp.where(is_p0, state.p1_stack, state.p0_stack)
            to_call = opp_bet - cur_bet

            # Engine-style "no side pots" cap:
            # max_contribution = min(my_stack, opp_stack + to_call)
            eff_cap = jnp.minimum(cur_stack, opp_stack + to_call)
            
            # Initialize Mask (13 actions)
            # 0=Fold, 1=Check, 2=Call, 3-8=Raises, 9=AllIn, 10-12=Discards
            mask_bits = jnp.zeros(13, dtype=bool)
            
            # Fold always legal
            mask_bits = mask_bits.at[ACT_FOLD].set(to_call > 0)
            
            # Check vs Call Logic
            mask_bits = mask_bits.at[ACT_CHECK].set(to_call == 0)
            mask_bits = mask_bits.at[ACT_CALL].set(to_call > 0)
            
            # All-in is:
            # - If facing a bet: a (possibly short) call is always meaningful
            # - If not facing a bet: only legal if both players have chips (engine forbids betting when either is 0)
            can_bet_when_checked = (to_call == 0) & (cur_stack > 0) & (opp_stack > 0)
            can_all_in = (to_call > 0) | can_bet_when_checked
            mask_bits = mask_bits.at[ACT_ALLIN].set(can_all_in)
            
            # --- Min Raise Logic ---
            min_raise_increment = jnp.maximum(to_call, state.last_raise_amt)
            min_total = to_call + min_raise_increment
            can_min_raise = (eff_cap >= min_total) & (opp_stack > 0) & (cur_stack > to_call)
            mask_bits = mask_bits.at[ACT_RAISE_MIN].set(can_min_raise)
            
            # --- Fraction Raises Logic ---
            pot_base = state.pot + state.p0_bet + state.p1_bet + to_call
            
            # Increments
            inc_33 = pot_base // 3
            inc_50 = pot_base // 2
            inc_75 = (pot_base * 3) // 4
            inc_125 = (pot_base * 5) // 4
            inc_200 = pot_base * 2
            
            increments = jnp.array([inc_33, inc_50, inc_75, inc_125, inc_200])
            
            # Valid if: Increment >= last_raise_amt AND (to_call + increment) <= eff_cap
            # Also requires opponent has chips and we can cover the call (otherwise it's a call-all-in, not a raise).
            valid_fracs = (
                (increments >= state.last_raise_amt)
                & (eff_cap >= (to_call + increments))
                & (opp_stack > 0)
                & (cur_stack > to_call)
            )
            
            # Set indices 4,5,6,7,8 (which correspond to ACT_RAISE_33 through 200)
            # Note: ACT_RAISE_33 is index 4 now
            mask_bits = mask_bits.at[4:9].set(valid_fracs)
            
            return mask_bits

        return jax.lax.cond(is_discard, _discard_mask, _betting_mask)

    def step(self, state: State, key: chex.PRNGKey, action: int) -> State:
        is_discard = (state.phase == PHASE_DISCARD_OOP) | (state.phase == PHASE_DISCARD_DLR)
        return jax.lax.cond(
            is_discard,
            self._step_discard,
            self._step_betting,
            state, action, key
        )

    def _step_discard(self, state: State, action: int, key) -> State:
        # Adjustment: Discard actions are now 10, 11, 12. 
        # Subtract ACT_DISCARD_0 (10) to get 0, 1, 2.
        idx = action - ACT_DISCARD_0 
        is_p0 = state.current_player == 0
        
        hole = jnp.where(is_p0, state.p0_hole, state.p1_hole)
        card_val = hole[idx]
        
        p0_mask = jnp.where(is_p0, state.p0_active_mask.at[idx].set(False), state.p0_active_mask)
        p1_mask = jnp.where(~is_p0, state.p1_active_mask.at[idx].set(False), state.p1_active_mask)
        
        board_idx = jnp.where(state.phase == PHASE_DISCARD_OOP, 2, 3)
        new_board = state.board.at[board_idx].set(card_val)
        
        next_phase = jnp.where(state.phase == PHASE_DISCARD_OOP, PHASE_DISCARD_DLR, PHASE_FLOP)
        dealer_id = jnp.where(state.is_dealer_p0, 0, 1)
        next_player = jnp.where(next_phase == PHASE_DISCARD_DLR, dealer_id, 1 - dealer_id)
        
        return state.replace(
            board=new_board, p0_active_mask=p0_mask, p1_active_mask=p1_mask,
            phase=next_phase, current_player=next_player
        )

    def _step_betting(self, state: State, action: int, key) -> State:
        is_p0 = state.current_player == 0
        cur_stack = jnp.where(is_p0, state.p0_stack, state.p1_stack)
        cur_bet = jnp.where(is_p0, state.p0_bet, state.p1_bet)
        opp_bet = jnp.where(is_p0, state.p1_bet, state.p0_bet)
        opp_stack = jnp.where(is_p0, state.p1_stack, state.p0_stack)
        
        def _fold(s, _):
            return s.replace(done=True, winner=(1 - s.current_player))
        
        def _check(s, _):
            # Explicitly commits 0 chips
            return self._check_street_end(s.replace(
                p0_acted_this_street=jnp.where(is_p0, True, s.p0_acted_this_street),
                p1_acted_this_street=jnp.where(~is_p0, True, s.p1_acted_this_street)
            ), key)

        def _call(s, _):
            to_call = opp_bet - cur_bet
            actual_call = jnp.minimum(to_call, cur_stack)
            return self._check_street_end(s.replace(
                p0_stack=jnp.where(is_p0, s.p0_stack - actual_call, s.p0_stack),
                p1_stack=jnp.where(~is_p0, s.p1_stack - actual_call, s.p1_stack),
                p0_bet=jnp.where(is_p0, s.p0_bet + actual_call, s.p0_bet),
                p1_bet=jnp.where(~is_p0, s.p1_bet + actual_call, s.p1_bet),
                p0_acted_this_street=jnp.where(is_p0, True, s.p0_acted_this_street),
                p1_acted_this_street=jnp.where(~is_p0, True, s.p1_acted_this_street)
            ), key)

        def _raise(s, phase_idx):
            to_call = opp_bet - cur_bet
            pot_base = s.pot + s.p0_bet + s.p1_bet + to_call
            eff_cap = jnp.minimum(cur_stack, opp_stack + to_call)
            
            raise_idx = phase_idx - 3
            target_inc = jax.lax.switch(
                raise_idx,
                [
                    lambda: s.last_raise_amt,
                    lambda: pot_base // 3,
                    lambda: pot_base // 2,
                    lambda: (pot_base * 3) // 4,
                    lambda: (pot_base * 5) // 4,
                    lambda: pot_base * 2
                ]
            )
            
            # Clamp to min raise
            raise_amt = jnp.maximum(target_inc, s.last_raise_amt)
            
            total_needed = to_call + raise_amt
            
            # Engine-style cap (no side pots): can't contribute more than opponent can call.
            # Still clamp to own stack via eff_cap definition.
            actual_total = jnp.minimum(total_needed, eff_cap)
            
            new_bet_total = cur_bet + actual_total
            actual_inc = new_bet_total - opp_bet
            
            new_last_raise = jnp.where(actual_inc >= s.last_raise_amt, actual_inc, s.last_raise_amt)

            return s.replace(
                p0_stack=jnp.where(is_p0, s.p0_stack - actual_total, s.p0_stack),
                p1_stack=jnp.where(~is_p0, s.p1_stack - actual_total, s.p1_stack),
                p0_bet=jnp.where(is_p0, new_bet_total, s.p0_bet),
                p1_bet=jnp.where(~is_p0, new_bet_total, s.p1_bet),
                last_raise_amt=new_last_raise,
                current_player=1 - s.current_player,
                p0_acted_this_street=jnp.where(is_p0, True, False),
                p1_acted_this_street=jnp.where(~is_p0, True, False)
            )

        def _all_in(s, _):
            to_call = opp_bet - cur_bet
            eff_cap = jnp.minimum(cur_stack, opp_stack + to_call)

            # If we can't cover the call (or can't raise beyond it), treat this as a call-all-in.
            def _all_in_as_call(s2):
                actual_call = jnp.minimum(to_call, cur_stack)
                return self._check_street_end(s2.replace(
                    p0_stack=jnp.where(is_p0, s2.p0_stack - actual_call, s2.p0_stack),
                    p1_stack=jnp.where(~is_p0, s2.p1_stack - actual_call, s2.p1_stack),
                    p0_bet=jnp.where(is_p0, s2.p0_bet + actual_call, s2.p0_bet),
                    p1_bet=jnp.where(~is_p0, s2.p1_bet + actual_call, s2.p1_bet),
                    p0_acted_this_street=jnp.where(is_p0, True, s2.p0_acted_this_street),
                    p1_acted_this_street=jnp.where(~is_p0, True, s2.p1_acted_this_street)
                ), key)

            # Otherwise, contribute the maximum allowed (engine-style "all-in" to cap).
            def _all_in_as_raise(s2):
                actual_total = eff_cap
                new_bet = cur_bet + actual_total
                actual_inc = new_bet - opp_bet
                new_last_raise = jnp.maximum(s2.last_raise_amt, actual_inc)

                return s2.replace(
                    p0_stack=jnp.where(is_p0, s2.p0_stack - actual_total, s2.p0_stack),
                    p1_stack=jnp.where(~is_p0, s2.p1_stack - actual_total, s2.p1_stack),
                    p0_bet=jnp.where(is_p0, new_bet, s2.p0_bet),
                    p1_bet=jnp.where(~is_p0, new_bet, s2.p1_bet),
                    last_raise_amt=new_last_raise,
                    current_player=1 - s2.current_player,
                    p0_acted_this_street=jnp.where(is_p0, True, False),
                    p1_acted_this_street=jnp.where(~is_p0, True, False)
                )

            return jax.lax.cond(eff_cap <= to_call, _all_in_as_call, _all_in_as_raise, s)

        def _noop(s, _):
            return s

        return jax.lax.switch(
            action,
            [
                _fold,            # 0: Fold
                _check,           # 1: Check
                _call,            # 2: Call
                _raise,           # 3: Min
                _raise,           # 4: 1/3
                _raise,           # 5: 1/2
                _raise,           # 6: 3/4
                _raise,           # 7: 5/4
                _raise,           # 8: 2x
                _all_in,          # 9: All-In
                _noop,            # 10: Discard 0 (noop here, handled in _step_discard)
                _noop,            # 11: Discard 1
                _noop             # 12: Discard 2
            ],
            state, action
        )

    def _check_street_end(self, state: State, key) -> State:
        bets_equal = state.p0_bet == state.p1_bet
        both_acted = state.p0_acted_this_street & state.p1_acted_this_street
        is_all_in = (state.p0_stack == 0) | (state.p1_stack == 0)
        
        street_over = both_acted & (bets_equal | is_all_in)
        
        return jax.lax.cond(
            street_over,
            lambda s, k: self._advance_street(s, k),
            lambda s, _: s.replace(current_player=(1 - s.current_player)),
            state, key
        )

    def _advance_street(self, state: State, key) -> State:
        new_pot = state.pot + state.p0_bet + state.p1_bet
        
        def _reset_street_state(s, new_phase, new_player, deck_inc=0, new_board=None):
            if new_board is None: new_board = s.board
            return s.replace(
                pot=new_pot,
                phase=new_phase,
                current_player=new_player,
                deck_ptr=s.deck_ptr + deck_inc,
                board=new_board,
                p0_bet=0, p1_bet=0,
                last_raise_amt=BB,
                p0_acted_this_street=False,
                p1_acted_this_street=False
            )

        dealer_id = jnp.where(state.is_dealer_p0, 0, 1)
        opp_id = 1 - dealer_id
        
        def _goto_discard(s):
            flop_cards = jax.lax.dynamic_slice(s.deck, (s.deck_ptr, ), 2)
            new_board = s.board.at[0:2].set(flop_cards)
            return _reset_street_state(s, PHASE_DISCARD_OOP, opp_id, 2, new_board)
            
        def _goto_turn(s):
            turn_card = jax.lax.dynamic_slice(s.deck, (s.deck_ptr, ), 1)[0]
            new_board = s.board.at[4].set(turn_card)
            return _reset_street_state(s, PHASE_TURN, opp_id, 1, new_board)

        def _goto_river(s):
            river_card = jax.lax.dynamic_slice(s.deck, (s.deck_ptr, ), 1)[0]
            new_board = s.board.at[5].set(river_card)
            return _reset_street_state(s, PHASE_RIVER, opp_id, 1, new_board)

        def _goto_showdown(s):
            result = get_showdown_result(s.replace(pot=new_pot))
            winner_idx = jnp.select([result == 1.0, result == -1.0], [0, 1], default=2)
            return s.replace(pot=new_pot, phase=PHASE_SHOWDOWN, done=True, winner=winner_idx)
        
        def _goto_noop(s):
            return s

        return jax.lax.switch(
            state.phase,
            [
                _goto_discard,      # Preflop
                _goto_noop,         # Discard OOP
                _goto_noop,         # Discard DLR
                _goto_turn,         # Flop
                _goto_river,        # Turn
                _goto_showdown,     # River
                _goto_noop,         # Showdown
            ],
            state
        )