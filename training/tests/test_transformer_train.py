import os
import sys

import jax.numpy as jnp

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import transformer_train as tt
from game import State


def _make_state():
    return State(
        deck=jnp.arange(52, dtype=jnp.int32),
        deck_ptr=0,
        p0_hole=jnp.array([10, 20, 30], dtype=jnp.int32),
        p1_hole=jnp.array([1, 2, 3], dtype=jnp.int32),
        board=jnp.full((6,), -1, dtype=jnp.int32),
        p0_active_mask=jnp.array([True, True, True]),
        p1_active_mask=jnp.array([True, True, True]),
        pot=0,
        p0_stack=tt.STACK_SIZE,
        p1_stack=tt.STACK_SIZE,
        p0_bet=0,
        p1_bet=0,
        last_raise_amt=2,
        current_player=0,
        is_dealer_p0=True,
        phase=tt.PHASE_PREFLOP,
        p0_acted_this_street=False,
        p1_acted_this_street=False,
        done=False,
        winner=-1,
    )


def test_phase_to_round_idx():
    print("test_phase_to_round_idx: start")
    assert int(tt._phase_to_round_idx(jnp.array(tt.PHASE_PREFLOP))) == 0
    assert int(tt._phase_to_round_idx(jnp.array(tt.PHASE_FLOP))) == 2
    assert int(tt._phase_to_round_idx(jnp.array(tt.PHASE_RIVER))) == 4
    print("test_phase_to_round_idx: ok")


def test_phase_to_street():
    print("test_phase_to_street: start")
    assert int(tt.phase_to_street(jnp.array(tt.PHASE_PREFLOP))) == tt.STREET_PREFLOP
    assert int(tt.phase_to_street(jnp.array(tt.PHASE_DISCARD_OOP))) == tt.STREET_DISCARD
    assert int(tt.phase_to_street(jnp.array(tt.PHASE_DISCARD_DLR))) == tt.STREET_DISCARD
    assert int(tt.phase_to_street(jnp.array(tt.PHASE_FLOP))) == tt.STREET_FLOP
    print("test_phase_to_street: ok")


def test_encode_cards_shapes_and_invalids():
    print("test_encode_cards_shapes_and_invalids: start")
    state = _make_state()
    card_tensor = tt.encode_cards(state, jnp.array(0))

    assert card_tensor.shape == (tt.CARD_SEQ_LEN, tt.CARD_INPUT_DIM)

    # Active hole cards: order is [hole[2], hole[1]] due to sorting logic
    assert int(card_tensor[0, 1]) == 30
    assert int(card_tensor[1, 1]) == 20

    # Board cards are unset and should remain invalid
    assert jnp.all(card_tensor[2:, 1] == -1)
    print("test_encode_cards_shapes_and_invalids: ok")


def test_encode_actions_empty_mask():
    print("test_encode_actions_empty_mask: start")
    aux = tt.aux_reset()
    action_tensor, action_mask = tt.encode_actions(aux, jnp.array(0))

    assert action_tensor.shape == (tt.ACTION_SEQ_LEN, tt.ACTION_INPUT_DIM)
    assert action_mask.shape == (tt.ACTION_SEQ_LEN,)
    assert not bool(jnp.any(action_mask))
    print("test_encode_actions_empty_mask: ok")


def test_aux_record_action_broadcast_guard():
    print("test_aux_record_action_broadcast_guard: start")
    aux = tt.aux_reset()
    state = _make_state()
    action = jnp.array(2)
    pot = jnp.array(10)
    legal_mask = jnp.zeros(tt.NUM_ACTIONS, dtype=bool)
    legal_mask = legal_mask.at[2].set(True)

    aux2 = tt.aux_record_action(aux, state, action, pot, legal_mask)
    entry = aux2.action_tensor[0]

    assert int(entry[0]) == 0
    assert float(entry[1]) == float(action)
    assert float(entry[3]) == 0.0
    print("test_aux_record_action_broadcast_guard: ok")


def test_unroll_buffer_init_shapes():
    print("test_unroll_buffer_init_shapes: start")
    buf = tt.unroll_buffer_init(capacity=2, rollout_len=4, num_envs=3)
    assert buf.card.shape == (2, 4, 3, tt.CARD_SEQ_LEN, tt.CARD_INPUT_DIM)
    assert buf.action_hist.shape == (2, 4, 3, tt.ACTION_SEQ_LEN, tt.ACTION_INPUT_DIM)
    assert int(buf.ptr) == 0
    assert int(buf.size) == 0
    print("test_unroll_buffer_init_shapes: ok")
