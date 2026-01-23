import os
import sys

import jax
import jax.numpy as jnp

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import model


def test_create_action_mask():
    action_tensor = jnp.zeros((2, model.ACTION_SEQ_LEN, model.ACTION_INPUT_DIM), dtype=jnp.float32)
    action_tensor = action_tensor.at[0, 0, 3].set(1.0)

    mask = model.create_action_mask(action_tensor)

    assert mask.shape == (2, model.ACTION_SEQ_LEN)
    assert bool(mask[0, 0])
    assert not bool(mask[0, 1])
    assert not bool(jnp.any(mask[1]))


def test_apply_legal_action_mask():
    logits = jnp.array([[1.0, 2.0, 3.0]], dtype=jnp.float32)
    legal = jnp.array([[True, False, True]])

    masked = model.apply_legal_action_mask(logits, legal)

    assert masked.shape == logits.shape
    assert float(masked[0, 0]) == float(logits[0, 0])
    assert float(masked[0, 1]) < -1e8
    assert float(masked[0, 2]) == float(logits[0, 2])


def test_model_forward_shapes():
    key = jax.random.PRNGKey(0)
    net = model.TossHoldemModel()

    card_tensor = jnp.zeros((2, model.CARD_SEQ_LEN, model.CARD_INPUT_DIM), dtype=jnp.int32)
    action_tensor = jnp.zeros((2, model.ACTION_SEQ_LEN, model.ACTION_INPUT_DIM), dtype=jnp.float32)
    action_mask = jnp.zeros((2, model.ACTION_SEQ_LEN), dtype=jnp.bool_)

    params = net.init(key, card_tensor, action_tensor, action_mask)
    policy_logits, value = net.apply(params, card_tensor, action_tensor, action_mask)

    assert policy_logits.shape == (2, model.NUM_ACTIONS)
    assert value.shape == (2,)
