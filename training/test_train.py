"""Smoke + invariants tests for train.py / game.py.

These tests are intentionally lightweight: they don't try to prove the RL algorithm
learns, only that core plumbing (encoding, stepping, rollout collection, replay,
loss/update) runs without shape / JIT / RNG / reset bugs.

Run:
  python -m pytest -q
"""

import importlib
import math

import jax
import jax.numpy as jnp


def _fresh_train_module():
    """Import train.py fresh, then downscale config for fast CPU tests."""
    import train as tr

    # Reload to avoid cross-test JIT caches capturing prior CFG.
    tr = importlib.reload(tr)

    # Downscale for tests (fast + small memory).
    tr.CFG = tr.Config(
        num_envs=8,
        rollout_len=16,
        replay_capacity=10_000,
        batch_size=512,
        num_minibatches=4,
        update_epochs=1,
        warmup_steps=0,
        recent_window=0,
        k_best=2,
        pool_max=4,
        eval_hands=32,
        add_to_pool_every_updates=999999,
        total_updates=1,
        seed=0,
    )

    # Ensure ENV is a fresh instance.
    tr.ENV = tr.TossHoldemEnv()
    return tr


def test_encode_obs_shapes_and_legal_mask():
    tr = _fresh_train_module()
    key = jax.random.PRNGKey(0)
    key, k_reset = jax.random.split(key)

    s = tr.ENV.reset(k_reset)
    aux = tr.aux_reset()
    aux = tr.aux_maybe_advance_round(aux, s.phase)

    for pid in [jnp.int32(0), jnp.int32(1)]:
        card, act, legal = tr.encode_obs(s, aux, pid)
        assert card.shape == (tr.CARD_CHANS, tr.SUITS, tr.RANKS)
        assert act.shape == (tr.ACT_CHANS, 4, tr.NB_ACTIONS)
        assert legal.shape == (tr.NB_ACTIONS,)
        assert legal.dtype == jnp.bool_
        assert bool(jnp.any(legal)), "At least one action should be legal"


def test_env_step_and_optional_reset_semantics():
    tr = _fresh_train_module()
    env = tr.ENV
    key = jax.random.PRNGKey(1)
    key, k_reset = jax.random.split(key)
    s = env.reset(k_reset)

    # Take one legal action.
    legal = env.get_legal_actions(s)
    a = int(jnp.argmax(legal))
    s_next, s_cont, done, key = tr.env_step_and_optional_reset(env, s, key, jnp.int32(a))

    assert s_next is not None
    assert s_cont is not None
    assert done.shape == ()

    # If the step ended the hand, s_cont must be a freshly reset non-terminal.
    if bool(done):
        assert bool(s_next.done)
        assert not bool(s_cont.done)


def test_terminal_rewards_zero_sum_on_sampled_terminals():
    """Sample multiple random hands; for each terminal, r0+r1 should be ~0.

    This indirectly checks the pot_total formula (pot + outstanding bets) is
    consistent with game.py, including folds.
    """
    tr = _fresh_train_module()
    env = tr.ENV
    key = jax.random.PRNGKey(2)

    def play_random_hand(k):
        k, k_reset = jax.random.split(k)
        s = env.reset(k_reset)
        aux = tr.aux_reset()
        aux = tr.aux_maybe_advance_round(aux, s.phase)

        def cond(carry):
            s, aux, k = carry
            return ~s.done

        def body(carry):
            s, aux, k = carry
            aux = tr.aux_maybe_advance_round(aux, s.phase)
            legal = env.get_legal_actions(s)
            # Random legal action
            k, sub = jax.random.split(k)
            logits = jnp.where(legal, 0.0, -1e9)
            a = jax.random.categorical(sub, logits)
            aux = tr.aux_record_action(aux, s.current_player.astype(jnp.int32), a.astype(jnp.int32), legal)
            k, k_step = jax.random.split(k)
            s = env.step(s, k_step, a.astype(jnp.int32))
            return s, aux, k

        sT, _, k = jax.lax.while_loop(cond, body, (s, aux, k))
        r0, r1 = tr.terminal_rewards(sT)
        # Use float for easier asserts.
        return sT, float(r0), float(r1)

    # Try a handful of hands.
    for i in range(30):
        key, sub = jax.random.split(key)
        sT, r0, r1 = play_random_hand(sub)
        assert bool(sT.done)
        assert math.isfinite(r0) and math.isfinite(r1)
        # Zero-sum chip deltas (allow tiny fp noise)
        assert abs((r0 + r1)) < 1e-4


def test_replay_add_batch_and_sample_smoke():
    tr = _fresh_train_module()
    key = jax.random.PRNGKey(3)

    buf = tr.replay_init(1024)

    B = 64
    # Make a simple synthetic batch.
    card = jnp.zeros((B, tr.CARD_CHANS, tr.SUITS, tr.RANKS), dtype=jnp.float32)
    act = jnp.zeros((B, tr.ACT_CHANS, 4, tr.NB_ACTIONS), dtype=jnp.float32)
    legal = jnp.ones((B, tr.NB_ACTIONS), dtype=jnp.bool_)
    action = jnp.zeros((B,), dtype=jnp.int32)
    logp_old = jnp.zeros((B,), dtype=jnp.float32)
    reward = jnp.zeros((B,), dtype=jnp.float32)
    done = jnp.zeros((B,), dtype=jnp.bool_)
    adv = jnp.zeros((B,), dtype=jnp.float32)
    ret = jnp.zeros((B,), dtype=jnp.float32)
    cs = jnp.ones((B,), dtype=jnp.int32)
    co = jnp.ones((B,), dtype=jnp.int32)
    valid = jnp.ones((B,), dtype=jnp.bool_)

    tr_batch = tr.Transition(
        card=card,
        act=act,
        card_v=card,
        act_v=act,
        legal=legal,
        action=action,
        logp_old=logp_old,
        reward=reward,
        done=done,
        adv=adv,
        ret=ret,
        contrib_self=cs,
        contrib_opp=co,
        valid=valid,
    )

    buf = tr.replay_add_batch(buf, tr_batch)
    assert int(buf.size) == B

    samp, key2 = tr.replay_sample(buf, key, 32)
    assert samp.card.shape[0] == 32
    assert samp.card_v.shape[0] == 32
    assert samp.act.shape[0] == 32
    assert samp.act_v.shape[0] == 32
    assert key2.shape == key.shape


def test_collect_rollout_and_update_smoke():
    """Full plumbing smoke test: reset -> collect -> update."""
    tr = _fresh_train_module()
    key = jax.random.PRNGKey(4)
    key, k_init = jax.random.split(key)
    ts = tr.make_train_state(k_init)

    buf = tr.replay_init(tr.CFG.replay_capacity)

    pool = tr.pool_init(ts.params)
    pool = tr.pool_add(pool, ts.params, jnp.float32(0.0))

    key, k_reset = jax.random.split(key)
    vec, key = tr.vec_reset(tr.ENV, k_reset, tr.CFG.num_envs)

    # Collect one rollout.
    buf, vec, key = tr.collect_rollout(ts.params, pool, buf, vec, key)
    expected = tr.CFG.num_envs * tr.CFG.rollout_len
    assert int(buf.size) == expected

    # Ensure aux indices are in sane range.
    assert int(jnp.max(vec.aux.slot_idx)) <= 6

    # Run one update epoch (your make_update_step wraps the full minibatch loop).
    do_update = tr.make_update_step()
    ts2, metrics, key = do_update(ts, buf, key)

    m = jax.device_get(metrics)
    # Basic sanity: finite scalars.
    for k in ["loss", "policy_loss", "value_loss", "entropy"]:
        assert math.isfinite(float(m[k]))
