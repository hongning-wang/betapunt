from __future__ import annotations

import argparse
import os
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as oc
import torch

try:
    # If run as `python -m training.export_checkpoint_to_torch` from repo root.
    from training.torch_model import (  # type: ignore
        AlphaHoldemNetConfig,
        AlphaHoldemNetTorch,
        load_flax_params_into_torch,
        quantize_dynamic_int8,
    )
except Exception:
    # If run as `python training/export_checkpoint_to_torch.py`.
    from torch_model import (  # type: ignore
        AlphaHoldemNetConfig,
        AlphaHoldemNetTorch,
        load_flax_params_into_torch,
        quantize_dynamic_int8,
    )


def _restore_orbax_pytree_on_cpu(ckpt_dir: str) -> Dict[str, Any]:
    ckpt_dir = os.path.abspath(ckpt_dir)
    ckpter = oc.PyTreeCheckpointer()
    md = ckpter.metadata(ckpt_dir)
    meta_tree = md.item_metadata.tree

    def is_leaf(x):
        return isinstance(x, oc.metadata.ArrayMetadata)

    def to_target(x):
        if isinstance(x, oc.metadata.ArrayMetadata):
            return jax.ShapeDtypeStruct(shape=tuple(x.shape), dtype=jnp.dtype(x.dtype))
        return x

    target = jax.tree_util.tree_map(to_target, meta_tree, is_leaf=is_leaf)

    sd_sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])

    def to_sharding(x):
        if isinstance(x, jax.ShapeDtypeStruct):
            return sd_sharding
        return None

    sharding_tree = jax.tree_util.tree_map(
        to_sharding, target, is_leaf=lambda y: isinstance(y, jax.ShapeDtypeStruct)
    )
    restore_args = oc.checkpoint_utils.construct_restore_args(target, sharding_tree=sharding_tree)
    return ckpter.restore(ckpt_dir, item=target, restore_args=restore_args)


def _infer_config_from_params(params: Dict[str, Any]) -> AlphaHoldemNetConfig:
    trunk_channels = int(np.asarray(params["Conv_0"]["bias"]).shape[0])
    fc_dim = int(np.asarray(params["Dense_2"]["bias"]).shape[0])
    num_blocks = len([k for k in params.keys() if k.startswith("ResidualBlock_")])

    # Sanity checks (optional, but helpful)
    flat_in = int(np.asarray(params["Dense_2"]["kernel"]).shape[0])
    expected_flat = 4 * 13 * trunk_channels
    if flat_in != expected_flat:
        raise ValueError(f"Unexpected flatten dim: Dense_2.in={flat_in}, expected={expected_flat}")

    return AlphaHoldemNetConfig(trunk_channels=trunk_channels, num_blocks=num_blocks, fc_dim=fc_dim)


def _sizeof_mb(path: str) -> float:
    return os.path.getsize(path) / (1024 * 1024)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="training/step_1300", help="Orbax checkpoint directory")
    ap.add_argument("--out", default="training/step_1300.torch.pt", help="Output .pt path (float32)")
    ap.add_argument(
        "--out-quant",
        default="training/step_1300.torch.int8.pt",
        help="Output .pt path (dynamic int8 quantized)",
    )
    args = ap.parse_args()

    restored = _restore_orbax_pytree_on_cpu(args.ckpt)
    params = restored["params"]

    cfg = _infer_config_from_params(params)
    model = AlphaHoldemNetTorch(cfg)
    load_flax_params_into_torch(model, params)
    model.eval()

    payload = {"config": cfg.__dict__, "state_dict": model.state_dict()}
    torch.save(payload, args.out)

    qmodel = quantize_dynamic_int8(model)
    qpayload = {"config": cfg.__dict__, "state_dict": qmodel.state_dict()}
    torch.save(qpayload, args.out_quant)

    # Quick smoke test with dummy inputs (shape-only)
    with torch.no_grad():
        card = torch.zeros((2, 4, 13, cfg.card_chans), dtype=torch.float32)
        act = torch.zeros((2, 4, 13, cfg.act_chans), dtype=torch.float32)
        logits, value = model(card, act)
        qlogits, qvalue = qmodel(card, act)
        assert logits.shape == (2, cfg.nb_actions)
        assert value.shape == (2,)
        assert qlogits.shape == (2, cfg.nb_actions)
        assert qvalue.shape == (2,)

    print(f"Saved float model to {args.out} ({_sizeof_mb(args.out):.2f} MiB)")
    print(f"Saved int8  model to {args.out_quant} ({_sizeof_mb(args.out_quant):.2f} MiB)")


if __name__ == "__main__":
    main()

