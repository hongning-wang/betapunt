"""Export JAX/Flax transformer checkpoint to PyTorch format.

Usage:
    python export_transformer_to_torch.py --ckpt ./checkpoints/step_1000 --out model.pt

Handles TPU checkpoints being loaded on CPU machines.
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict

import numpy as np
import torch

# JAX imports
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp

from python_bot_new.torch_transformer_model import (
    TossHoldemTransformerConfig,
    TossHoldemTransformerTorch,
)


def _as_numpy(x: Any) -> np.ndarray:
    """Convert JAX array to numpy."""
    arr = np.array(x, copy=True)
    if getattr(arr.dtype, "name", "") == "bfloat16":
        return arr.astype(np.float32)
    return arr


def restore_checkpoint_to_cpu(ckpt_dir: str) -> Dict[str, Any]:
    """Restore Orbax checkpoint to CPU, even if saved on TPU."""
    ckpt_dir = os.path.abspath(ckpt_dir)
    
    ckpter = ocp.PyTreeCheckpointer()
    
    # Get metadata to understand structure
    print("Reading checkpoint metadata...")
    metadata = ckpter.metadata(ckpt_dir)
    
    # Get the tree structure from metadata
    if hasattr(metadata, 'tree'):
        tree_meta = metadata.tree
    elif hasattr(metadata, 'item_metadata'):
        item_meta = metadata.item_metadata
        if hasattr(item_meta, 'tree'):
            tree_meta = item_meta.tree
        else:
            tree_meta = item_meta
    else:
        raise ValueError("Could not find tree structure in checkpoint metadata")
    
    # Build target structure with proper shapes/dtypes but on CPU
    cpu_device = jax.devices("cpu")[0]
    cpu_sharding = jax.sharding.SingleDeviceSharding(cpu_device)
    
    def meta_to_target(x):
        """Convert metadata to a ShapeDtypeStruct target."""
        if isinstance(x, ocp.metadata.ArrayMetadata):
            return jax.ShapeDtypeStruct(shape=tuple(x.shape), dtype=jnp.dtype(x.dtype))
        return x
    
    def is_array_meta(x):
        return isinstance(x, ocp.metadata.ArrayMetadata)
    
    target = jax.tree_util.tree_map(meta_to_target, tree_meta, is_leaf=is_array_meta)
    
    # Build sharding tree - all arrays go to CPU
    def to_cpu_sharding(x):
        if isinstance(x, jax.ShapeDtypeStruct):
            return cpu_sharding
        return None
    
    sharding_tree = jax.tree_util.tree_map(
        to_cpu_sharding, 
        target, 
        is_leaf=lambda y: isinstance(y, jax.ShapeDtypeStruct)
    )
    
    # Build restore args
    restore_args = ocp.checkpoint_utils.construct_restore_args(target, sharding_tree=sharding_tree)
    
    print("Restoring checkpoint to CPU...")
    restored = ckpter.restore(ckpt_dir, item=target, restore_args=restore_args)
    
    return restored


def load_flax_params_into_torch(
    model: TossHoldemTransformerTorch,
    flax_params: Dict[str, Any],
) -> None:
    """Load Flax/JAX params into PyTorch model."""

    def _get_norm_key(container: Dict[str, Any], idx: int) -> str:
        """Return LayerNorm/RMSNorm key for a given index."""
        ln_key = f"LayerNorm_{idx}"
        if ln_key in container:
            return ln_key
        return f"RMSNorm_{idx}"

    def put_linear_no_bias_dict(param_dict: Dict, torch_linear: torch.nn.Linear) -> None:
        """Load a linear layer without bias from a param dict."""
        k = _as_numpy(param_dict["kernel"])  # (in, out)
        torch_linear.weight.data.copy_(torch.from_numpy(k.T).to(torch_linear.weight.dtype))

    def put_linear_dict(param_dict: Dict, torch_linear: torch.nn.Linear) -> None:
        """Load a linear layer with bias from a param dict."""
        k = _as_numpy(param_dict["kernel"])  # (in, out)
        b = _as_numpy(param_dict["bias"])    # (out,)
        torch_linear.weight.data.copy_(torch.from_numpy(k.T).to(torch_linear.weight.dtype))
        torch_linear.bias.data.copy_(torch.from_numpy(b).to(torch_linear.bias.dtype))

    def put_norm_dict(param_dict: Dict, torch_norm: torch.nn.Module) -> None:
        """Load a LayerNorm/RMSNorm from a param dict."""
        scale = _as_numpy(param_dict["scale"])

        if hasattr(torch_norm, "weight"):
            torch_norm.weight.data.copy_(torch.from_numpy(scale).to(torch_norm.weight.dtype))
        elif hasattr(torch_norm, "scale"):
            torch_norm.scale.data.copy_(torch.from_numpy(scale).to(torch_norm.scale.dtype))
        else:
            raise AttributeError("Unsupported norm module: missing weight/scale")

        if hasattr(torch_norm, "bias"):
            if "bias" in param_dict:
                bias = _as_numpy(param_dict["bias"])
                torch_norm.bias.data.copy_(torch.from_numpy(bias).to(torch_norm.bias.dtype))
            else:
                torch_norm.bias.data.zero_()

    def put_transformer_block(flax_block: Dict[str, Any], torch_block: torch.nn.Module) -> None:
        """Load a TransformerBlock (pre-norm) from Flax params."""
        norm1_key = _get_norm_key(flax_block, 0)
        norm2_key = _get_norm_key(flax_block, 1)

        put_norm_dict(flax_block[norm1_key], torch_block.norm1)
        put_norm_dict(flax_block[norm2_key], torch_block.norm2)

        mha = flax_block["MultiHeadAttention_0"]
        put_linear_no_bias_dict(mha["Dense_0"], torch_block.attn.q_proj)
        put_linear_no_bias_dict(mha["Dense_1"], torch_block.attn.k_proj)
        put_linear_no_bias_dict(mha["Dense_2"], torch_block.attn.v_proj)
        put_linear_no_bias_dict(mha["Dense_3"], torch_block.attn.out_proj)

        put_linear_dict(flax_block["Dense_0"], torch_block.ffn[0])
        put_linear_dict(flax_block["Dense_1"], torch_block.ffn[2])

    # =========================================================================
    # CardEncoder (transformer-based)
    # =========================================================================
    card_enc = flax_params["CardEncoder_0"]

    # Card encoder embeddings
    model.card_encoder.rank_emb.weight.data.copy_(
        torch.from_numpy(_as_numpy(card_enc["Embed_0"]["embedding"])).to(model.card_encoder.rank_emb.weight.dtype)
    )
    model.card_encoder.suit_emb.weight.data.copy_(
        torch.from_numpy(_as_numpy(card_enc["Embed_1"]["embedding"])).to(model.card_encoder.suit_emb.weight.dtype)
    )
    model.card_encoder.type_emb.weight.data.copy_(
        torch.from_numpy(_as_numpy(card_enc["Embed_2"]["embedding"])).to(model.card_encoder.type_emb.weight.dtype)
    )

    # Optional transformer block for card encoding (new model)
    if "TransformerBlock_0" in card_enc:
        flax_block = card_enc["TransformerBlock_0"]
        if hasattr(model.card_encoder, "transformer"):
            if hasattr(model.card_encoder.transformer, "layers"):
                put_transformer_block(flax_block, model.card_encoder.transformer.layers[0])
            else:
                put_transformer_block(flax_block, model.card_encoder.transformer)
        elif hasattr(model.card_encoder, "transformer_block"):
            put_transformer_block(flax_block, model.card_encoder.transformer_block)
    elif "Dense_0" in card_enc and hasattr(model.card_encoder, "projection"):
        # Legacy projection-based card encoder
        put_linear_dict(card_enc["Dense_0"], model.card_encoder.projection)

    # =========================================================================
    # ActionEncoder (actor/event/street/time + continuous)
    # =========================================================================
    act_enc = flax_params["ActionEncoder_0"]

    # Action encoder embeddings
    model.action_encoder.actor_emb.weight.data.copy_(
        torch.from_numpy(_as_numpy(act_enc["Embed_0"]["embedding"])).to(model.action_encoder.actor_emb.weight.dtype)
    )
    model.action_encoder.event_emb.weight.data.copy_(
        torch.from_numpy(_as_numpy(act_enc["Embed_1"]["embedding"])).to(model.action_encoder.event_emb.weight.dtype)
    )
    model.action_encoder.street_emb.weight.data.copy_(
        torch.from_numpy(_as_numpy(act_enc["Embed_2"]["embedding"])).to(model.action_encoder.street_emb.weight.dtype)
    )
    model.action_encoder.time_emb.weight.data.copy_(
        torch.from_numpy(_as_numpy(act_enc["Embed_3"]["embedding"])).to(model.action_encoder.time_emb.weight.dtype)
    )

    # Continuous projection
    put_linear_dict(act_enc["Dense_0"], model.action_encoder.continuous_proj)

    # Action encoder main projection
    put_linear_dict(act_enc["Dense_1"], model.action_encoder.projection)

    # Action encoder transformer
    trans_enc = act_enc["TransformerEncoder_0"]
    for layer_idx in range(model.cfg.num_transformer_layers):
        layer_key = f"layer_{layer_idx}"
        flax_layer = trans_enc[layer_key]
        torch_layer = model.action_encoder.transformer.layers[layer_idx]
        put_transformer_block(flax_layer, torch_layer)

    # Transformer encoder final norm
    final_norm_key = "LayerNorm_0" if "LayerNorm_0" in trans_enc else "RMSNorm_0"
    put_norm_dict(trans_enc[final_norm_key], model.action_encoder.transformer.final_norm)

    # =========================================================================
    # Query tokens
    # =========================================================================
    query_tokens = _as_numpy(flax_params["query_tokens"])  # (num_query_tokens, emb_dim)
    model.query_tokens.data.copy_(torch.from_numpy(query_tokens).to(model.query_tokens.dtype))

    # =========================================================================
    # Cross-attention layers
    # =========================================================================
    for layer_idx in range(model.cfg.num_cross_attention_layers):
        layer_key = f"cross_attn_{layer_idx}"
        flax_cross = flax_params[layer_key]
        torch_cross = model.cross_attention_layers[layer_idx]

        # Norms
        norm0_key = _get_norm_key(flax_cross, 0)
        norm1_key = _get_norm_key(flax_cross, 1)
        norm2_key = _get_norm_key(flax_cross, 2)
        put_norm_dict(flax_cross[norm0_key], torch_cross.query_norm)
        put_norm_dict(flax_cross[norm1_key], torch_cross.context_norm)
        put_norm_dict(flax_cross[norm2_key], torch_cross.ffn_norm)

        # MultiHeadAttention_0
        mha = flax_cross["MultiHeadAttention_0"]
        put_linear_no_bias_dict(mha["Dense_0"], torch_cross.cross_attn.q_proj)
        put_linear_no_bias_dict(mha["Dense_1"], torch_cross.cross_attn.k_proj)
        put_linear_no_bias_dict(mha["Dense_2"], torch_cross.cross_attn.v_proj)
        put_linear_no_bias_dict(mha["Dense_3"], torch_cross.cross_attn.out_proj)

        # FFN: Dense_0, Dense_1
        put_linear_dict(flax_cross["Dense_0"], torch_cross.ffn[0])
        put_linear_dict(flax_cross["Dense_1"], torch_cross.ffn[2])

    # Final cross-attention norm
    if "LayerNorm_0" in flax_params:
        put_norm_dict(flax_params["LayerNorm_0"], model.cross_attn_final_norm)
    elif "RMSNorm_0" in flax_params:
        put_norm_dict(flax_params["RMSNorm_0"], model.cross_attn_final_norm)

    # Optional top-level norms
    if hasattr(model, "card_norm") and "card_norm" in flax_params:
        put_norm_dict(flax_params["card_norm"], model.card_norm)
    if hasattr(model, "action_norm") and "action_norm" in flax_params:
        put_norm_dict(flax_params["action_norm"], model.action_norm)

    # =========================================================================
    # Processing MLP
    # =========================================================================
    put_linear_dict(flax_params["Dense_0"], model.processing_fc1)
    put_linear_dict(flax_params["Dense_1"], model.processing_fc2)

    # =========================================================================
    # Output heads
    # =========================================================================
    put_linear_dict(flax_params["betting_hidden"], model.betting_hidden)
    put_linear_dict(flax_params["betting_out"], model.betting_out)
    put_linear_dict(flax_params["discard_hidden"], model.discard_hidden)
    put_linear_dict(flax_params["discard_out"], model.discard_out)
    put_linear_dict(flax_params["value_hidden"], model.value_hidden)
    put_linear_dict(flax_params["value_out"], model.value_out)


def _sizeof_mb(path: str) -> float:
    return os.path.getsize(path) / (1024 * 1024)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Orbax checkpoint directory")
    ap.add_argument("--out", default="model.pt", help="Output .pt path")
    args = ap.parse_args()

    print(f"Loading checkpoint from {args.ckpt}...")
    restored = restore_checkpoint_to_cpu(args.ckpt)
    
    # The checkpoint contains a TrainState, params are under 'params' key
    if "params" in restored:
        params = restored["params"]
    else:
        # Maybe the restored object IS the params directly
        params = restored
    
    # Debug: print top-level keys to understand structure
    print(f"Checkpoint keys: {list(params.keys())}")

    print("Creating PyTorch model...")
    cfg = TossHoldemTransformerConfig()
    model = TossHoldemTransformerTorch(cfg)

    print("Loading weights into PyTorch model...")
    load_flax_params_into_torch(model, params)
    model.eval()

    # Save
    payload = {"config": cfg.__dict__, "state_dict": model.state_dict()}
    torch.save(payload, args.out)

    # Smoke test
    print("Running smoke test...")
    with torch.no_grad():
        card = torch.zeros((2, cfg.card_seq_len, cfg.card_input_dim), dtype=torch.long)
        act = torch.zeros((2, cfg.action_seq_len, cfg.action_input_dim), dtype=torch.float32)
        mask = torch.ones((2, cfg.action_seq_len), dtype=torch.bool)
        logits, value = model(card, act, mask)
        assert logits.shape == (2, cfg.num_actions), f"Expected (2, {cfg.num_actions}), got {logits.shape}"
        assert value.shape == (2,), f"Expected (2,), got {value.shape}"
        print(f"  Logits shape: {logits.shape} ✓")
        print(f"  Value shape: {value.shape} ✓")

    print(f"\n✓ Saved model to {args.out} ({_sizeof_mb(args.out):.2f} MiB)")


if __name__ == "__main__":
    main()