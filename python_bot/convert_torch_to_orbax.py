"""Convert PyTorch model.pt to Orbax checkpoint format.

Usage:
    uv run python convert_torch_to_orbax.py --pt model.pt --out ./checkpoint_dir

This is the reverse of export_checkpoint_to_torch.py.
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


def load_torch_model(pt_path: str) -> Dict[str, Any]:
    """Load PyTorch model from .pt file."""
    payload = torch.load(pt_path, map_location="cpu")
    config = payload["config"]
    state_dict = payload["state_dict"]
    return config, state_dict


def torch_to_numpy(t: torch.Tensor) -> np.ndarray:
    """Convert PyTorch tensor to numpy array."""
    return t.detach().cpu().numpy()


def build_flax_params_from_torch(
    config: Dict[str, Any],
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, Any]:
    """Build Flax parameter structure from PyTorch state dict.

    PyTorch Linear weight shape: (out_features, in_features)
    Flax Dense kernel shape: (in_features, out_features)
    So we need to transpose linear weights.
    """
    params = {}

    # Helper functions
    def get_linear_params(prefix: str, has_bias: bool = True) -> Dict[str, np.ndarray]:
        """Extract linear layer params, transposing weight."""
        weight = torch_to_numpy(state_dict[f"{prefix}.weight"])
        result = {"kernel": weight.T}  # Transpose for Flax
        if has_bias:
            bias = torch_to_numpy(state_dict[f"{prefix}.bias"])
            result["bias"] = bias
        return result

    def get_embedding_params(prefix: str) -> Dict[str, np.ndarray]:
        """Extract embedding params (no transpose needed)."""
        weight = torch_to_numpy(state_dict[f"{prefix}.weight"])
        return {"embedding": weight}

    def get_layer_norm_params(prefix: str) -> Dict[str, np.ndarray]:
        """Extract LayerNorm params (weight->scale, bias->bias)."""
        weight = torch_to_numpy(state_dict[f"{prefix}.weight"])
        bias = torch_to_numpy(state_dict[f"{prefix}.bias"])
        return {"scale": weight, "bias": bias}

    # ==========================================================================
    # CardEncoder_0
    # ==========================================================================
    card_enc = {}
    card_enc["Embed_0"] = get_embedding_params("card_encoder.rank_emb")
    card_enc["Embed_1"] = get_embedding_params("card_encoder.suit_emb")
    card_enc["Embed_2"] = get_embedding_params("card_encoder.type_emb")

    # Card encoder has a 2-layer structure in PyTorch version
    # Check if hidden layer exists
    if "card_encoder.hidden.weight" in state_dict:
        card_enc["Dense_0"] = get_linear_params("card_encoder.hidden")
        card_enc["Dense_1"] = get_linear_params("card_encoder.projection")
    else:
        card_enc["Dense_0"] = get_linear_params("card_encoder.projection")

    params["CardEncoder_0"] = card_enc

    # ==========================================================================
    # ActionEncoder_0
    # ==========================================================================
    act_enc = {}
    act_enc["Embed_0"] = get_embedding_params("action_encoder.actor_emb")
    act_enc["Embed_1"] = get_embedding_params("action_encoder.event_emb")

    # Check if we have combined street_time_emb or separate
    if "action_encoder.street_time_emb.weight" in state_dict:
        act_enc["Embed_2"] = get_embedding_params("action_encoder.street_time_emb")
    else:
        # Separate street and time embeddings
        act_enc["Embed_2"] = get_embedding_params("action_encoder.street_emb")
        act_enc["Embed_3"] = get_embedding_params("action_encoder.time_emb")

    act_enc["Dense_0"] = get_linear_params("action_encoder.continuous_proj")
    act_enc["Dense_1"] = get_linear_params("action_encoder.projection")

    # Transformer encoder
    trans_enc = {}
    num_layers = config.get("num_transformer_layers", 2)

    for layer_idx in range(num_layers):
        layer_key = f"layer_{layer_idx}"
        torch_prefix = f"action_encoder.transformer.layers.{layer_idx}"

        flax_layer = {}
        flax_layer["LayerNorm_0"] = get_layer_norm_params(f"{torch_prefix}.norm1")
        flax_layer["LayerNorm_1"] = get_layer_norm_params(f"{torch_prefix}.norm2")

        # MultiHeadAttention
        mha = {}
        mha["Dense_0"] = get_linear_params(f"{torch_prefix}.attn.q_proj", has_bias=False)
        mha["Dense_1"] = get_linear_params(f"{torch_prefix}.attn.k_proj", has_bias=False)
        mha["Dense_2"] = get_linear_params(f"{torch_prefix}.attn.v_proj", has_bias=False)
        mha["Dense_3"] = get_linear_params(f"{torch_prefix}.attn.out_proj", has_bias=False)
        flax_layer["MultiHeadAttention_0"] = mha

        # FFN
        flax_layer["Dense_0"] = get_linear_params(f"{torch_prefix}.ffn.0")
        flax_layer["Dense_1"] = get_linear_params(f"{torch_prefix}.ffn.2")

        trans_enc[layer_key] = flax_layer

    # Final norm
    trans_enc["LayerNorm_0"] = get_layer_norm_params("action_encoder.transformer.final_norm")
    act_enc["TransformerEncoder_0"] = trans_enc

    params["ActionEncoder_0"] = act_enc

    # ==========================================================================
    # Query tokens
    # ==========================================================================
    query_tokens = torch_to_numpy(state_dict["query_tokens"])
    params["query_tokens"] = query_tokens

    # ==========================================================================
    # Cross-attention layers
    # ==========================================================================
    num_cross_layers = config.get("num_cross_attention_layers", 2)

    for layer_idx in range(num_cross_layers):
        layer_key = f"cross_attn_{layer_idx}"
        torch_prefix = f"cross_attention_layers.{layer_idx}"

        flax_cross = {}
        flax_cross["LayerNorm_0"] = get_layer_norm_params(f"{torch_prefix}.query_norm")
        flax_cross["LayerNorm_1"] = get_layer_norm_params(f"{torch_prefix}.context_norm")
        flax_cross["LayerNorm_2"] = get_layer_norm_params(f"{torch_prefix}.ffn_norm")

        # Cross-attention MultiHeadAttention
        mha = {}
        mha["Dense_0"] = get_linear_params(f"{torch_prefix}.cross_attn.q_proj", has_bias=False)
        mha["Dense_1"] = get_linear_params(f"{torch_prefix}.cross_attn.k_proj", has_bias=False)
        mha["Dense_2"] = get_linear_params(f"{torch_prefix}.cross_attn.v_proj", has_bias=False)
        mha["Dense_3"] = get_linear_params(f"{torch_prefix}.cross_attn.out_proj", has_bias=False)
        flax_cross["MultiHeadAttention_0"] = mha

        # FFN
        flax_cross["Dense_0"] = get_linear_params(f"{torch_prefix}.ffn.0")
        flax_cross["Dense_1"] = get_linear_params(f"{torch_prefix}.ffn.2")

        params[layer_key] = flax_cross

    # Final cross-attention norm
    params["LayerNorm_0"] = get_layer_norm_params("cross_attn_final_norm")

    # ==========================================================================
    # Processing MLP
    # ==========================================================================
    params["Dense_0"] = get_linear_params("processing_fc1")
    params["Dense_1"] = get_linear_params("processing_fc2")

    # ==========================================================================
    # Output heads
    # ==========================================================================
    params["betting_hidden"] = get_linear_params("betting_hidden")
    params["betting_out"] = get_linear_params("betting_out")
    params["discard_hidden"] = get_linear_params("discard_hidden")
    params["discard_out"] = get_linear_params("discard_out")
    params["value_hidden"] = get_linear_params("value_hidden")
    params["value_out"] = get_linear_params("value_out")

    return params


def convert_to_bfloat16(params: Dict[str, Any]) -> Dict[str, Any]:
    """Convert all float arrays in params to bfloat16 (as numpy arrays).

    We keep everything as numpy arrays to avoid embedding device-specific
    sharding information in the checkpoint.
    """
    import ml_dtypes  # For bfloat16 support in numpy

    def convert_leaf(x):
        if isinstance(x, np.ndarray) and np.issubdtype(x.dtype, np.floating):
            # Convert to bfloat16 using ml_dtypes (numpy-compatible bfloat16)
            return x.astype(ml_dtypes.bfloat16)
        return x

    return jax.tree_util.tree_map(convert_leaf, params)


def save_orbax_checkpoint(params: Dict[str, Any], output_dir: str) -> None:
    """Save params as Orbax checkpoint without device-specific sharding.

    We save numpy arrays directly, which won't embed JAX device info.
    This allows the checkpoint to be loaded on any device (CPU/TPU).
    """
    output_dir = os.path.abspath(output_dir)

    # Create checkpoint structure matching training (params under "params" key)
    checkpoint_data = {"params": params}

    # Remove existing checkpoint if present
    if os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir)

    # Save using PyTreeCheckpointer WITHOUT save_args_from_target
    # This avoids embedding device-specific sharding info
    ckpter = ocp.PyTreeCheckpointer()
    ckpter.save(output_dir, checkpoint_data)

    print(f"Saved Orbax checkpoint to {output_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert PyTorch model to Orbax checkpoint")
    ap.add_argument("--pt", required=True, help="Input PyTorch .pt file")
    ap.add_argument("--out", required=True, help="Output Orbax checkpoint directory")
    ap.add_argument("--bfloat16", action="store_true", help="Convert weights to bfloat16")
    args = ap.parse_args()

    print(f"Loading PyTorch model from {args.pt}...")
    config, state_dict = load_torch_model(args.pt)

    print(f"Model config: {config}")
    print(f"State dict keys: {list(state_dict.keys())[:10]}...")

    print("Converting to Flax params...")
    params = build_flax_params_from_torch(config, state_dict)

    if args.bfloat16:
        print("Converting to bfloat16...")
        params = convert_to_bfloat16(params)

    print("Saving Orbax checkpoint...")
    save_orbax_checkpoint(params, args.out)

    print("Done!")


if __name__ == "__main__":
    main()
