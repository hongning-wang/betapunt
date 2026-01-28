"""Export JAX/Flax transformer checkpoint to PyTorch format.

Usage:
    python export_checkpoint_to_torch.py --ckpt ./checkpoints/step_26000 --out model.pt

Handles TPU checkpoints being loaded on CPU machines.
Updated for new architecture with HistoryEncoder, RMSNorm, and SwiGLU.
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

from python_history.torch_transformer_model import (
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
    
    print("Reading checkpoint metadata...")
    metadata = ckpter.metadata(ckpt_dir)
    
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
    
    cpu_device = jax.devices("cpu")[0]
    cpu_sharding = jax.sharding.SingleDeviceSharding(cpu_device)
    
    def meta_to_target(x):
        if isinstance(x, ocp.metadata.ArrayMetadata):
            return jax.ShapeDtypeStruct(shape=tuple(x.shape), dtype=jnp.dtype(x.dtype))
        return x
    
    def is_array_meta(x):
        return isinstance(x, ocp.metadata.ArrayMetadata)
    
    target = jax.tree_util.tree_map(meta_to_target, tree_meta, is_leaf=is_array_meta)
    
    def to_cpu_sharding(x):
        if isinstance(x, jax.ShapeDtypeStruct):
            return cpu_sharding
        return None
    
    sharding_tree = jax.tree_util.tree_map(
        to_cpu_sharding, 
        target, 
        is_leaf=lambda y: isinstance(y, jax.ShapeDtypeStruct)
    )
    
    restore_args = ocp.checkpoint_utils.construct_restore_args(target, sharding_tree=sharding_tree)
    
    print("Restoring checkpoint to CPU...")
    restored = ckpter.restore(ckpt_dir, item=target, restore_args=restore_args)
    
    return restored


def load_flax_params_into_torch(
    model: TossHoldemTransformerTorch,
    flax_params: Dict[str, Any],
) -> None:
    """Load Flax/JAX params into PyTorch model (new architecture with RMSNorm/SwiGLU)."""

    def put_linear_no_bias(param_dict: Dict, torch_linear: torch.nn.Linear) -> None:
        """Load a linear layer without bias."""
        k = _as_numpy(param_dict["kernel"])  # (in, out)
        torch_linear.weight.data.copy_(torch.from_numpy(k.T).to(torch_linear.weight.dtype))

    def put_linear(param_dict: Dict, torch_linear: torch.nn.Linear) -> None:
        """Load a linear layer with bias."""
        k = _as_numpy(param_dict["kernel"])
        b = _as_numpy(param_dict["bias"])
        torch_linear.weight.data.copy_(torch.from_numpy(k.T).to(torch_linear.weight.dtype))
        torch_linear.bias.data.copy_(torch.from_numpy(b).to(torch_linear.bias.dtype))

    def put_rms_norm(param_dict: Dict, torch_rms: torch.nn.Module) -> None:
        """Load RMSNorm (scale only)."""
        scale = _as_numpy(param_dict["scale"])
        torch_rms.scale.data.copy_(torch.from_numpy(scale).to(torch_rms.scale.dtype))

    def put_layer_norm(param_dict: Dict, torch_ln: torch.nn.LayerNorm) -> None:
        """Load LayerNorm (scale and bias)."""
        scale = _as_numpy(param_dict["scale"])
        bias = _as_numpy(param_dict["bias"])
        torch_ln.weight.data.copy_(torch.from_numpy(scale).to(torch_ln.weight.dtype))
        torch_ln.bias.data.copy_(torch.from_numpy(bias).to(torch_ln.bias.dtype))

    def put_embedding(flax_emb: Dict, torch_emb: torch.nn.Embedding) -> None:
        """Load an embedding layer."""
        emb = _as_numpy(flax_emb["embedding"])
        torch_emb.weight.data.copy_(torch.from_numpy(emb).to(torch_emb.weight.dtype))

    def put_swiglu_ffn(flax_ffn: Dict, torch_ffn) -> None:
        """Load SwiGLU FeedForward (gate, up, down projections)."""
        # Flax FeedForward has: Dense_0 (gate), Dense_1 (up), Dense_2 (down)
        put_linear(flax_ffn["Dense_0"], torch_ffn.gate_proj)
        put_linear(flax_ffn["Dense_1"], torch_ffn.up_proj)
        put_linear(flax_ffn["Dense_2"], torch_ffn.down_proj)

    def put_transformer_block(flax_block: Dict, torch_block) -> None:
        """Load a TransformerBlock with RMSNorm and SwiGLU."""
        # RMSNorm_0, MultiHeadAttention_0, RMSNorm_1, FeedForward_0
        put_rms_norm(flax_block["RMSNorm_0"], torch_block.norm1)
        put_rms_norm(flax_block["RMSNorm_1"], torch_block.norm2)
        
        mha = flax_block["MultiHeadAttention_0"]
        put_linear_no_bias(mha["Dense_0"], torch_block.attn.q_proj)
        put_linear_no_bias(mha["Dense_1"], torch_block.attn.k_proj)
        put_linear_no_bias(mha["Dense_2"], torch_block.attn.v_proj)
        put_linear_no_bias(mha["Dense_3"], torch_block.attn.out_proj)
        
        put_swiglu_ffn(flax_block["FeedForward_0"], torch_block.ffn)

    # =========================================================================
    # CardEncoder
    # =========================================================================
    card_enc = flax_params["CardEncoder_0"]
    print(f"  CardEncoder keys: {list(card_enc.keys())}")
    
    put_embedding(card_enc["Embed_0"], model.card_encoder.rank_emb)
    put_embedding(card_enc["Embed_1"], model.card_encoder.suit_emb)
    put_embedding(card_enc["Embed_2"], model.card_encoder.type_emb)
    
    # CardEncoder has a TransformerBlock_0
    put_transformer_block(card_enc["TransformerBlock_0"], model.card_encoder.transformer_block)

    # =========================================================================
    # ActionEncoder
    # =========================================================================
    act_enc = flax_params["ActionEncoder_0"]
    print(f"  ActionEncoder keys: {list(act_enc.keys())}")

    put_embedding(act_enc["Embed_0"], model.action_encoder.actor_emb)
    put_embedding(act_enc["Embed_1"], model.action_encoder.event_emb)
    put_embedding(act_enc["Embed_2"], model.action_encoder.street_time_emb)
    put_embedding(act_enc["Embed_3"], model.action_encoder.is_dealer_emb)

    put_linear(act_enc["Dense_0"], model.action_encoder.continuous_proj)
    put_linear(act_enc["Dense_1"], model.action_encoder.projection)

    # ActionEncoder transformer
    trans_enc = act_enc["TransformerEncoder_0"]
    print(f"  ActionEncoder TransformerEncoder keys: {list(trans_enc.keys())}")
    
    for layer_idx in range(model.cfg.num_transformer_layers):
        layer_key = f"layer_{layer_idx}"
        flax_layer = trans_enc[layer_key]
        torch_layer = model.action_encoder.transformer.layers[layer_idx]
        put_transformer_block(flax_layer, torch_layer)

    # Final norm
    put_rms_norm(trans_enc["RMSNorm_0"], model.action_encoder.transformer.final_norm)

    # =========================================================================
    # HistoryEncoder
    # =========================================================================
    hist_enc = flax_params["HistoryEncoder_0"]
    print(f"  HistoryEncoder keys: {list(hist_enc.keys())}")
    
    put_linear(hist_enc["Dense_0"], model.history_encoder.fc1)
    put_linear(hist_enc["Dense_1"], model.history_encoder.fc2)

    # =========================================================================
    # Modality Layer Norms
    # =========================================================================
    put_layer_norm(flax_params["card_norm"], model.card_norm)
    put_layer_norm(flax_params["action_norm"], model.action_norm)
    put_layer_norm(flax_params["history_norm"], model.history_norm)

    # =========================================================================
    # Query tokens
    # =========================================================================
    query_tokens = _as_numpy(flax_params["query_tokens"])
    model.query_tokens.data.copy_(torch.from_numpy(query_tokens).to(model.query_tokens.dtype))

    # =========================================================================
    # Cross-attention layers
    # =========================================================================
    for layer_idx in range(model.cfg.num_cross_attention_layers):
        layer_key = f"cross_attn_{layer_idx}"
        flax_cross = flax_params[layer_key]
        torch_cross = model.cross_attention_layers[layer_idx]
        print(f"  {layer_key} keys: {list(flax_cross.keys())}")

        # RMSNorm_0 (query), RMSNorm_1 (context), RMSNorm_2 (ffn)
        put_rms_norm(flax_cross["RMSNorm_0"], torch_cross.query_norm)
        put_rms_norm(flax_cross["RMSNorm_1"], torch_cross.context_norm)
        put_rms_norm(flax_cross["RMSNorm_2"], torch_cross.ffn_norm)

        mha = flax_cross["MultiHeadAttention_0"]
        put_linear_no_bias(mha["Dense_0"], torch_cross.cross_attn.q_proj)
        put_linear_no_bias(mha["Dense_1"], torch_cross.cross_attn.k_proj)
        put_linear_no_bias(mha["Dense_2"], torch_cross.cross_attn.v_proj)
        put_linear_no_bias(mha["Dense_3"], torch_cross.cross_attn.out_proj)

        put_swiglu_ffn(flax_cross["FeedForward_0"], torch_cross.ffn)

    # Final cross-attention norm (RMSNorm_0 at top level)
    put_rms_norm(flax_params["RMSNorm_0"], model.cross_attn_final_norm)

    # =========================================================================
    # Processing MLP
    # =========================================================================
    put_linear(flax_params["Dense_0"], model.processing_fc1)
    put_linear(flax_params["Dense_1"], model.processing_fc2)

    # =========================================================================
    # Output heads
    # =========================================================================
    put_linear(flax_params["betting_hidden"], model.betting_hidden)
    put_linear(flax_params["betting_out"], model.betting_out)
    put_linear(flax_params["discard_hidden"], model.discard_hidden)
    put_linear(flax_params["discard_out"], model.discard_out)
    put_linear(flax_params["value_hidden"], model.value_hidden)
    put_linear(flax_params["value_out"], model.value_out)


def _sizeof_mb(path: str) -> float:
    return os.path.getsize(path) / (1024 * 1024)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Orbax checkpoint directory")
    ap.add_argument("--out", default="model.pt", help="Output .pt path")
    args = ap.parse_args()

    print(f"Loading checkpoint from {args.ckpt}...")
    restored = restore_checkpoint_to_cpu(args.ckpt)
    
    if "params" in restored:
        params = restored["params"]
    else:
        params = restored
    
    print(f"Checkpoint keys: {list(params.keys())}")

    print("Creating PyTorch model...")
    cfg = TossHoldemTransformerConfig()
    model = TossHoldemTransformerTorch(cfg)

    print("Loading weights into PyTorch model...")
    load_flax_params_into_torch(model, params)
    model.eval()

    # Save
    payload = {"config": cfg.__dict__, "state_dict": model.state_dict()}
    cfg = dict(
        emb_dim=8,

        rank_emb_dim=8,
        suit_emb_dim=8,
        actor_emb_dim=8,
        event_emb_dim=8,
        street_time_emb_dim=8,
        continuous_emb_dim=8,
        is_dealer_emb_dim=8,

        ffn_dim=32,
        processing_dim=32,
        num_heads=1,
    )

    payload = {
        "model": model.state_dict(),
        "config": cfg,
    }

    torch.save(payload, args.out)

    # Smoke test
    print("Running smoke test...")
    with torch.no_grad():
        card = torch.zeros((2, cfg.card_seq_len, cfg.card_input_dim), dtype=torch.long)
        act = torch.zeros((2, cfg.action_seq_len, cfg.action_input_dim), dtype=torch.float32)
        mask = torch.ones((2, cfg.action_seq_len), dtype=torch.bool)
        hist = torch.zeros((2, cfg.history_input_dim), dtype=torch.float32)
        logits, value = model(card, act, mask, hist)
        assert logits.shape == (2, cfg.num_actions), f"Expected (2, {cfg.num_actions}), got {logits.shape}"
        assert value.shape == (2,), f"Expected (2,), got {value.shape}"
        print(f"  Logits shape: {logits.shape} ✓")
        print(f"  Value shape: {value.shape} ✓")

    print(f"\n✓ Saved model to {args.out} ({_sizeof_mb(args.out):.2f} MiB)")


if __name__ == "__main__":
    main()