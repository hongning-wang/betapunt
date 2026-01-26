"""PyTorch implementation of the Transformer-based AlphaHoldem model.

This mirrors the JAX/Flax model from training/model.py for inference.

Architecture:
- Card Encoder: (B, 9, 2) -> Embeddings (concat) -> 2-layer MLP -> (B, 9, 256)
- Action Encoder: (B, 32, 7) -> Embeddings (concat) -> Projection -> Transformer -> (B, 32, 256)
- Cross-Attention Fusion: Multiple layers of learnable queries attending to context
- Separate Output Heads: Betting Policy (10), Discard Policy (3), Value (1)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Configuration (matching training/model.py)
# =============================================================================

@dataclass(frozen=True)
class TossHoldemTransformerConfig:
    """Configuration matching the JAX model."""
    num_actions: int = 13
    num_betting_actions: int = 10
    num_discard_actions: int = 3
    card_seq_len: int = 9
    action_seq_len: int = 32
    num_card_types: int = 6  # heropriv=0, flop=1, herodiscard=2, villaindiscard=3, turn=4, river=5
    card_input_dim: int = 2
    action_input_dim: int = 7

    emb_dim: int = 256
    num_heads: int = 2
    ffn_dim: int = 512
    num_transformer_layers: int = 2
    num_cross_attention_layers: int = 2
    num_query_tokens: int = 6
    processing_dim: int = 256

    # Card encoder embedding dims (matching Flax CardEncoder)
    rank_emb_dim: int = 64
    suit_emb_dim: int = 16
    type_emb_dim: int = 48
    card_hidden_dim: int = 256

    # Action encoder embedding dims (matching Flax ActionEncoder)
    actor_emb_dim: int = 8
    event_emb_dim: int = 48
    street_time_emb_dim: int = 48
    continuous_emb_dim: int = 24


# =============================================================================
# Transformer Components
# =============================================================================

class MultiHeadAttention(nn.Module):
    """Multi-head attention matching JAX implementation."""

    def __init__(self, emb_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        projection_dim = num_heads * self.head_dim

        self.q_proj = nn.Linear(emb_dim, projection_dim, bias=False)
        self.k_proj = nn.Linear(emb_dim, projection_dim, bias=False)
        self.v_proj = nn.Linear(emb_dim, projection_dim, bias=False)
        self.out_proj = nn.Linear(projection_dim, emb_dim, bias=False)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, q_len, _ = query.shape
        kv_len = key.shape[1]

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Reshape to (B, num_heads, seq_len, head_dim)
        q = q.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, kv_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention
        scale = self.head_dim ** 0.5
        attn_weights = torch.einsum('bhqd,bhkd->bhqk', q, k) / scale

        if mask is not None:
            # mask: (B, kv_len) -> (B, 1, 1, kv_len)
            mask = mask[:, None, None, :]
            attn_weights = torch.where(mask, attn_weights, torch.tensor(-1e9, device=attn_weights.device))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.einsum('bhqk,bhkd->bhqd', attn_weights, v)

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, q_len, -1)
        output = self.out_proj(attn_output)

        return output


class TransformerBlock(nn.Module):
    """Single transformer encoder block with pre-norm."""

    def __init__(self, emb_dim: int, num_heads: int, ffn_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_dim)
        self.attn = MultiHeadAttention(emb_dim, num_heads)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, emb_dim),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm self-attention
        residual = x
        x = self.norm1(x)
        x = self.attn(x, x, x, mask)
        x = residual + x

        # Pre-norm FFN
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x

        return x


class TransformerEncoder(nn.Module):
    """Stack of transformer encoder blocks."""

    def __init__(self, emb_dim: int, num_heads: int, ffn_dim: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(emb_dim, num_heads, ffn_dim)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(emb_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        x = self.final_norm(x)
        return x


class CrossAttentionBlock(nn.Module):
    """Cross-attention block where queries attend to context."""

    def __init__(self, emb_dim: int, num_heads: int, ffn_dim: int):
        super().__init__()
        self.query_norm = nn.LayerNorm(emb_dim)
        self.context_norm = nn.LayerNorm(emb_dim)
        self.cross_attn = MultiHeadAttention(emb_dim, num_heads)
        self.ffn_norm = nn.LayerNorm(emb_dim)
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, emb_dim),
        )

    def forward(
        self,
        query_tokens: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Cross-attention
        attn_input = self.query_norm(query_tokens)
        context_normed = self.context_norm(context)
        attn_output = self.cross_attn(attn_input, context_normed, context_normed, context_mask)
        post_attn = query_tokens + attn_output

        # FFN
        ffn_input = self.ffn_norm(post_attn)
        ffn_out = self.ffn(ffn_input)
        output = post_attn + ffn_out

        return output


# =============================================================================
# Encoder Modules
# =============================================================================

class CardEncoder(nn.Module):
    """Encodes card tensor using embeddings and 2-layer MLP.

    Matches the Flax CardEncoder architecture from training/model.py.

    Input: (B, 9, 2) where each entry is [card_index, type_id].
    - card_index: 0-51 (canonized suits), -1 for invalid
    - type_id: 0-5 valid types, -1 for invalid
        0 = heropriv (all 3 hero private cards)
        1 = flop (both flop cards)
        2 = herodiscard
        3 = villaindiscard
        4 = turn
        5 = river
    """

    def __init__(self, cfg: TossHoldemTransformerConfig):
        super().__init__()
        self.cfg = cfg

        # Embeddings for rank, suit, type (concatenated, matching Flax dims)
        self.rank_emb = nn.Embedding(14, cfg.rank_emb_dim)  # 13 ranks + invalid
        self.suit_emb = nn.Embedding(5, cfg.suit_emb_dim)   # 4 suits + invalid
        self.type_emb = nn.Embedding(cfg.num_card_types + 1, cfg.type_emb_dim)  # 6 types + invalid

        # 2-layer MLP: concat -> hidden -> gelu -> emb_dim (matching Flax)
        concat_dim = cfg.rank_emb_dim + cfg.suit_emb_dim + cfg.type_emb_dim  # 64+16+48=128
        self.hidden = nn.Linear(concat_dim, cfg.card_hidden_dim)
        self.projection = nn.Linear(cfg.card_hidden_dim, cfg.emb_dim)

    def forward(self, card_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            card_tensor: (B, 9, 2) int32 tensor with [card_index, type_id]

        Returns:
            card_representation: (B, 9, emb_dim)
            valid_mask: (B, 9) boolean mask
        """
        card_index = card_tensor[..., 0].long()  # (B, 9)
        type_id = card_tensor[..., 1].long()     # (B, 9)

        # Valid mask
        valid_mask = card_index >= 0  # (B, 9)

        # Safe indices for invalid cards
        card_index_safe = torch.where(card_index >= 0, card_index, torch.tensor(52, device=card_index.device))

        # Derive rank and suit
        rank = card_index_safe % 13
        suit = card_index_safe // 13

        # Use invalid buckets for invalid cards
        is_invalid = card_index < 0
        rank_safe = torch.where(is_invalid, torch.tensor(13, device=rank.device), rank)
        suit_safe = torch.where(is_invalid, torch.tensor(4, device=suit.device), suit)
        type_safe = torch.where(type_id >= 0, type_id, torch.tensor(self.cfg.num_card_types, device=type_id.device))

        # Embeddings
        rank_e = self.rank_emb(rank_safe)
        suit_e = self.suit_emb(suit_safe)
        type_e = self.type_emb(type_safe)

        # Concatenate and 2-layer MLP (matching Flax)
        concat = torch.cat([rank_e, suit_e, type_e], dim=-1)  # (B, 9, 128)
        hidden = F.gelu(self.hidden(concat))  # (B, 9, 256)
        card_repr = self.projection(hidden)  # (B, 9, emb_dim)

        return card_repr, valid_mask


class ActionEncoder(nn.Module):
    """Encodes action history tensor using embeddings and transformer.

    Matches the Flax ActionEncoder architecture from training/model.py.

    Input: (B, 32, 7) where each action entry contains:
        [0]: actor_id (0 or 1) - categorical
        [1]: event_type (0-12) - categorical
        [2]: street_id (0-4) - categorical
        [3]: time_id (0..T-1) - categorical (unused in Flax, position computed from street)
        [4]: to_call - continuous
        [5]: pot_base - continuous
        [6]: effective_stack - continuous
    """

    def __init__(self, cfg: TossHoldemTransformerConfig):
        super().__init__()
        self.cfg = cfg

        # Categorical embeddings (matching Flax dims)
        self.actor_emb = nn.Embedding(2, cfg.actor_emb_dim)  # 8
        self.event_emb = nn.Embedding(13, cfg.event_emb_dim)  # 48
        # Combined street+position embedding (matching Flax)
        self.street_time_emb = nn.Embedding(5 * cfg.action_seq_len, cfg.street_time_emb_dim)  # 48

        # Continuous features projection
        self.continuous_proj = nn.Linear(3, cfg.continuous_emb_dim)  # 24

        # Project to emb_dim
        # 8 + 48 + 48 + 24 = 128
        concat_dim = cfg.actor_emb_dim + cfg.event_emb_dim + cfg.street_time_emb_dim + cfg.continuous_emb_dim
        self.projection = nn.Linear(concat_dim, cfg.emb_dim)

        # Transformer
        self.transformer = TransformerEncoder(
            emb_dim=cfg.emb_dim,
            num_heads=cfg.num_heads,
            ffn_dim=cfg.ffn_dim,
            num_layers=cfg.num_transformer_layers,
        )

    def forward(
        self,
        action_tensor: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            action_tensor: (B, 32, 7) float32 tensor
            action_mask: (B, 32) boolean mask

        Returns:
            action_representation: (B, 32, emb_dim)
        """
        # Extract components
        actor_id = action_tensor[..., 0].long()
        event_type = action_tensor[..., 1].long()
        street_id = action_tensor[..., 2].long()
        to_call = action_tensor[..., 4:5]
        pot_base = action_tensor[..., 5:6]
        eff_stack = action_tensor[..., 6:7]

        # Clamp indices to valid ranges
        actor_id = actor_id.clamp(0, 1)
        event_type = event_type.clamp(0, 12)
        street_id = street_id.clamp(0, 4)

        # Compute position within street (matching Flax logic)
        # Detect street boundaries: where street differs from previous position
        batch_size, seq_len = street_id.shape
        indices = torch.arange(seq_len, device=street_id.device, dtype=torch.long)

        prev_street = torch.cat([street_id[..., :1] - 1, street_id[..., :-1]], dim=-1)
        is_new_street = (street_id != prev_street)

        # Mark reset positions with their index, others with -1
        reset_markers = torch.where(is_new_street, indices, torch.tensor(-1, device=street_id.device, dtype=torch.long))

        # Cumulative maximum to propagate the last reset index forward
        # PyTorch doesn't have associative_scan, so we use a different approach
        last_reset_idx = reset_markers.clone()
        for i in range(1, seq_len):
            last_reset_idx[..., i] = torch.maximum(last_reset_idx[..., i], last_reset_idx[..., i-1])

        # Position within street = current index - index of last reset
        position_in_street = indices - last_reset_idx

        # Combined embedding index: street * 32 + position_in_street
        combined_idx = (street_id * self.cfg.action_seq_len + position_in_street).clamp(0, 5 * self.cfg.action_seq_len - 1)

        # Categorical embeddings
        actor_e = self.actor_emb(actor_id)  # (B, 32, 8)
        event_e = self.event_emb(event_type)  # (B, 32, 48)
        street_time_e = self.street_time_emb(combined_idx)  # (B, 32, 48)

        # Continuous features (log-transformed, matching Flax)
        eps = 1e-6
        pot_base_safe = torch.clamp(pot_base, min=eps)
        cont1 = torch.log1p(to_call / pot_base_safe)
        cont2 = torch.log1p(pot_base_safe)
        cont3 = torch.log1p(eff_stack / pot_base_safe)
        continuous = torch.cat([cont1, cont2, cont3], dim=-1)
        continuous_e = self.continuous_proj(continuous)  # (B, 32, 24)

        # Concatenate all embeddings (matching Flax order)
        concat = torch.cat([actor_e, event_e, street_time_e, continuous_e], dim=-1)  # (B, 32, 128)

        # Project to emb_dim
        combined = self.projection(concat)  # (B, 32, emb_dim)

        # Transformer
        encoded = self.transformer(combined, action_mask)

        return encoded


# =============================================================================
# Main Model
# =============================================================================

class TossHoldemTransformerTorch(nn.Module):
    """PyTorch implementation of the transformer-based Toss-or-Hold'em model."""

    def __init__(self, cfg: TossHoldemTransformerConfig):
        super().__init__()
        self.cfg = cfg

        # Encoders
        self.card_encoder = CardEncoder(cfg)
        self.action_encoder = ActionEncoder(cfg)

        # Learnable query tokens
        self.query_tokens = nn.Parameter(torch.randn(cfg.num_query_tokens, cfg.emb_dim) * 0.02)

        # Cross-attention layers
        self.cross_attention_layers = nn.ModuleList([
            CrossAttentionBlock(cfg.emb_dim, cfg.num_heads, cfg.ffn_dim)
            for _ in range(cfg.num_cross_attention_layers)
        ])
        self.cross_attn_final_norm = nn.LayerNorm(cfg.emb_dim)

        # Processing MLP
        flat_dim = cfg.num_query_tokens * cfg.emb_dim
        self.processing_fc1 = nn.Linear(flat_dim, 512)
        self.processing_fc2 = nn.Linear(512, cfg.processing_dim)

        # Betting policy head (actions 0-9)
        self.betting_hidden = nn.Linear(cfg.processing_dim, 128)
        self.betting_out = nn.Linear(128, cfg.num_betting_actions)

        # Discard policy head (actions 10-12)
        self.discard_hidden = nn.Linear(cfg.processing_dim, 128)
        self.discard_out = nn.Linear(128, cfg.num_discard_actions)

        # Value head
        self.value_hidden = nn.Linear(cfg.processing_dim, 128)
        self.value_out = nn.Linear(128, 1)

    def forward(
        self,
        card_tensor: torch.Tensor,
        action_tensor: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            card_tensor: (B, 9, 2) int32 - card tensor
            action_tensor: (B, 32, 7) float32 - action history
            action_mask: (B, 32) boolean - action history mask

        Returns:
            policy_logits: (B, 13)
            value: (B,)
        """
        batch_size = card_tensor.shape[0]

        # 1. Encode cards and actions
        card_repr, card_valid_mask = self.card_encoder(card_tensor)  # (B, 9, 128), (B, 9)
        action_repr = self.action_encoder(action_tensor, action_mask)  # (B, 32, 256)

        # 2. Concatenate to form context
        context = torch.cat([card_repr, action_repr], dim=1)  # (B, 41, 256)
        context_mask = torch.cat([card_valid_mask, action_mask], dim=1)  # (B, 41)

        # 3. Cross-attention with learnable queries
        fused = self.query_tokens.unsqueeze(0).expand(batch_size, -1, -1)  # (B, 6, 128)

        for cross_attn in self.cross_attention_layers:
            fused = cross_attn(fused, context, context_mask)

        fused = self.cross_attn_final_norm(fused)

        # 4. Processing MLP (using GELU to match Flax)
        flat = fused.reshape(batch_size, -1)  # (B, 768)
        processed = F.gelu(self.processing_fc1(flat))  # (B, 512)
        processed = F.gelu(self.processing_fc2(processed))  # (B, 256)

        # 5. Output heads (using GELU to match Flax)
        betting_h = F.gelu(self.betting_hidden(processed))
        betting_logits = self.betting_out(betting_h)  # (B, 10)

        discard_h = F.gelu(self.discard_hidden(processed))
        discard_logits = self.discard_out(discard_h)  # (B, 3)

        policy_logits = torch.cat([betting_logits, discard_logits], dim=-1)  # (B, 13)

        value_h = F.gelu(self.value_hidden(processed))
        value = self.value_out(value_h).squeeze(-1)  # (B,)

        return policy_logits, value


# =============================================================================
# Utility Functions
# =============================================================================

def apply_legal_action_mask(
    policy_logits: torch.Tensor,
    legal_mask: torch.Tensor,
) -> torch.Tensor:
    """Apply legal action masking to policy logits."""
    neg_inf = -1e9
    masked_logits = torch.where(legal_mask, policy_logits, torch.tensor(neg_inf, device=policy_logits.device))
    return masked_logits


def load_exported_transformer_model(
    path: str,
    device: str = "cpu",
) -> TossHoldemTransformerTorch:
    """Load an exported transformer model from disk."""
    payload = torch.load(path, map_location="cpu")
    cfg_dict = payload["config"]
    cfg = TossHoldemTransformerConfig(**cfg_dict)

    model = TossHoldemTransformerTorch(cfg)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model.to(device)


def create_default_model() -> TossHoldemTransformerTorch:
    """Create a model with default configuration."""
    cfg = TossHoldemTransformerConfig()
    return TossHoldemTransformerTorch(cfg)