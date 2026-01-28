"""PyTorch implementation of the Transformer-based Toss-or-Hold'em model.

This mirrors the JAX/Flax model from training/model.py for inference.

Architecture:
- Card Encoder: (B, 9, 2) -> Embeddings (concat) -> 1-layer Transformer -> (B, 9, emb_dim)
- Action Encoder: (B, 32, 8) -> Embeddings (concat) -> Projection -> Transformer -> (B, 32, emb_dim)
- History Encoder: (B, 64) -> Dense -> GELU -> Dense -> (B, 1, emb_dim)
- Cross-Attention Fusion: Multiple layers of learnable queries attending to context
- Separate Output Heads: Betting Policy (10), Discard Policy (3), Value (1)

Key differences from standard transformers:
- Uses RMSNorm instead of LayerNorm
- Uses SwiGLU FeedForward instead of Dense-GELU-Dense
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
    action_input_dim: int = 8  # Updated: added is_dealer
    history_input_dim: int = 64  # Opponent history features

    emb_dim: int = 128
    num_heads: int = 2
    ffn_dim: int = 256
    num_transformer_layers: int = 2
    num_cross_attention_layers: int = 2
    num_query_tokens: int = 4
    processing_dim: int = 128

    # Card encoder embedding dims
    rank_emb_dim: int = 48
    suit_emb_dim: int = 32
    # type_emb_dim computed as: emb_dim - rank_emb_dim - suit_emb_dim = 48

    # Action encoder embedding dims
    actor_emb_dim: int = 8
    event_emb_dim: int = 48
    street_time_emb_dim: int = 48
    continuous_emb_dim: int = 24
    is_dealer_emb_dim: int = 8

    # History encoder
    history_hidden_dim: int = 128


# =============================================================================
# Basic Components
# =============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.
    
    Faster than LayerNorm as it skips mean centering.
    Only has scale parameter, no bias.
    """
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMSNorm: x * rsqrt(mean(x^2) + eps) * scale
        dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (x * self.scale).to(dtype)


class SwiGLUFeedForward(nn.Module):
    """SwiGLU-style feed forward network.
    
    gate = silu(Dense(x))
    up = Dense(x)
    out = Dense(gate * up)
    """
    
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(in_dim, hidden_dim)
        self.up_proj = nn.Linear(in_dim, hidden_dim)
        self.down_proj = nn.Linear(hidden_dim, out_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


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


# =============================================================================
# Transformer Components
# =============================================================================

class TransformerBlock(nn.Module):
    """Single transformer encoder block with pre-norm RMSNorm and SwiGLU."""

    def __init__(self, emb_dim: int, num_heads: int, ffn_dim: int):
        super().__init__()
        self.norm1 = RMSNorm(emb_dim)
        self.attn = MultiHeadAttention(emb_dim, num_heads)
        self.norm2 = RMSNorm(emb_dim)
        self.ffn = SwiGLUFeedForward(emb_dim, ffn_dim, emb_dim)

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
        self.final_norm = RMSNorm(emb_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        x = self.final_norm(x)
        return x


class CrossAttentionBlock(nn.Module):
    """Cross-attention block where queries attend to context."""

    def __init__(self, emb_dim: int, num_heads: int, ffn_dim: int):
        super().__init__()
        self.query_norm = RMSNorm(emb_dim)
        self.context_norm = RMSNorm(emb_dim)
        self.cross_attn = MultiHeadAttention(emb_dim, num_heads)
        self.ffn_norm = RMSNorm(emb_dim)
        self.ffn = SwiGLUFeedForward(emb_dim, ffn_dim, emb_dim)

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
    """Encodes card tensor using embeddings and 1-layer transformer.

    Input: (B, 9, 2) where each entry is [card_index, type_id].
    - card_index: 0-51 (canonized suits), -1 for invalid
    - type_id: 0-5 valid types, -1 for invalid

    Output: (B, 9, emb_dim)
    """

    def __init__(self, cfg: TossHoldemTransformerConfig):
        super().__init__()
        self.cfg = cfg

        # Embedding dimensions
        type_emb_dim = cfg.emb_dim - cfg.rank_emb_dim - cfg.suit_emb_dim

        # Embeddings: 14 ranks (13 + invalid), 5 suits (4 + invalid), 7 types (6 + invalid)
        self.rank_emb = nn.Embedding(14, cfg.rank_emb_dim)
        self.suit_emb = nn.Embedding(5, cfg.suit_emb_dim)
        self.type_emb = nn.Embedding(cfg.num_card_types + 1, type_emb_dim)

        # 1-layer transformer for card relationships
        self.transformer_block = TransformerBlock(cfg.emb_dim, 2, 256)

    def forward(self, card_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        card_index = card_tensor[..., 0].long()
        type_id = card_tensor[..., 1].long()

        # Valid token if we have a real card
        valid_mask = card_index >= 0

        # Safe indices
        card_index_safe = torch.where(card_index >= 0, card_index, torch.tensor(52, device=card_index.device))
        rank = (card_index_safe % 13).long()
        suit = (card_index_safe // 13).long()

        is_invalid = card_index < 0
        rank_safe = torch.where(is_invalid, torch.tensor(13, device=rank.device), rank)
        suit_safe = torch.where(is_invalid, torch.tensor(4, device=suit.device), suit)
        type_safe = torch.where(type_id >= 0, type_id, torch.tensor(self.cfg.num_card_types, device=type_id.device))

        # Embeddings
        rank_e = self.rank_emb(rank_safe)
        suit_e = self.suit_emb(suit_safe)
        type_e = self.type_emb(type_safe)

        # Concatenate
        concat = torch.cat([rank_e, suit_e, type_e], dim=-1)

        # 1-layer transformer
        card_repr = self.transformer_block(concat, valid_mask)

        return card_repr, valid_mask


class ActionEncoder(nn.Module):
    """Encodes action history tensor using embeddings and transformer.

    Input: (B, 32, 8)
    Output: (B, 32, emb_dim)
    """

    def __init__(self, cfg: TossHoldemTransformerConfig):
        super().__init__()
        self.cfg = cfg

        # Categorical embeddings
        self.actor_emb = nn.Embedding(2, cfg.actor_emb_dim)  # 8
        self.event_emb = nn.Embedding(13, cfg.event_emb_dim)  # 48
        self.street_time_emb = nn.Embedding(5 * cfg.action_seq_len, cfg.street_time_emb_dim)  # 48
        self.is_dealer_emb = nn.Embedding(2, cfg.is_dealer_emb_dim)  # 8

        # Continuous features projection
        self.continuous_proj = nn.Linear(3, cfg.continuous_emb_dim)  # 24

        # Project to emb_dim: 8 + 48 + 48 + 24 + 8 = 136
        concat_dim = (cfg.actor_emb_dim + cfg.event_emb_dim + cfg.street_time_emb_dim + 
                      cfg.continuous_emb_dim + cfg.is_dealer_emb_dim)
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
        # Extract components
        actor_id = action_tensor[..., 0].long()
        event_type = action_tensor[..., 1].long()
        street_id = action_tensor[..., 2].long()
        to_call = action_tensor[..., 4:5]
        pot_base = action_tensor[..., 5:6]
        eff_stack = action_tensor[..., 6:7]
        is_dealer = action_tensor[..., 7].long()

        # Clamp indices
        actor_id = actor_id.clamp(0, 1)
        event_type = event_type.clamp(0, 12)
        street_id = street_id.clamp(0, 4)
        is_dealer = is_dealer.clamp(0, 1)

        # Compute position within street
        batch_size, seq_len = street_id.shape
        indices = torch.arange(seq_len, device=street_id.device, dtype=torch.long)

        prev_street = torch.cat([street_id[..., :1] - 1, street_id[..., :-1]], dim=-1)
        is_new_street = (street_id != prev_street)

        reset_markers = torch.where(is_new_street, indices, torch.tensor(-1, device=street_id.device, dtype=torch.long))
        last_reset_idx = reset_markers.clone()
        for i in range(1, seq_len):
            last_reset_idx[..., i] = torch.maximum(last_reset_idx[..., i], last_reset_idx[..., i-1])

        position_in_street = indices - last_reset_idx
        combined_idx = (street_id * self.cfg.action_seq_len + position_in_street).clamp(0, 5 * self.cfg.action_seq_len - 1)

        # Categorical embeddings
        actor_e = self.actor_emb(actor_id)
        event_e = self.event_emb(event_type)
        street_time_e = self.street_time_emb(combined_idx)
        is_dealer_e = self.is_dealer_emb(is_dealer)

        # Continuous features (log-transformed)
        eps = 1e-6
        pot_base_safe = torch.clamp(pot_base, min=eps)
        cont1 = torch.log1p(to_call / pot_base_safe)
        cont2 = torch.log1p(pot_base_safe)
        cont3 = torch.log1p(eff_stack / pot_base_safe)
        continuous = torch.cat([cont1, cont2, cont3], dim=-1)
        continuous_e = self.continuous_proj(continuous)

        # Concatenate all embeddings
        concat = torch.cat([actor_e, event_e, street_time_e, continuous_e, is_dealer_e], dim=-1)

        # Project to emb_dim
        combined = self.projection(concat)

        # Transformer
        encoded = self.transformer(combined, action_mask)

        return encoded


class HistoryEncoder(nn.Module):
    """Encodes opponent history statistics into a context token.
    
    Input: (B, 64) opponent history features
    Output: (B, 1, emb_dim) history representation token
    """
    
    def __init__(self, cfg: TossHoldemTransformerConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.history_input_dim, cfg.history_hidden_dim)
        self.fc2 = nn.Linear(cfg.history_hidden_dim, cfg.emb_dim)
    
    def forward(self, history_tensor: torch.Tensor) -> torch.Tensor:
        x = self.fc1(history_tensor)
        x = F.gelu(x)
        x = self.fc2(x)
        return x.unsqueeze(1)


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
        self.history_encoder = HistoryEncoder(cfg)

        # Modality layer norms
        self.card_norm = nn.LayerNorm(cfg.emb_dim)
        self.action_norm = nn.LayerNorm(cfg.emb_dim)
        self.history_norm = nn.LayerNorm(cfg.emb_dim)

        # Learnable query tokens
        self.query_tokens = nn.Parameter(torch.randn(cfg.num_query_tokens, cfg.emb_dim) * 0.02)

        # Cross-attention layers
        self.cross_attention_layers = nn.ModuleList([
            CrossAttentionBlock(cfg.emb_dim, cfg.num_heads, cfg.ffn_dim)
            for _ in range(cfg.num_cross_attention_layers)
        ])
        self.cross_attn_final_norm = RMSNorm(cfg.emb_dim)

        # Processing MLP
        flat_dim = cfg.num_query_tokens * cfg.emb_dim
        self.processing_fc1 = nn.Linear(flat_dim, 512)
        self.processing_fc2 = nn.Linear(512, cfg.processing_dim)

        # Output heads
        self.betting_hidden = nn.Linear(cfg.processing_dim, 128)
        self.betting_out = nn.Linear(128, cfg.num_betting_actions)
        self.discard_hidden = nn.Linear(cfg.processing_dim, 128)
        self.discard_out = nn.Linear(128, cfg.num_discard_actions)
        self.value_hidden = nn.Linear(cfg.processing_dim, 128)
        self.value_out = nn.Linear(128, 1)

    def forward(
        self,
        card_tensor: torch.Tensor,
        action_tensor: torch.Tensor,
        action_mask: torch.Tensor,
        history_tensor: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = card_tensor.shape[0]
        device = card_tensor.device

        # 1. Encode cards and actions
        card_repr, card_valid_mask = self.card_encoder(card_tensor)
        action_repr = self.action_encoder(action_tensor, action_mask)

        # 2. Encode history
        if history_tensor is not None:
            history_repr = self.history_encoder(history_tensor)
        else:
            history_repr = torch.zeros(batch_size, 1, self.cfg.emb_dim, device=device)
        
        history_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=device)

        # 3. Normalize and concatenate to form context
        card_repr = self.card_norm(card_repr)
        action_repr = self.action_norm(action_repr)
        history_repr = self.history_norm(history_repr)

        context = torch.cat([card_repr, action_repr, history_repr], dim=1)
        context_mask = torch.cat([card_valid_mask, action_mask, history_mask], dim=1)

        # 4. Cross-attention with learnable queries
        fused = self.query_tokens.unsqueeze(0).expand(batch_size, -1, -1)

        for cross_attn in self.cross_attention_layers:
            fused = cross_attn(fused, context, context_mask)

        fused = self.cross_attn_final_norm(fused)

        # 5. Processing MLP
        flat = fused.reshape(batch_size, -1)
        processed = F.gelu(self.processing_fc1(flat))
        processed = F.gelu(self.processing_fc2(processed))

        # 6. Output heads
        betting_h = F.gelu(self.betting_hidden(processed))
        betting_logits = self.betting_out(betting_h)

        discard_h = F.gelu(self.discard_hidden(processed))
        discard_logits = self.discard_out(discard_h)

        policy_logits = torch.cat([betting_logits, discard_logits], dim=-1)

        value_h = F.gelu(self.value_hidden(processed))
        value = self.value_out(value_h).squeeze(-1)

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