"""PyTorch implementation of the Transformer-based AlphaHoldem model.

This mirrors the JAX/Flax model from training/model.py for inference.

Architecture:
- Card Encoder: (B, 9, 2) -> Embeddings (concat) -> Self-Attention (no FFN) -> (B, 9, 128)
- Action Encoder: (B, 32, 7) -> Embeddings (concat) -> Projection -> 1-layer Transformer -> (B, 32, 128)
- Context Self-Attention: 1-layer Transformer on concatenated cards+actions (B, 41, 128)
- Cross-Attention Fusion: 2 layers of learnable queries (6 tokens) attending to context
- (Optional) Opponent History Encoder: (B, 8) -> (B, 32) opponent tendencies
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
# Configuration (matching model.py)
# =============================================================================

@dataclass(frozen=True)
class TossHoldemTransformerConfig:
    """Configuration matching the JAX model."""
    num_actions: int = 13
    num_betting_actions: int = 10
    num_discard_actions: int = 3
    card_seq_len: int = 9
    action_seq_len: int = 32
    num_card_types: int = 6  # 0=hero_private, 1=flop, 2=hero_discard, 3=villain_discard, 4=turn, 5=river
    card_input_dim: int = 2
    action_input_dim: int = 7

    emb_dim: int = 128
    num_heads: int = 2
    ffn_dim: int = 512
    num_transformer_layers: int = 1  # ActionEncoder uses 1 layer
    num_cross_attention_layers: int = 2
    num_query_tokens: int = 6  # 6 query tokens for cross-attention
    processing_dim: int = 256

    # Card encoder embedding dims (must sum to emb_dim)
    rank_emb_dim: int = 48
    suit_emb_dim: int = 32
    # position_emb_dim = emb_dim - rank_emb_dim - suit_emb_dim = 48

    # Action encoder embedding dims
    action_component_emb_dim: int = 24  # 24 * 4 + 32 = 128

    # Opponent history
    opp_history_dim: int = 4  # [fold_rate, aggression, wtsd, hands_played]
    opp_history_emb_dim: int = 32

    # Opponent hand tier prediction (auxiliary task)
    num_hand_tiers: int = 9  # 0=high card through 8=straight flush


# =============================================================================
# Transformer Components
# =============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Faster than LayerNorm as it skips mean centering.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (x * self.scale).to(dtype)


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
        self.norm1 = RMSNorm(emb_dim)
        self.attn = MultiHeadAttention(emb_dim, num_heads)
        self.norm2 = RMSNorm(emb_dim)
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
    """Encodes card tensor using embeddings and self-attention.

    Input: (B, 9, 2) where each entry is [card_index, type_id].

    Position semantics:
        0-2: Hero's private cards
        3-4: Flop (community)
        5: Hero's discard
        6: Villain's discard
        7: Turn
        8: River

    Output: (B, 9, emb_dim)

    Uses self-attention only (no FFN) to learn card relationships (flush/straight patterns).
    FFN removed since it would wastefully process all 9 positions even when few cards are valid.
    """

    def __init__(self, cfg: TossHoldemTransformerConfig):
        super().__init__()
        self.cfg = cfg

        # Embedding dims sum to emb_dim (no projection needed)
        position_emb_dim = cfg.emb_dim - cfg.rank_emb_dim - cfg.suit_emb_dim

        # Embeddings for rank, suit, position/type (concatenated)
        self.rank_emb = nn.Embedding(14, cfg.rank_emb_dim)  # 13 ranks + invalid
        self.suit_emb = nn.Embedding(5, cfg.suit_emb_dim)   # 4 suits + invalid
        self.type_emb = nn.Embedding(cfg.num_card_types + 1, position_emb_dim)  # types + invalid

        # Self-attention only (no FFN) for card relationships
        self.norm = RMSNorm(cfg.emb_dim)
        self.attn = MultiHeadAttention(cfg.emb_dim, cfg.num_heads)

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

        # Concatenate (embeddings sum to emb_dim)
        concat = torch.cat([rank_e, suit_e, type_e], dim=-1)  # (B, 9, emb_dim)

        # Self-attention only (no FFN) for card relationships
        normed = self.norm(concat)
        attn_out = self.attn(normed, normed, normed, valid_mask)
        card_repr = concat + attn_out  # (B, 9, emb_dim)

        return card_repr, valid_mask


class ActionEncoder(nn.Module):
    """Encodes action history tensor using embeddings and transformer."""

    def __init__(self, cfg: TossHoldemTransformerConfig):
        super().__init__()
        self.cfg = cfg

        # Categorical embeddings
        self.actor_emb = nn.Embedding(2, cfg.action_component_emb_dim)
        self.event_emb = nn.Embedding(13, cfg.action_component_emb_dim)
        self.street_emb = nn.Embedding(5, cfg.action_component_emb_dim)
        self.time_emb = nn.Embedding(cfg.action_seq_len, cfg.action_component_emb_dim)

        # Continuous features projection
        self.continuous_proj = nn.Linear(3, 32)

        # Project to emb_dim
        concat_dim = 4 * cfg.action_component_emb_dim + 32
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
        time_id = action_tensor[..., 3].long()
        to_call = action_tensor[..., 4:5]
        pot_base = action_tensor[..., 5:6]
        eff_stack = action_tensor[..., 6:7]

        # Clamp indices to valid ranges
        actor_id = actor_id.clamp(0, 1)
        event_type = event_type.clamp(0, 12)
        street_id = street_id.clamp(0, 4)
        time_mod = time_id % self.cfg.action_seq_len

        # Categorical embeddings
        actor_e = self.actor_emb(actor_id)
        event_e = self.event_emb(event_type)
        street_e = self.street_emb(street_id)
        time_e = self.time_emb(time_mod)

        # Continuous features (log-transformed)
        eps = 1e-6
        pot_base_safe = torch.clamp(pot_base, min=eps)
        cont1 = torch.log1p(to_call / pot_base_safe)
        cont2 = torch.log1p(pot_base_safe)
        cont3 = torch.log1p(eff_stack / pot_base_safe)
        continuous = torch.cat([cont1, cont2, cont3], dim=-1)
        continuous_e = self.continuous_proj(continuous)

        # Concatenate all embeddings
        concat = torch.cat([actor_e, event_e, street_e, time_e, continuous_e], dim=-1)

        # Project to emb_dim
        combined = self.projection(concat)

        # Transformer
        encoded = self.transformer(combined, action_mask)

        return encoded


# =============================================================================
# Opponent History Encoder
# =============================================================================

class OpponentHistoryEncoder(nn.Module):
    """Encodes opponent history statistics into an embedding.

    Input: (B, OPP_HISTORY_DIM=4) where each entry contains:
        [0]: fold_rate - how often they give up (tightness)
        [1]: aggression - raises / (raises + calls) (passive vs aggressive)
        [2]: wtsd - went to showdown rate (station vs folder)
        [3]: hands_played - log1p of hands (sample size / confidence)

    Output: (B, opp_history_emb_dim)
    """

    def __init__(self, cfg: TossHoldemTransformerConfig):
        super().__init__()
        self.opp_history_fc1 = nn.Linear(cfg.opp_history_dim, 64)
        self.opp_history_fc2 = nn.Linear(64, cfg.opp_history_emb_dim)

    def forward(self, opp_history: torch.Tensor) -> torch.Tensor:
        """
        Args:
            opp_history: (B, OPP_HISTORY_DIM) float32 opponent statistics

        Returns:
            opp_history_emb: (B, opp_history_emb_dim) opponent embedding
        """
        x = F.relu(self.opp_history_fc1(opp_history))
        x = self.opp_history_fc2(x)
        return x


# =============================================================================
# Main Model
# =============================================================================

class TossHoldemTransformerTorch(nn.Module):
    """PyTorch implementation of the transformer-based Toss-or-Hold'em model.

    Architecture:
        1. Card Encoder: (B, 9, 2) -> (B, 9, 128) with embeddings (concatenated)
        2. Action Encoder: (B, 32, 7) -> (B, 32, 128) with embeddings (concatenated)
        3. (Optional) Opponent History Encoder: (B, 8) -> (B, 32) opponent tendencies
        4. Normalize and concatenate: (B, 41, 128)
        5. Context Self-Attention: Cards+actions attend to each other
        6. Multiple Cross-Attention Fusion layers with learnable queries: (B, 6, 128)
        7. Processing MLP: Flatten -> Dense -> ReLU -> Dense -> ReLU -> (B, 256)
           (with opponent history embedding concatenated if provided)
        8. Separate Output Heads:
           - Betting Policy: (B, 10)
           - Discard Policy: (B, 3)
           - Value: (B,)
    """

    def __init__(self, cfg: TossHoldemTransformerConfig):
        super().__init__()
        self.cfg = cfg

        # Encoders
        self.card_encoder = CardEncoder(cfg)
        self.action_encoder = ActionEncoder(cfg)

        # Normalization layers before concatenation (matching JAX model)
        self.card_norm = RMSNorm(cfg.emb_dim)
        self.action_norm = RMSNorm(cfg.emb_dim)

        # Context self-attention (cards + actions attend to each other before cross-attention)
        self.context_self_attn = TransformerBlock(
            emb_dim=cfg.emb_dim,
            num_heads=cfg.num_heads,
            ffn_dim=cfg.ffn_dim,
        )

        # Learnable query tokens
        self.query_tokens = nn.Parameter(torch.randn(cfg.num_query_tokens, cfg.emb_dim) * 0.02)

        # Cross-attention layers
        self.cross_attention_layers = nn.ModuleList([
            CrossAttentionBlock(cfg.emb_dim, cfg.num_heads, cfg.ffn_dim)
            for _ in range(cfg.num_cross_attention_layers)
        ])
        self.cross_attn_final_norm = RMSNorm(cfg.emb_dim)

        # Opponent history encoder (optional)
        self.opp_history_encoder = OpponentHistoryEncoder(cfg)

        # Processing MLP
        # Note: flat_dim depends on whether opp_history is provided
        flat_dim = cfg.num_query_tokens * cfg.emb_dim
        self.processing_fc1 = nn.Linear(flat_dim, 512)
        self.processing_fc1_with_opp = nn.Linear(flat_dim + cfg.opp_history_emb_dim, 512)
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
        opp_history: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            card_tensor: (B, 9, 2) int32 - card tensor
            action_tensor: (B, 32, 7) float32 - action history
            action_mask: (B, 32) boolean - action history mask
            opp_history: (B, OPP_HISTORY_DIM=4) float32 - opponent statistics (optional)
                [fold_rate, aggression, wtsd, hands_played]

        Returns:
            policy_logits: (B, 13)
            value: (B,)
        """
        batch_size = card_tensor.shape[0]

        # =================================================================
        # 1. Encode cards and actions
        # =================================================================
        card_repr, card_valid_mask = self.card_encoder(card_tensor)  # (B, 9, 128), (B, 9)
        action_repr = self.action_encoder(action_tensor, action_mask)  # (B, 32, 128)

        # =================================================================
        # 2. Normalize and concatenate to form context
        # =================================================================
        card_repr = self.card_norm(card_repr)
        action_repr = self.action_norm(action_repr)

        context = torch.cat([card_repr, action_repr], dim=1)  # (B, 41, 128)
        context_mask = torch.cat([card_valid_mask, action_mask], dim=1)  # (B, 41)

        # =================================================================
        # 3. Context self-attention (cards + actions attend to each other)
        # =================================================================
        context = self.context_self_attn(context, context_mask)  # (B, 41, 128)

        # =================================================================
        # 4. Cross-attention with learnable queries
        # =================================================================
        fused = self.query_tokens.unsqueeze(0).expand(batch_size, -1, -1)  # (B, 6, 128)

        for cross_attn in self.cross_attention_layers:
            fused = cross_attn(fused, context, context_mask)

        fused = self.cross_attn_final_norm(fused)

        # =================================================================
        # 5. Process opponent history (if provided)
        # =================================================================
        opp_history_emb = None
        if opp_history is not None:
            opp_history_emb = self.opp_history_encoder(opp_history)  # (B, 32)

        # =================================================================
        # 6. Processing MLP (with optional opponent history)
        # =================================================================
        flat = fused.reshape(batch_size, -1)  # (B, 768) = 6 * 128

        if opp_history_emb is not None:
            flat = torch.cat([flat, opp_history_emb], dim=-1)  # (B, 768 + 32)
            processed = F.relu(self.processing_fc1_with_opp(flat))  # (B, 512)
        else:
            processed = F.relu(self.processing_fc1(flat))  # (B, 512)

        processed = F.relu(self.processing_fc2(processed))  # (B, 256)

        # =================================================================
        # 7. Output heads
        # =================================================================
        betting_h = F.relu(self.betting_hidden(processed))
        betting_logits = self.betting_out(betting_h)  # (B, 10)

        discard_h = F.relu(self.discard_hidden(processed))
        discard_logits = self.discard_out(discard_h)  # (B, 3)

        policy_logits = torch.cat([betting_logits, discard_logits], dim=-1)  # (B, 13)

        value_h = F.relu(self.value_hidden(processed))
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