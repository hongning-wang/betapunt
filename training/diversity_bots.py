"""Diversity bots for training: pre-trained models with distinct strategies.

This module provides infrastructure for loading and using pre-trained models
as fixed opponents during training. Two diversity bots are supported:

1. step_40000 (python_bot): Larger model (emb_dim=256), plays tight/precise strategy
   Uses OldTossHoldemModel with LayerNorm.

2. step_34000 (python_bot_new): Smaller model (emb_dim=128), plays more aggressive
   Uses NewV1TossHoldemModel (frozen architecture with CardEncoder FFN).

These bots use different architectures, so we need separate model definitions.
The NewV1 architecture is frozen to match the checkpoint step_34000 was trained with.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple, Optional

import jax
import jax.numpy as jnp
import flax.linen as nn
import chex
import orbax.checkpoint


# =============================================================================
# Old Architecture Model (step_40000 from python_bot)
# =============================================================================

# Configuration for old model
OLD_EMB_DIM = 256
OLD_NUM_HEADS = 2
OLD_FFN_DIM = 512
OLD_NUM_TRANSFORMER_LAYERS = 2
OLD_NUM_CROSS_ATTENTION_LAYERS = 2
OLD_NUM_QUERY_TOKENS = 6
OLD_PROCESSING_DIM = 256
OLD_NUM_CARD_TYPES = 6  # hero0, hero1, hero2, flop0, flop1, etc.
OLD_CARD_RANK_EMB_DIM = 64
OLD_CARD_SUIT_EMB_DIM = 16
OLD_CARD_TYPE_EMB_DIM = 48  # 64+16+48=128 -> project to 256

NUM_ACTIONS = 13
NUM_BETTING_ACTIONS = 10
NUM_DISCARD_ACTIONS = 3
CARD_SEQ_LEN = 9
ACTION_SEQ_LEN = 32


class OldMultiHeadAttention(nn.Module):
    """Multi-head attention for old architecture (uses LayerNorm)."""
    num_heads: int
    head_dim: int

    @nn.compact
    def __call__(
        self,
        query: chex.Array,
        key: chex.Array,
        value: chex.Array,
        mask: Optional[chex.Array] = None,
    ) -> chex.Array:
        batch_size, q_len, input_dim = query.shape
        kv_len = key.shape[1]
        projection_dim = self.num_heads * self.head_dim

        query = query.astype(jnp.bfloat16)
        key = key.astype(jnp.bfloat16)
        value = value.astype(jnp.bfloat16)

        q = nn.Dense(projection_dim, use_bias=False, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16)(query)
        k = nn.Dense(projection_dim, use_bias=False, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16)(key)
        v = nn.Dense(projection_dim, use_bias=False, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16)(value)

        q = q.reshape(batch_size, q_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, kv_len, self.num_heads, self.head_dim)
        v = v.reshape(batch_size, kv_len, self.num_heads, self.head_dim)

        bias = jnp.where(mask[:, None, None, :], 0.0, -1e9)
        attn_output = jax.nn.dot_product_attention(q, k, v, bias=bias)

        attn_output = attn_output.reshape(batch_size, q_len, projection_dim)
        output = nn.Dense(input_dim, use_bias=False, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16)(attn_output)

        return output


class OldTransformerBlock(nn.Module):
    """Transformer block for old architecture (uses LayerNorm)."""
    emb_dim: int
    num_heads: int
    ffn_dim: int

    @nn.compact
    def __call__(self, x: chex.Array, mask: Optional[chex.Array] = None) -> chex.Array:
        x = x.astype(jnp.bfloat16)
        head_dim = self.emb_dim // self.num_heads

        # Pre-norm self-attention
        residual = x
        x = nn.LayerNorm(dtype=jnp.bfloat16, param_dtype=jnp.bfloat16)(x)
        x = OldMultiHeadAttention(num_heads=self.num_heads, head_dim=head_dim, name='MultiHeadAttention_0')(x, x, x, mask)
        x = residual + x

        # Pre-norm FFN
        residual = x
        x = nn.LayerNorm(dtype=jnp.bfloat16, param_dtype=jnp.bfloat16)(x)
        x = nn.Dense(self.ffn_dim, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.emb_dim, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16)(x)
        x = residual + x

        return x


class OldTransformerEncoder(nn.Module):
    """Transformer encoder for old architecture."""
    emb_dim: int
    num_heads: int
    ffn_dim: int
    num_layers: int

    @nn.compact
    def __call__(self, x: chex.Array, mask: Optional[chex.Array] = None) -> chex.Array:
        x = x.astype(jnp.bfloat16)
        for layer_idx in range(self.num_layers):
            x = OldTransformerBlock(
                emb_dim=self.emb_dim,
                num_heads=self.num_heads,
                ffn_dim=self.ffn_dim,
                name=f'layer_{layer_idx}',
            )(x, mask)
        x = nn.LayerNorm(dtype=jnp.bfloat16, param_dtype=jnp.bfloat16)(x)
        return x


class OldCrossAttentionBlock(nn.Module):
    """Cross-attention block for old architecture."""
    emb_dim: int
    num_heads: int
    ffn_dim: int

    @nn.compact
    def __call__(
        self,
        query_tokens: chex.Array,
        context: chex.Array,
        context_mask: Optional[chex.Array] = None,
    ) -> chex.Array:
        query_tokens = query_tokens.astype(jnp.bfloat16)
        context = context.astype(jnp.bfloat16)
        head_dim = self.emb_dim // self.num_heads

        attn_input = nn.LayerNorm(dtype=jnp.bfloat16, param_dtype=jnp.bfloat16)(query_tokens)
        context_normed = nn.LayerNorm(dtype=jnp.bfloat16, param_dtype=jnp.bfloat16)(context)

        attn_output = OldMultiHeadAttention(
            num_heads=self.num_heads,
            head_dim=head_dim,
            name='MultiHeadAttention_0',
        )(attn_input, context_normed, context_normed, context_mask)

        post_attn = query_tokens + attn_output

        ffn_input = nn.LayerNorm(dtype=jnp.bfloat16, param_dtype=jnp.bfloat16)(post_attn)
        ffn_hidden = nn.Dense(self.ffn_dim, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16)(ffn_input)
        ffn_hidden = nn.gelu(ffn_hidden)
        ffn_out = nn.Dense(self.emb_dim, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16)(ffn_hidden)

        return post_attn + ffn_out


class OldCardEncoder(nn.Module):
    """Card encoder for old architecture (2-layer MLP projection)."""
    emb_dim: int = OLD_EMB_DIM
    rank_emb_dim: int = OLD_CARD_RANK_EMB_DIM
    suit_emb_dim: int = OLD_CARD_SUIT_EMB_DIM
    type_emb_dim: int = OLD_CARD_TYPE_EMB_DIM
    hidden_dim: int = 256  # intermediate hidden dimension

    @nn.compact
    def __call__(self, card_tensor: chex.Array) -> Tuple[chex.Array, chex.Array]:
        card_index = card_tensor[..., 0].astype(jnp.int32)
        type_id = card_tensor[..., 1].astype(jnp.int32)

        valid_mask = card_index >= 0
        card_index_safe = jnp.where(card_index >= 0, card_index, 52).astype(jnp.int32)

        rank = (card_index_safe % 13).astype(jnp.int32)
        suit = (card_index_safe // 13).astype(jnp.int32)

        is_invalid = card_index < 0
        rank_safe = jnp.where(is_invalid, 13, rank).astype(jnp.int32)
        suit_safe = jnp.where(is_invalid, 4, suit).astype(jnp.int32)
        type_safe = jnp.where(type_id >= 0, type_id, OLD_NUM_CARD_TYPES).astype(jnp.int32)

        rank_emb = nn.Embed(14, self.rank_emb_dim, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, name='Embed_0')(rank_safe)
        suit_emb = nn.Embed(5, self.suit_emb_dim, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, name='Embed_1')(suit_safe)
        type_emb = nn.Embed(OLD_NUM_CARD_TYPES + 1, self.type_emb_dim, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, name='Embed_2')(type_safe)

        concat = jnp.concatenate([rank_emb, suit_emb, type_emb], axis=-1)
        # 2-layer MLP: Dense_0 = hidden, Dense_1 = projection
        hidden = nn.Dense(self.hidden_dim, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, name='Dense_0')(concat)
        hidden = nn.gelu(hidden)
        card_repr = nn.Dense(self.emb_dim, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, name='Dense_1')(hidden)

        return card_repr, valid_mask


class OldActionEncoder(nn.Module):
    """Action encoder for old architecture."""
    emb_dim: int = OLD_EMB_DIM
    num_heads: int = OLD_NUM_HEADS
    ffn_dim: int = OLD_FFN_DIM
    num_layers: int = OLD_NUM_TRANSFORMER_LAYERS
    # Separate embedding dims for each component
    actor_emb_dim: int = 8
    event_emb_dim: int = 48
    street_time_emb_dim: int = 48  # combined street+time: 5*32=160 entries
    continuous_dim: int = 24

    @nn.compact
    def __call__(self, action_tensor: chex.Array, action_mask: Optional[chex.Array] = None) -> chex.Array:
        action_tensor = action_tensor.astype(jnp.bfloat16)

        actor_id = action_tensor[..., 0].astype(jnp.int32)
        event_type = action_tensor[..., 1].astype(jnp.int32)
        street_id = action_tensor[..., 2].astype(jnp.int32)
        time_id = action_tensor[..., 3].astype(jnp.int32)
        to_call = action_tensor[..., 4:5]
        pot_base = action_tensor[..., 5:6]
        eff_stack = action_tensor[..., 6:7]

        actor_emb = nn.Embed(2, self.actor_emb_dim, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, name='Embed_0')(actor_id)
        event_emb = nn.Embed(13, self.event_emb_dim, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, name='Embed_1')(event_type)
        # Combined street+time embedding: street_id * 32 + time_id
        time_mod = jnp.mod(time_id, jnp.int32(ACTION_SEQ_LEN))
        street_time_idx = street_id * ACTION_SEQ_LEN + time_mod
        street_time_emb = nn.Embed(5 * ACTION_SEQ_LEN, self.street_time_emb_dim, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, name='Embed_2')(street_time_idx)

        eps = jnp.bfloat16(1e-6)
        pot_base_safe = jnp.maximum(pot_base, eps)
        cont1 = jnp.log1p(to_call / pot_base_safe)
        cont2 = jnp.log1p(pot_base_safe)
        cont3 = jnp.log1p(eff_stack / pot_base_safe)
        continuous = jnp.concatenate([cont1, cont2, cont3], axis=-1)
        continuous_emb = nn.Dense(self.continuous_dim, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, name='Dense_0')(continuous)

        concat = jnp.concatenate([actor_emb, event_emb, street_time_emb, continuous_emb], axis=-1)
        combined = nn.Dense(self.emb_dim, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, name='Dense_1')(concat)

        encoded = OldTransformerEncoder(
            emb_dim=self.emb_dim,
            num_heads=self.num_heads,
            ffn_dim=self.ffn_dim,
            num_layers=self.num_layers,
            name='TransformerEncoder_0',
        )(combined, mask=action_mask)

        return encoded


class OldTossHoldemModel(nn.Module):
    """Old architecture model (step_40000 from python_bot).

    Key differences from new model:
    - emb_dim: 256 (vs 128)
    - num_query_tokens: 6 (vs 4)
    - num_card_types: 9 (vs 6)
    - Uses LayerNorm (vs RMSNorm)
    - Card encoder: simple projection (vs transformer block)
    """

    emb_dim: int = OLD_EMB_DIM
    num_heads: int = OLD_NUM_HEADS
    ffn_dim: int = OLD_FFN_DIM
    num_layers: int = OLD_NUM_TRANSFORMER_LAYERS
    num_cross_attention_layers: int = OLD_NUM_CROSS_ATTENTION_LAYERS
    num_query_tokens: int = OLD_NUM_QUERY_TOKENS
    processing_dim: int = OLD_PROCESSING_DIM

    @nn.compact
    def __call__(
        self,
        card_tensor: chex.Array,
        action_tensor: chex.Array,
        action_mask: chex.Array,
    ) -> Tuple[chex.Array, chex.Array]:
        batch_size = card_tensor.shape[0]
        action_tensor = action_tensor.astype(jnp.bfloat16)

        card_repr, card_valid_mask = OldCardEncoder(emb_dim=self.emb_dim, name='CardEncoder_0')(card_tensor)
        action_repr = OldActionEncoder(
            emb_dim=self.emb_dim,
            num_heads=self.num_heads,
            ffn_dim=self.ffn_dim,
            num_layers=self.num_layers,
            name='ActionEncoder_0',
        )(action_tensor, action_mask)

        context = jnp.concatenate([card_repr, action_repr], axis=1)
        context_mask = jnp.concatenate([card_valid_mask, action_mask], axis=1)

        query_tokens = self.param(
            'query_tokens',
            nn.initializers.lecun_normal(),
            (self.num_query_tokens, self.emb_dim),
            jnp.bfloat16,
        )

        fused = jnp.broadcast_to(
            query_tokens[None, :, :],
            (batch_size, self.num_query_tokens, self.emb_dim),
        ).astype(jnp.bfloat16)

        for layer_idx in range(self.num_cross_attention_layers):
            fused = OldCrossAttentionBlock(
                emb_dim=self.emb_dim,
                num_heads=self.num_heads,
                ffn_dim=self.ffn_dim,
                name=f'cross_attn_{layer_idx}',
            )(fused, context, context_mask)

        fused = nn.LayerNorm(dtype=jnp.bfloat16, param_dtype=jnp.bfloat16)(fused)

        flat = fused.reshape(batch_size, -1)
        processed = nn.Dense(512, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16)(flat)
        processed = nn.relu(processed)
        processed = nn.Dense(self.processing_dim, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16)(processed)
        processed = nn.relu(processed)

        # Betting policy head
        betting_hidden = nn.Dense(128, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, name='betting_hidden')(processed)
        betting_hidden = nn.relu(betting_hidden)
        betting_logits = nn.Dense(NUM_BETTING_ACTIONS, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, name='betting_out')(betting_hidden)

        # Discard policy head
        discard_hidden = nn.Dense(128, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, name='discard_hidden')(processed)
        discard_hidden = nn.relu(discard_hidden)
        discard_logits = nn.Dense(NUM_DISCARD_ACTIONS, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, name='discard_out')(discard_hidden)

        policy_logits = jnp.concatenate([betting_logits, discard_logits], axis=-1)

        # Value head
        value_hidden = nn.Dense(128, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, name='value_hidden')(processed)
        value_hidden = nn.relu(value_hidden)
        value = nn.Dense(1, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, name='value_out')(value_hidden)
        value = value.squeeze(-1)

        return policy_logits.astype(jnp.float32), value.astype(jnp.float32)


# =============================================================================
# Diversity Bot Container
# =============================================================================

@jax.tree_util.register_pytree_node_class
@dataclass
class DiversityBots:
    """Container for diversity bot parameters and models."""
    # Old architecture (step_40000, tight/precise)
    old_params: Any  # Parameters for OldTossHoldemModel
    # New V1 architecture (step_34000, more aggressive)
    new_params: Any  # Parameters for NewV1TossHoldemModel

    def tree_flatten(self):
        """Flatten for JAX pytree."""
        return (self.old_params, self.new_params), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Unflatten from JAX pytree."""
        old_params, new_params = children
        return cls(old_params=old_params, new_params=new_params)


def load_diversity_bots(
    old_checkpoint_path: str = "./python_bot/step_40000",
    new_checkpoint_path: str = "./python_bot_new/step_34000",
) -> DiversityBots:
    """Load pre-trained diversity bots from checkpoints.

    Args:
        old_checkpoint_path: Path to step_40000 checkpoint (old architecture)
        new_checkpoint_path: Path to step_34000 checkpoint (new architecture)

    Returns:
        DiversityBots container with loaded parameters
    """
    ckpter = orbax.checkpoint.PyTreeCheckpointer()

    # Load old architecture checkpoint
    print(f"Loading old architecture from {old_checkpoint_path}")
    old_restored = ckpter.restore(old_checkpoint_path)
    old_params = old_restored.get("params", old_restored)

    # Load new architecture checkpoint
    print(f"Loading new architecture from {new_checkpoint_path}")
    new_restored = ckpter.restore(new_checkpoint_path)
    new_params = new_restored.get("params", new_restored)

    return DiversityBots(old_params=old_params, new_params=new_params)


def init_diversity_bot_models(rng: chex.PRNGKey) -> Tuple[OldTossHoldemModel, NewV1TossHoldemModel]:
    """Initialize diversity bot models for parameter structure.

    This is needed to create the correct parameter structure before loading.
    """
    old_model = OldTossHoldemModel()
    new_model = NewV1TossHoldemModel()

    # Dummy inputs
    dummy_cards = jnp.zeros((1, CARD_SEQ_LEN, 2), dtype=jnp.int32)
    dummy_actions = jnp.zeros((1, ACTION_SEQ_LEN, 7), dtype=jnp.float32)
    dummy_mask = jnp.ones((1, ACTION_SEQ_LEN), dtype=jnp.bool_)

    rng1, rng2 = jax.random.split(rng)
    old_vars = old_model.init(rng1, dummy_cards, dummy_actions, dummy_mask)
    new_vars = new_model.init(rng2, dummy_cards, dummy_actions, dummy_mask)

    return old_model, new_model


# Module-level model instances for JIT compatibility
_OLD_DIVERSITY_MODEL = OldTossHoldemModel()


def _convert_card_encoding_new_to_old(card_tensor: chex.Array) -> chex.Array:
    """Convert new-style card encoding (semantic type_ids) to old-style (position-based).

    New encoding type_ids: 0=hero_private, 1=flop, 2=hero_discard, 3=villain_discard, 4=turn, 5=river
    Old encoding type_ids: position index 0-8

    The mapping is:
      positions 0-2 (hero private): new type_id=0 -> old type_id=0,1,2
      positions 3-4 (flop): new type_id=1 -> old type_id=3,4
      position 5 (hero discard): new type_id=2 -> old type_id=5
      position 6 (villain discard): new type_id=3 -> old type_id=6
      position 7 (turn): new type_id=4 -> old type_id=7
      position 8 (river): new type_id=5 -> old type_id=8

    Args:
        card_tensor: (B, 9, 2) with [card_index, new_type_id]

    Returns:
        (B, 9, 2) with [card_index, old_type_id]
    """
    card_indices = card_tensor[..., 0]  # (B, 9)
    new_type_ids = card_tensor[..., 1]  # (B, 9)

    # Old type_ids are just position indices
    position_ids = jnp.broadcast_to(
        jnp.arange(CARD_SEQ_LEN, dtype=jnp.int32)[None, :],
        card_indices.shape
    )

    # Keep -1 for invalid entries
    old_type_ids = jnp.where(new_type_ids >= 0, position_ids, jnp.int32(-1))

    return jnp.stack([card_indices, old_type_ids], axis=-1)


# =============================================================================
# New V1 Architecture (step_34000 from python_bot_new) - FROZEN
# This is the architecture step_34000 was trained with (CardEncoder has FFN)
# =============================================================================

NEW_V1_EMB_DIM = 128
NEW_V1_NUM_HEADS = 2
NEW_V1_FFN_DIM = 512
NEW_V1_NUM_TRANSFORMER_LAYERS = 2
NEW_V1_NUM_CROSS_ATTENTION_LAYERS = 2
NEW_V1_NUM_QUERY_TOKENS = 4
NEW_V1_PROCESSING_DIM = 256
NEW_V1_NUM_CARD_TYPES = 6


class NewV1RMSNorm(nn.Module):
    """RMSNorm for new v1 architecture."""
    epsilon: float = 1e-6

    @nn.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        dtype = x.dtype
        x = x.astype(jnp.float32)
        variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        x = x * jax.lax.rsqrt(variance + self.epsilon)
        scale = self.param('scale', nn.initializers.ones, (x.shape[-1],), jnp.float32)
        return (x * scale).astype(dtype)


class NewV1MultiHeadAttention(nn.Module):
    """MHA for new v1 architecture."""
    num_heads: int
    head_dim: int

    @nn.compact
    def __call__(
        self,
        query: chex.Array,
        key: chex.Array,
        value: chex.Array,
        mask: Optional[chex.Array] = None,
    ) -> chex.Array:
        batch_size, q_len, input_dim = query.shape
        kv_len = key.shape[1]
        projection_dim = self.num_heads * self.head_dim

        query = query.astype(jnp.bfloat16)
        key = key.astype(jnp.bfloat16)
        value = value.astype(jnp.bfloat16)

        q = nn.Dense(projection_dim, use_bias=False, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, name='Dense_0')(query)
        k = nn.Dense(projection_dim, use_bias=False, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, name='Dense_1')(key)
        v = nn.Dense(projection_dim, use_bias=False, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, name='Dense_2')(value)

        q = q.reshape(batch_size, q_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, kv_len, self.num_heads, self.head_dim)
        v = v.reshape(batch_size, kv_len, self.num_heads, self.head_dim)

        bias = jnp.where(mask[:, None, None, :], 0.0, -1e9)
        attn_output = jax.nn.dot_product_attention(q, k, v, bias=bias)

        attn_output = attn_output.reshape(batch_size, q_len, projection_dim)
        output = nn.Dense(input_dim, use_bias=False, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, name='Dense_3')(attn_output)
        return output


class NewV1TransformerBlock(nn.Module):
    """Transformer block for new v1 architecture (has FFN)."""
    emb_dim: int
    num_heads: int
    ffn_dim: int

    @nn.compact
    def __call__(self, x: chex.Array, mask: Optional[chex.Array] = None) -> chex.Array:
        x = x.astype(jnp.bfloat16)
        head_dim = self.emb_dim // self.num_heads

        # Pre-norm self-attention
        residual = x
        x = NewV1RMSNorm(name='RMSNorm_0')(x)
        x = NewV1MultiHeadAttention(num_heads=self.num_heads, head_dim=head_dim, name='MultiHeadAttention_0')(x, x, x, mask)
        x = residual + x

        # Pre-norm FFN
        residual = x
        x = NewV1RMSNorm(name='RMSNorm_1')(x)
        x = nn.Dense(self.ffn_dim, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, name='Dense_0')(x)
        x = nn.gelu(x)
        x = nn.Dense(self.emb_dim, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, name='Dense_1')(x)
        x = residual + x

        return x


class NewV1TransformerEncoder(nn.Module):
    """Transformer encoder for new v1 architecture."""
    emb_dim: int
    num_heads: int
    ffn_dim: int
    num_layers: int

    @nn.compact
    def __call__(self, x: chex.Array, mask: Optional[chex.Array] = None) -> chex.Array:
        x = x.astype(jnp.bfloat16)
        for layer_idx in range(self.num_layers):
            x = NewV1TransformerBlock(
                emb_dim=self.emb_dim,
                num_heads=self.num_heads,
                ffn_dim=self.ffn_dim,
                name=f'layer_{layer_idx}',
            )(x, mask)
        x = NewV1RMSNorm(name='RMSNorm_0')(x)
        return x


class NewV1CrossAttentionBlock(nn.Module):
    """Cross-attention for new v1 architecture."""
    emb_dim: int
    num_heads: int
    ffn_dim: int

    @nn.compact
    def __call__(
        self,
        query_tokens: chex.Array,
        context: chex.Array,
        context_mask: Optional[chex.Array] = None,
    ) -> chex.Array:
        query_tokens = query_tokens.astype(jnp.bfloat16)
        context = context.astype(jnp.bfloat16)
        head_dim = self.emb_dim // self.num_heads

        attn_input = NewV1RMSNorm(name='RMSNorm_0')(query_tokens)
        context_normed = NewV1RMSNorm(name='RMSNorm_1')(context)

        attn_output = NewV1MultiHeadAttention(
            num_heads=self.num_heads,
            head_dim=head_dim,
            name='MultiHeadAttention_0',
        )(attn_input, context_normed, context_normed, context_mask)

        post_attn = query_tokens + attn_output

        ffn_input = NewV1RMSNorm(name='RMSNorm_2')(post_attn)
        ffn_hidden = nn.Dense(self.ffn_dim, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, name='Dense_0')(ffn_input)
        ffn_hidden = nn.gelu(ffn_hidden)
        ffn_out = nn.Dense(self.emb_dim, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, name='Dense_1')(ffn_hidden)

        return post_attn + ffn_out


class NewV1CardEncoder(nn.Module):
    """Card encoder for new v1 architecture (has TransformerBlock with FFN)."""
    emb_dim: int = NEW_V1_EMB_DIM
    rank_emb_dim: int = 48
    suit_emb_dim: int = 32
    num_heads: int = 2
    ffn_dim: int = 256

    @nn.compact
    def __call__(self, card_tensor: chex.Array) -> Tuple[chex.Array, chex.Array]:
        card_index = card_tensor[..., 0].astype(jnp.int32)
        type_id = card_tensor[..., 1].astype(jnp.int32)

        valid_mask = card_index >= 0
        card_index_safe = jnp.where(card_index >= 0, card_index, 52).astype(jnp.int32)

        rank = (card_index_safe % 13).astype(jnp.int32)
        suit = (card_index_safe // 13).astype(jnp.int32)

        is_invalid = card_index < 0
        rank_safe = jnp.where(is_invalid, 13, rank).astype(jnp.int32)
        suit_safe = jnp.where(is_invalid, 4, suit).astype(jnp.int32)
        type_safe = jnp.where(type_id >= 0, type_id, NEW_V1_NUM_CARD_TYPES).astype(jnp.int32)

        position_emb_dim = self.emb_dim - self.rank_emb_dim - self.suit_emb_dim

        rank_emb = nn.Embed(14, self.rank_emb_dim, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, name='Embed_0')(rank_safe)
        suit_emb = nn.Embed(5, self.suit_emb_dim, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, name='Embed_1')(suit_safe)
        position_emb = nn.Embed(NEW_V1_NUM_CARD_TYPES + 1, position_emb_dim, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, name='Embed_2')(type_safe)

        concat = jnp.concatenate([rank_emb, suit_emb, position_emb], axis=-1)

        # TransformerBlock WITH FFN (this is what step_34000 was trained with)
        card_repr = NewV1TransformerBlock(
            emb_dim=self.emb_dim,
            num_heads=self.num_heads,
            ffn_dim=self.ffn_dim,
            name='TransformerBlock_0',
        )(concat, valid_mask)

        return card_repr, valid_mask


class NewV1ActionEncoder(nn.Module):
    """Action encoder for new v1 architecture."""
    emb_dim: int = NEW_V1_EMB_DIM
    num_heads: int = NEW_V1_NUM_HEADS
    ffn_dim: int = NEW_V1_FFN_DIM
    num_layers: int = NEW_V1_NUM_TRANSFORMER_LAYERS
    component_emb_dim: int = 24

    @nn.compact
    def __call__(self, action_tensor: chex.Array, action_mask: Optional[chex.Array] = None) -> chex.Array:
        action_tensor = action_tensor.astype(jnp.bfloat16)

        actor_id = action_tensor[..., 0].astype(jnp.int32)
        event_type = action_tensor[..., 1].astype(jnp.int32)
        street_id = action_tensor[..., 2].astype(jnp.int32)
        time_id = action_tensor[..., 3].astype(jnp.int32)
        to_call = action_tensor[..., 4:5]
        pot_base = action_tensor[..., 5:6]
        eff_stack = action_tensor[..., 6:7]

        actor_emb = nn.Embed(2, self.component_emb_dim, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, name='Embed_0')(actor_id)
        event_emb = nn.Embed(13, self.component_emb_dim, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, name='Embed_1')(event_type)
        street_emb = nn.Embed(5, self.component_emb_dim, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, name='Embed_2')(street_id)
        time_mod = jnp.mod(time_id, jnp.int32(ACTION_SEQ_LEN))
        time_emb = nn.Embed(ACTION_SEQ_LEN, self.component_emb_dim, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, name='Embed_3')(time_mod)

        eps = jnp.bfloat16(1e-6)
        pot_base_safe = jnp.maximum(pot_base, eps)
        cont1 = jnp.log1p(to_call / pot_base_safe)
        cont2 = jnp.log1p(pot_base_safe)
        cont3 = jnp.log1p(eff_stack / pot_base_safe)
        continuous = jnp.concatenate([cont1, cont2, cont3], axis=-1)
        continuous_emb = nn.Dense(32, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, name='Dense_0')(continuous)

        concat = jnp.concatenate([actor_emb, event_emb, street_emb, time_emb, continuous_emb], axis=-1)
        combined = nn.Dense(self.emb_dim, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, name='Dense_1')(concat)

        encoded = NewV1TransformerEncoder(
            emb_dim=self.emb_dim,
            num_heads=self.num_heads,
            ffn_dim=self.ffn_dim,
            num_layers=self.num_layers,
            name='TransformerEncoder_0',
        )(combined, mask=action_mask)

        return encoded


class NewV1TossHoldemModel(nn.Module):
    """New V1 architecture (step_34000) - FROZEN.

    This is the architecture step_34000 was trained with.
    CardEncoder has TransformerBlock with FFN.
    """
    emb_dim: int = NEW_V1_EMB_DIM
    num_heads: int = NEW_V1_NUM_HEADS
    ffn_dim: int = NEW_V1_FFN_DIM
    num_layers: int = NEW_V1_NUM_TRANSFORMER_LAYERS
    num_cross_attention_layers: int = NEW_V1_NUM_CROSS_ATTENTION_LAYERS
    num_query_tokens: int = NEW_V1_NUM_QUERY_TOKENS
    processing_dim: int = NEW_V1_PROCESSING_DIM

    @nn.compact
    def __call__(
        self,
        card_tensor: chex.Array,
        action_tensor: chex.Array,
        action_mask: chex.Array,
    ) -> Tuple[chex.Array, chex.Array]:
        batch_size = card_tensor.shape[0]
        action_tensor = action_tensor.astype(jnp.bfloat16)

        card_repr, card_valid_mask = NewV1CardEncoder(emb_dim=self.emb_dim, name='CardEncoder_0')(card_tensor)
        action_repr = NewV1ActionEncoder(
            emb_dim=self.emb_dim,
            num_heads=self.num_heads,
            ffn_dim=self.ffn_dim,
            num_layers=self.num_layers,
            name='ActionEncoder_0',
        )(action_tensor, action_mask)

        card_repr = NewV1RMSNorm(name='card_norm')(card_repr)
        action_repr = NewV1RMSNorm(name='action_norm')(action_repr)

        context = jnp.concatenate([card_repr, action_repr], axis=1)
        context_mask = jnp.concatenate([card_valid_mask, action_mask], axis=1)

        query_tokens = self.param(
            'query_tokens',
            nn.initializers.lecun_normal(),
            (self.num_query_tokens, self.emb_dim),
            jnp.bfloat16,
        )

        fused = jnp.broadcast_to(
            query_tokens[None, :, :],
            (batch_size, self.num_query_tokens, self.emb_dim),
        ).astype(jnp.bfloat16)

        for layer_idx in range(self.num_cross_attention_layers):
            fused = NewV1CrossAttentionBlock(
                emb_dim=self.emb_dim,
                num_heads=self.num_heads,
                ffn_dim=self.ffn_dim,
                name=f'cross_attn_{layer_idx}',
            )(fused, context, context_mask)

        fused = NewV1RMSNorm(name='RMSNorm_0')(fused)

        flat = fused.reshape(batch_size, -1)
        processed = nn.Dense(512, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, name='Dense_0')(flat)
        processed = nn.relu(processed)
        processed = nn.Dense(self.processing_dim, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, name='Dense_1')(processed)
        processed = nn.relu(processed)

        # Betting policy head
        betting_hidden = nn.Dense(128, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, name='betting_hidden')(processed)
        betting_hidden = nn.relu(betting_hidden)
        betting_logits = nn.Dense(NUM_BETTING_ACTIONS, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, name='betting_out')(betting_hidden)

        # Discard policy head
        discard_hidden = nn.Dense(128, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, name='discard_hidden')(processed)
        discard_hidden = nn.relu(discard_hidden)
        discard_logits = nn.Dense(NUM_DISCARD_ACTIONS, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, name='discard_out')(discard_hidden)

        policy_logits = jnp.concatenate([betting_logits, discard_logits], axis=-1)

        # Value head
        value_hidden = nn.Dense(128, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, name='value_hidden')(processed)
        value_hidden = nn.relu(value_hidden)
        value = nn.Dense(1, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, name='value_out')(value_hidden)
        value = value.squeeze(-1)

        return policy_logits.astype(jnp.float32), value.astype(jnp.float32)


# Module-level model instances for JIT compatibility
_OLD_DIVERSITY_MODEL = OldTossHoldemModel()
_NEW_V1_DIVERSITY_MODEL = NewV1TossHoldemModel()


@jax.jit
def diversity_old_forward(
    params: Any,
    card: chex.Array,
    action_hist: chex.Array,
    action_mask: chex.Array,
) -> Tuple[chex.Array, chex.Array]:
    """Forward pass through old diversity bot.

    Converts new-style card encoding to old-style before forwarding.
    """
    # Convert card encoding from new (semantic type_ids) to old (position-based)
    card_old = _convert_card_encoding_new_to_old(card)
    return _OLD_DIVERSITY_MODEL.apply({"params": params}, card_old, action_hist, action_mask)


@jax.jit
def diversity_new_v1_forward(
    params: Any,
    card: chex.Array,
    action_hist: chex.Array,
    action_mask: chex.Array,
) -> Tuple[chex.Array, chex.Array]:
    """Forward pass through new v1 diversity bot (step_34000)."""
    return _NEW_V1_DIVERSITY_MODEL.apply({"params": params}, card, action_hist, action_mask)
