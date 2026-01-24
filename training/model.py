"""Transformer-based AlphaHoldem model for Toss or Hold'em.

Architecture:
- Card Encoder: (B, 9, 2) -> Embeddings (concat) -> Projection -> (B, 9, 128)
  (No transformer - card positions have fixed semantics)
- Action Encoder: (B, 32, 7) -> Embeddings (concat) -> Projection -> Transformer -> (B, 32, 128)
- Cross-Attention Fusion: Multiple layers of learnable queries attending to context
- Separate Output Heads: Policy (13), Value (1) with independent final layers

Designed for JAX/Flax with JIT compilation.
"""

from typing import Tuple, Optional
import jax
import jax.numpy as jnp
import flax.linen as nn
import chex


# =============================================================================
# Configuration
# =============================================================================

NUM_ACTIONS = 13
NUM_BETTING_ACTIONS = 10  # Actions 0-9: Fold, Check, Call, raises, All-in
NUM_DISCARD_ACTIONS = 3   # Actions 10-12: Discard card 0, 1, or 2
CARD_SEQ_LEN = 9        # [heropriv (x3), flop (x2), herodiscard, villaindiscard, turn, river]
ACTION_SEQ_LEN = 32
NUM_CARD_TYPES = 6      # heropriv=0, flop=1, herodiscard=2, villaindiscard=3, turn=4, river=5
CARD_INPUT_DIM = 2      # [card_index, type_id]
ACTION_INPUT_DIM = 7    # [actor_id, event_type, street_id, time_id, to_call, pot_base, effective_stack]

EMB_DIM = 256
NUM_HEADS = 2
FFN_DIM = 512
NUM_TRANSFORMER_LAYERS = 2
NUM_CROSS_ATTENTION_LAYERS = 2
NUM_QUERY_TOKENS = 6
PROCESSING_DIM = 256

# Compute dtype - change this to jnp.float32 for full precision training
COMPUTE_DTYPE = jnp.bfloat16


# =============================================================================
# Transformer Components
# =============================================================================

class MultiHeadAttention(nn.Module):
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

        query = query.astype(COMPUTE_DTYPE)
        key = key.astype(COMPUTE_DTYPE)
        value = value.astype(COMPUTE_DTYPE)

        q = nn.Dense(projection_dim, use_bias=False, dtype=COMPUTE_DTYPE, param_dtype=COMPUTE_DTYPE)(query)
        k = nn.Dense(projection_dim, use_bias=False, dtype=COMPUTE_DTYPE, param_dtype=COMPUTE_DTYPE)(key)
        v = nn.Dense(projection_dim, use_bias=False, dtype=COMPUTE_DTYPE, param_dtype=COMPUTE_DTYPE)(value)

        q = q.reshape(batch_size, q_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, kv_len, self.num_heads, self.head_dim)
        v = v.reshape(batch_size, kv_len, self.num_heads, self.head_dim)

        # Convert boolean mask to bias: True (valid) -> 0, False (invalid) -> -1e9
        # Shape: (B, kv_len) -> (B, 1, 1, kv_len) for broadcasting
        bias = jnp.where(mask[:, None, None, :], 0.0, -1e9)

        attn_output = jax.nn.dot_product_attention(q, k, v, bias=bias)

        attn_output = attn_output.reshape(batch_size, q_len, projection_dim)
        output = nn.Dense(input_dim, use_bias=False, dtype=COMPUTE_DTYPE, param_dtype=COMPUTE_DTYPE)(attn_output)

        return output


class TransformerBlock(nn.Module):
    """Single transformer encoder block with pre-norm."""

    emb_dim: int
    num_heads: int
    ffn_dim: int

    @nn.compact
    def __call__(
        self,
        x: chex.Array,
        mask: Optional[chex.Array] = None,
    ) -> chex.Array:
        """
        Args:
            x: (B, S, D)
            mask: (B, S) boolean mask where True = valid

        Returns:
            output: (B, S, D)
        """
        x = x.astype(COMPUTE_DTYPE)
        head_dim = self.emb_dim // self.num_heads
        
        # Pre-norm self-attention
        residual = x
        x = nn.LayerNorm(dtype=COMPUTE_DTYPE, param_dtype=COMPUTE_DTYPE)(x)
        x = MultiHeadAttention(num_heads=self.num_heads, head_dim=head_dim)(x, x, x, mask)
        x = residual + x

        # Pre-norm FFN
        residual = x
        x = nn.LayerNorm(dtype=COMPUTE_DTYPE, param_dtype=COMPUTE_DTYPE)(x)
        x = nn.Dense(self.ffn_dim, dtype=COMPUTE_DTYPE, param_dtype=COMPUTE_DTYPE)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.emb_dim, dtype=COMPUTE_DTYPE, param_dtype=COMPUTE_DTYPE)(x)
        x = residual + x

        return x


class TransformerEncoder(nn.Module):
    """Stack of transformer encoder blocks."""

    emb_dim: int
    num_heads: int
    ffn_dim: int
    num_layers: int

    @nn.compact
    def __call__(
        self,
        x: chex.Array,
        mask: Optional[chex.Array] = None,
    ) -> chex.Array:
        """
        Args:
            x: (B, S, D)
            mask: (B, S) boolean mask where True = valid

        Returns:
            output: (B, S, D)
        """
        x = x.astype(COMPUTE_DTYPE)

        # Use explicit layer names for better jittability
        for layer_idx in range(self.num_layers):
            x = TransformerBlock(
                emb_dim=self.emb_dim,
                num_heads=self.num_heads,
                ffn_dim=self.ffn_dim,
                name=f'layer_{layer_idx}',
            )(x, mask)

        # Final layer norm
        x = nn.LayerNorm(dtype=COMPUTE_DTYPE, param_dtype=COMPUTE_DTYPE)(x)

        return x


class CrossAttentionBlock(nn.Module):
    """Cross-attention block where queries attend to context, followed by FFN."""

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
        """
        Args:
            query_tokens: (B, Q, D) learnable query tokens
            context: (B, C, D) concatenated card + action representations
            context_mask: (B, C) boolean mask where True = valid

        Returns:
            output: (B, Q, D)
        """
        query_tokens = query_tokens.astype(COMPUTE_DTYPE)
        context = context.astype(COMPUTE_DTYPE)
        head_dim = self.emb_dim // self.num_heads

        # === Cross-Attention Sub-block ===
        attn_input = nn.LayerNorm(dtype=COMPUTE_DTYPE, param_dtype=COMPUTE_DTYPE)(query_tokens)
        context_normed = nn.LayerNorm(dtype=COMPUTE_DTYPE, param_dtype=COMPUTE_DTYPE)(context)

        attn_output = MultiHeadAttention(
            num_heads=self.num_heads,
            head_dim=head_dim,
        )(attn_input, context_normed, context_normed, context_mask)

        post_attn = query_tokens + attn_output

        # === FFN Sub-block ===
        ffn_input = nn.LayerNorm(dtype=COMPUTE_DTYPE, param_dtype=COMPUTE_DTYPE)(post_attn)

        ffn_hidden = nn.Dense(
            self.ffn_dim,
            dtype=COMPUTE_DTYPE,
            param_dtype=COMPUTE_DTYPE,
        )(ffn_input)
        ffn_hidden = nn.gelu(ffn_hidden)

        ffn_out = nn.Dense(
            self.emb_dim,
            dtype=COMPUTE_DTYPE,
            param_dtype=COMPUTE_DTYPE,
        )(ffn_hidden)

        output = post_attn + ffn_out

        return output


# =============================================================================
# Encoder Modules
# =============================================================================

class CardEncoder(nn.Module):
    """Encodes card tensor using embeddings and MLP.

    Input: (B, 9, 2) where each entry is [card_index, type_id].
    - card_index: 0-51 (canonized suits), -1 for invalid
    - type_id: 0-5, -1 for invalid

    Position semantics (fixed, not learned):
        0-2: Hero's private cards (sorted by rank, all type_id=0)
        3-4: Flop (sorted by rank, all type_id=1)
        5: Hero's discard (type_id=2)
        6: Villain's discard (type_id=3)
        7: Turn (type_id=4)
        8: River (type_id=5)

    Cards are sorted by rank within hero hand and flop groups.
    Suits are canonized (first suit seen = 0, second = 1, etc.).

    Architecture:
    - Embed rank, suit, type independently
    - Concatenate embeddings
    - MLP to project to emb_dim
    - Cross-attention (downstream) learns card relationships

    Output: (B, 9, emb_dim)
    """

    emb_dim: int = EMB_DIM
    rank_emb_dim: int = 64
    suit_emb_dim: int = 16
    type_emb_dim: int = 48
    hidden_dim: int = 256

    @nn.compact
    def __call__(
        self,
        card_tensor: chex.Array,
    ) -> Tuple[chex.Array, chex.Array]:
        """
        Args:
            card_tensor: (B, 9, 2) int32 tensor with [card_index, type_id]

            card_index: 0-51 valid cards (canonized suits), -1 invalid/undealt
            type_id: 0-5 valid types:
                0 = heropriv (all 3 hero private cards)
                1 = flop (both flop cards)
                2 = herodiscard
                3 = villaindiscard
                4 = turn
                5 = river
                -1 = invalid/unused

        Returns:
            card_representation: (B, 9, emb_dim)
            valid_mask: (B, 9) boolean mask where True = valid card_index
        """
        # (B, 9)
        card_index = card_tensor[..., 0].astype(jnp.int32)
        type_id = card_tensor[..., 1].astype(jnp.int32)

        # Valid token if we have a real card
        valid_mask = card_index >= 0  # (B, 9)

        # -----------------------------
        # Safe / derived categorical ids
        # -----------------------------
        # Map invalid card_index (-1) to a safe placeholder 52 for computations
        card_index_safe = jnp.where(card_index >= 0, card_index, 52).astype(jnp.int32)

        # Derive rank/suit for valid cards
        # rank: 0-12, suit: 0-3
        rank = (card_index_safe % 13).astype(jnp.int32)
        suit = (card_index_safe // 13).astype(jnp.int32)

        # Use dedicated "invalid" buckets for embeddings:
        # rank_invalid = 13, suit_invalid = 4
        is_invalid = card_index < 0
        rank_safe = jnp.where(is_invalid, 13, rank).astype(jnp.int32)  # 0-12 + 13 invalid
        suit_safe = jnp.where(is_invalid, 4, suit).astype(jnp.int32)   # 0-3 + 4 invalid

        # Type embedding: 0-5 + 6 invalid
        type_safe = jnp.where(type_id >= 0, type_id, NUM_CARD_TYPES).astype(jnp.int32)

        # -----------------------------
        # Embeddings: CONCATENATE rank, suit, type
        # Dimensions scaled by information content
        # -----------------------------
        rank_emb = nn.Embed(
            num_embeddings=14,  # 13 ranks + invalid
            features=self.rank_emb_dim,
            dtype=COMPUTE_DTYPE,
            param_dtype=COMPUTE_DTYPE,
        )(rank_safe)  # (B, 9, 64)

        suit_emb = nn.Embed(
            num_embeddings=5,   # 4 suits + invalid
            features=self.suit_emb_dim,
            dtype=COMPUTE_DTYPE,
            param_dtype=COMPUTE_DTYPE,
        )(suit_safe)  # (B, 9, 16)

        type_emb = nn.Embed(
            num_embeddings=NUM_CARD_TYPES + 1,  # 6 types + invalid
            features=self.type_emb_dim,
            dtype=COMPUTE_DTYPE,
            param_dtype=COMPUTE_DTYPE,
        )(type_safe)  # (B, 9, 48)

        # -----------------------------
        # Concatenate and MLP
        # -----------------------------
        concat = jnp.concatenate([rank_emb, suit_emb, type_emb], axis=-1)  # (B, 9, 128)

        # MLP: concat -> hidden -> emb_dim
        hidden = nn.Dense(
            self.hidden_dim,
            dtype=COMPUTE_DTYPE,
            param_dtype=COMPUTE_DTYPE,
        )(concat)
        hidden = nn.gelu(hidden)

        card_repr = nn.Dense(
            self.emb_dim,
            dtype=COMPUTE_DTYPE,
            param_dtype=COMPUTE_DTYPE,
        )(hidden)  # (B, 9, emb_dim)

        return card_repr, valid_mask

class ActionEncoder(nn.Module):
    """Encodes action history tensor using embeddings and transformer.

    Input: (B, 32, 7) where each action entry contains:
        [0]: actor_id (0 or 1) - categorical
        [1]: event_type (0-12) - categorical
        [2]: street_id (0-4) - categorical
        [3]: time_id (0..T-1) - categorical
        [4]: to_call - continuous
        [5]: pot_base - continuous
        [6]: effective_stack - continuous

    Output: (B, 30, emb_dim)

    Uses CONCATENATION of embeddings (not addition) to preserve information.
    """

    emb_dim: int = EMB_DIM
    num_heads: int = NUM_HEADS
    ffn_dim: int = FFN_DIM
    num_layers: int = NUM_TRANSFORMER_LAYERS
    # Embedding dims scaled by information content
    actor_emb_dim: int = 8      # 2 values (hero/villain)
    event_emb_dim: int = 48     # 13 action types (high info)
    street_time_emb_dim: int = 48  # 160 combos (when it happened)
    continuous_emb_dim: int = 24   # projected continuous features

    @nn.compact
    def __call__(
        self,
        action_tensor: chex.Array,
        action_mask: Optional[chex.Array] = None,
    ) -> chex.Array:
        """
        Args:
            action_tensor: (B, ACTION_SEQ_LEN, 7) float32 tensor
            action_mask: (B, ACTION_SEQ_LEN) boolean mask where True = valid action

        Returns:
            action_representation: (B, ACTION_SEQ_LEN, emb_dim)
        """
        action_tensor = action_tensor.astype(COMPUTE_DTYPE)

        # === Extract components ===
        actor_id = action_tensor[..., 0].astype(jnp.int32)
        event_type = action_tensor[..., 1].astype(jnp.int32)
        street_id = action_tensor[..., 2].astype(jnp.int32)
        to_call = action_tensor[..., 4:5]
        pot_base = action_tensor[..., 5:6]
        eff_stack = action_tensor[..., 6:7]

        # === Categorical Embeddings (dims scaled by information content) ===
        actor_emb = nn.Embed(
            num_embeddings=2,
            features=self.actor_emb_dim,
            dtype=COMPUTE_DTYPE,
            param_dtype=COMPUTE_DTYPE,
        )(actor_id)  # (B, 32, 8)

        event_emb = nn.Embed(
            num_embeddings=13,
            features=self.event_emb_dim,
            dtype=COMPUTE_DTYPE,
            param_dtype=COMPUTE_DTYPE,
        )(event_type)  # (B, 32, 48)

        # Compute position within street (resets to 0 when street changes)
        seq_len = street_id.shape[-1]
        indices = jnp.arange(seq_len, dtype=jnp.int32)

        # Detect street boundaries: where street differs from previous position
        prev_street = jnp.concatenate([street_id[..., :1] - 1, street_id[..., :-1]], axis=-1)
        is_new_street = (street_id != prev_street)

        # Mark reset positions with their index, others with -1
        reset_markers = jnp.where(is_new_street, indices, jnp.int32(-1))

        # Cumulative maximum to propagate the last reset index forward
        last_reset_idx = jax.lax.associative_scan(jnp.maximum, reset_markers, axis=-1)

        # Position within street = current index - index of last reset
        position_in_street = indices - last_reset_idx

        # Combined embedding: (street, position_in_street) → unique representation
        combined_idx = street_id * ACTION_SEQ_LEN + position_in_street
        street_time_emb = nn.Embed(
            num_embeddings=5 * ACTION_SEQ_LEN,  # 5 streets × positions per street
            features=self.street_time_emb_dim,
            dtype=COMPUTE_DTYPE,
            param_dtype=COMPUTE_DTYPE,
        )(combined_idx)  # (B, 32, 48)

        # === Continuous Features Projection ===
        eps = COMPUTE_DTYPE(1e-6)
        pot_base_safe = jnp.maximum(pot_base, eps)
        cont1 = jnp.log1p(to_call / pot_base_safe)
        cont2 = jnp.log1p(pot_base_safe)
        cont3 = jnp.log1p(eff_stack / pot_base_safe)
        continuous = jnp.concatenate([cont1, cont2, cont3], axis=-1)  # (B, 32, 3)
        continuous_emb = nn.Dense(
            self.continuous_emb_dim,
            dtype=COMPUTE_DTYPE,
            param_dtype=COMPUTE_DTYPE,
        )(continuous)  # (B, 32, 24)

        # === CONCATENATE all embeddings (not add) ===
        concat = jnp.concatenate(
            [actor_emb, event_emb, street_time_emb, continuous_emb],
            axis=-1
        )  # (B, 32, 128)

        # Project to emb_dim
        combined = nn.Dense(
            self.emb_dim,
            dtype=COMPUTE_DTYPE,
            param_dtype=COMPUTE_DTYPE,
        )(concat)  # (B, 30, emb_dim)

        # === Transformer ===
        encoded = TransformerEncoder(
            emb_dim=self.emb_dim,
            num_heads=self.num_heads,
            ffn_dim=self.ffn_dim,
            num_layers=self.num_layers,
        )(combined, mask=action_mask)

        return encoded


# =============================================================================
# Main Model
# =============================================================================

class TossHoldemModel(nn.Module):
    """Full transformer-based model for Toss or Hold'em.

    Architecture:
        1. Card Encoder: (B, 9, 2) -> (B, 9, 128) with embeddings (concatenated)
        2. Action Encoder: (B, ACTION_SEQ_LEN, 7) -> (B, ACTION_SEQ_LEN, emb_dim) with embeddings (concatenated)
        3. Concatenate: (B, CARD_SEQ_LEN + ACTION_SEQ_LEN, emb_dim)
        4. Multiple Cross-Attention Fusion layers with learnable queries: (B, 8, 128)
        5. Processing MLP: Flatten -> Dense -> gelu -> Dense -> gelu -> Dense -> (B, 256)
        6. Separate Output Heads:
           - Policy: Independent MLP -> (B, 13)
           - Value: Independent MLP with stop_gradient -> (B,)
    """

    emb_dim: int = EMB_DIM
    num_heads: int = NUM_HEADS
    ffn_dim: int = FFN_DIM
    num_layers: int = NUM_TRANSFORMER_LAYERS
    num_cross_attention_layers: int = NUM_CROSS_ATTENTION_LAYERS
    num_query_tokens: int = NUM_QUERY_TOKENS
    processing_dim: int = PROCESSING_DIM
    num_actions: int = NUM_ACTIONS

    @nn.compact
    def __call__(
        self,
        card_tensor: chex.Array,
        action_tensor: chex.Array,
        action_mask: chex.Array,
    ) -> Tuple[chex.Array, chex.Array]:
        """
        Args:
            card_tensor: (B, 9, 2) int32 - [card_index, type_id]
            action_tensor: (B, ACTION_SEQ_LEN, 7) float32 - action history features
            action_mask: (B, ACTION_SEQ_LEN) boolean - True for valid actions, False for padding

        Returns:
            policy_logits: (B, 13) - raw logits for action distribution
            value: (B,) - estimated state value
        """
        batch_size = card_tensor.shape[0]
        action_tensor = action_tensor.astype(COMPUTE_DTYPE)

        # =================================================================
        # 1. Encode cards and actions
        # =================================================================

        card_repr, card_valid_mask = CardEncoder(
            emb_dim=self.emb_dim,
        )(card_tensor)  # (B, 9, 128), (B, 9)

        action_repr = ActionEncoder(
            emb_dim=self.emb_dim,
            num_heads=self.num_heads,
            ffn_dim=self.ffn_dim,
            num_layers=self.num_layers,
        )(action_tensor, action_mask)  # (B, ACTION_SEQ_LEN, emb_dim)
        
        # =================================================================
        # 2. Concatenate to form context
        # =================================================================

        context = jnp.concatenate([card_repr, action_repr], axis=1)  # (B, 39, 128)

        # Build context mask: use card valid mask + action mask
        if action_mask is not None:
            context_mask = jnp.concatenate([card_valid_mask, action_mask], axis=1)  # (B, 39)
        else:
            context_mask = jnp.concatenate([
                card_valid_mask,
                jnp.ones((batch_size, ACTION_SEQ_LEN), dtype=jnp.bool_)
            ], axis=1)

        # =================================================================
        # 3. Learnable query tokens + MULTIPLE Cross-attention fusion layers
        # =================================================================

        # Initialize learnable query tokens (will be broadcast across batch)
        query_tokens = self.param(
            'query_tokens',
            nn.initializers.normal(stddev=0.02),
            (self.num_query_tokens, self.emb_dim),
            COMPUTE_DTYPE,
        )

        # Broadcast to batch dimension: (Q, D) -> (B, Q, D)
        fused = jnp.broadcast_to(
            query_tokens[None, :, :],
            (batch_size, self.num_query_tokens, self.emb_dim),
        )
        fused = fused.astype(COMPUTE_DTYPE)

        # Apply multiple cross-attention layers for richer fusion
        for layer_idx in range(self.num_cross_attention_layers):
            fused = CrossAttentionBlock(
                emb_dim=self.emb_dim,
                num_heads=self.num_heads,
                ffn_dim=self.ffn_dim,
                name=f'cross_attn_{layer_idx}',
            )(fused, context, context_mask)  # (B, 6, 128)

        # Final layer norm after cross-attention
        fused = nn.LayerNorm(dtype=COMPUTE_DTYPE, param_dtype=COMPUTE_DTYPE)(fused)

        # =================================================================
        # 4. Processing MLP
        # =================================================================

        # Flatten: (B, 6, 128) -> (B, 768)
        flat = fused.reshape(batch_size, -1)

        # Layer 1: (B, 768) -> (B, 512)
        processed = nn.Dense(
            512,
            dtype=COMPUTE_DTYPE,
            param_dtype=COMPUTE_DTYPE,
        )(flat)
        processed = nn.gelu(processed)

        # Layer 2: (B, 512) -> (B, 256)
        processed = nn.Dense(
            self.processing_dim,
            dtype=COMPUTE_DTYPE,
            param_dtype=COMPUTE_DTYPE,
        )(processed)
        processed = nn.gelu(processed)

        # =================================================================
        # 5. SEPARATE Output heads (policy and value have independent MLPs)
        # =================================================================

        # --- Betting Policy head (actions 0-9) ---
        betting_hidden = nn.Dense(
            128,
            dtype=COMPUTE_DTYPE,
            param_dtype=COMPUTE_DTYPE,
            name='betting_hidden',
        )(processed)
        betting_hidden = nn.gelu(betting_hidden)
        betting_logits = nn.Dense(
            NUM_BETTING_ACTIONS,
            dtype=COMPUTE_DTYPE,
            param_dtype=COMPUTE_DTYPE,
            name='betting_out',
        )(betting_hidden)

        # --- Discard Policy head (actions 10-12) ---
        discard_hidden = nn.Dense(
            128,
            dtype=COMPUTE_DTYPE,
            param_dtype=COMPUTE_DTYPE,
            name='discard_hidden',
        )(processed)
        discard_hidden = nn.gelu(discard_hidden)
        discard_logits = nn.Dense(
            NUM_DISCARD_ACTIONS,
            dtype=COMPUTE_DTYPE,
            param_dtype=COMPUTE_DTYPE,
            name='discard_out',
        )(discard_hidden)

        # Concatenate for backward compatibility: (B, 10) + (B, 3) -> (B, 13)
        policy_logits = jnp.concatenate([betting_logits, discard_logits], axis=-1)

        # --- Value head: Independent MLP ---
        # Allow value gradients to flow into the shared representation
        value_input = processed
        value_hidden = nn.Dense(
            128,
            dtype=COMPUTE_DTYPE,
            param_dtype=COMPUTE_DTYPE,
            name='value_hidden',
        )(value_input)
        value_hidden = nn.gelu(value_hidden)
        value = nn.Dense(
            1,
            dtype=COMPUTE_DTYPE,
            param_dtype=COMPUTE_DTYPE,
            name='value_out',
        )(value_hidden)
        value = value.squeeze(-1)

        # Cast outputs to float32 for loss computation
        policy_logits = policy_logits.astype(jnp.float32)
        value = value.astype(jnp.float32)

        return policy_logits, value


# =============================================================================
# Utility Functions
# =============================================================================

def create_action_mask(action_tensor: chex.Array) -> chex.Array:
    """Create a boolean mask for valid actions in the action tensor.

    An action is considered valid if any of its features are non-zero.

    Args:
        action_tensor: (B, ACTION_SEQ_LEN, 7) action history tensor

    Returns:
        mask: (B, ACTION_SEQ_LEN) boolean mask where True = valid action
    """
    # Sum across features - if any feature is non-zero, the action is valid
    feature_sum = jnp.sum(jnp.abs(action_tensor), axis=-1)  # (B, ACTION_SEQ_LEN)
    mask = feature_sum > 0
    return mask


def apply_legal_action_mask(
    policy_logits: chex.Array,
    legal_mask: chex.Array,
) -> chex.Array:
    """Apply legal action masking to policy logits.

    Args:
        policy_logits: (B, 13) raw policy logits
        legal_mask: (B, 13) boolean mask where True = legal action

    Returns:
        masked_logits: (B, 13) logits with illegal actions set to -1e9
    """
    # Use -1e9 instead of finfo.min for better numerical stability
    neg_inf = -1e9
    masked_logits = jnp.where(legal_mask, policy_logits, neg_inf)
    return masked_logits


def init_model(key: chex.PRNGKey) -> Tuple[TossHoldemModel, dict]:
    """Initialize the model and return parameters.

    Args:
        key: JAX random key for initialization

    Returns:
        model: TossHoldemModel instance
        params: Initialized parameters (includes 'params' key for model params)
    """
    model = TossHoldemModel()

    # Create dummy inputs for initialization
    dummy_cards = jnp.zeros((1, CARD_SEQ_LEN, CARD_INPUT_DIM), dtype=jnp.int32)
    dummy_actions = jnp.zeros((1, ACTION_SEQ_LEN, ACTION_INPUT_DIM), dtype=COMPUTE_DTYPE)
    dummy_action_mask = jnp.ones((1, ACTION_SEQ_LEN), dtype=jnp.bool_)

    params = model.init(key, dummy_cards, dummy_actions, dummy_action_mask)

    return model, params


def count_parameters(params: dict) -> int:
    """Count the total number of parameters in the model.
    
    Args:
        params: Model parameters dict
        
    Returns:
        Total number of parameters
    """
    return sum(x.size for x in jax.tree_util.tree_leaves(params)) 