"""Transformer-based AlphaHoldem model for Toss or Hold'em.

Architecture:
- Card Encoder: (B, 9, 2) -> Embeddings (concat) -> 1-layer Transformer -> (B, 9, 128)
- Action Encoder: (B, 32, 8) -> Embeddings (concat) -> Projection -> 2-layer Transformer -> (B, 32, 128)
- History Encoder: (B, 64) -> Dense -> GELU -> (B, 1, 128)
- Cross-Attention Fusion: 2 layers of learnable queries (4 tokens) attending to context
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
ACTION_INPUT_DIM = 8    # [actor_id, event_type, street_id, time_id, to_call, pot_base, effective_stack, is_dealer]
HISTORY_INPUT_DIM = 64  # 64 opponent history features

EMB_DIM = 256
NUM_HEADS = 2
FFN_DIM = 512
NUM_TRANSFORMER_LAYERS = 2
NUM_CROSS_ATTENTION_LAYERS = 2
NUM_QUERY_TOKENS = 4
PROCESSING_DIM = 256

# Compute dtype - change this to jnp.float32 for full precision training
COMPUTE_DTYPE = jnp.bfloat16


# =============================================================================
# Transformer Components
# =============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Faster than LayerNorm as it skips mean centering.
    """
    epsilon: float = 1e-6

    @nn.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        dtype = x.dtype
        x = x.astype(jnp.float32)

        variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        x = x * jax.lax.rsqrt(variance + self.epsilon)

        scale = self.param(
            'scale',
            nn.initializers.ones,
            (x.shape[-1],),
            jnp.float32,
        )

        return (x * scale).astype(dtype)


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
        x = RMSNorm()(x)
        x = MultiHeadAttention(num_heads=self.num_heads, head_dim=head_dim)(x, x, x, mask)
        x = residual + x

        # Pre-norm FFN
        residual = x
        x = RMSNorm()(x)
        x = nn.Dense(self.ffn_dim, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16)(x)
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
        x = RMSNorm()(x)

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
        attn_input = RMSNorm()(query_tokens)
        context_normed = RMSNorm()(context)

        attn_output = MultiHeadAttention(
            num_heads=self.num_heads,
            head_dim=head_dim,
        )(attn_input, context_normed, context_normed, context_mask)

        post_attn = query_tokens + attn_output

        # === FFN Sub-block ===
        ffn_input = RMSNorm()(post_attn)

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
    """Encodes card tensor using embeddings and small transformer.

    Input: (B, 9, 2) where each entry is [card_index, type_id].

    Position semantics:
        0-2: Hero's private cards
        3-4: Flop (community)
        5: Hero's discard
        6: Villain's discard
        7: Turn
        8: River

    Output: (B, 9, emb_dim)

    Uses 1-layer transformer to learn card relationships (flush/straight patterns).
    """

    emb_dim: int = EMB_DIM
    rank_emb_dim: int = 48
    suit_emb_dim: int = 32
    # position_emb_dim computed to fill remainder: emb_dim - rank - suit
    num_heads: int = 2
    ffn_dim: int = 256

    @nn.compact
    def __call__(
        self,
        card_tensor: chex.Array,
    ) -> Tuple[chex.Array, chex.Array]:
        """
        Args:
            card_tensor: (B, 9, 2) int32 tensor with [card_index, type_id]

            card_index: 0-51 valid cards, -1 invalid/undealt
            type_id: 0-5 valid types (simplified scheme):
                0 = hero_private (positions 0-2)
                1 = flop (positions 3-4)
                2 = hero_discard (position 5)
                3 = villain_discard (position 6)
                4 = turn (position 7)
                5 = river (position 8)
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
        card_index_safe = jnp.where(card_index >= 0, card_index, 52).astype(jnp.int32)

        rank = (card_index_safe % 13).astype(jnp.int32)
        suit = (card_index_safe // 13).astype(jnp.int32)

        is_invalid = card_index < 0
        rank_safe = jnp.where(is_invalid, 13, rank).astype(jnp.int32)
        suit_safe = jnp.where(is_invalid, 4, suit).astype(jnp.int32)
        type_safe = jnp.where(type_id >= 0, type_id, NUM_CARD_TYPES).astype(jnp.int32)

        # -----------------------------
        # Embeddings: rank + suit + position = emb_dim
        # -----------------------------
        position_emb_dim = self.emb_dim - self.rank_emb_dim - self.suit_emb_dim

        rank_emb = nn.Embed(
            num_embeddings=14,  # 13 ranks + invalid
            features=self.rank_emb_dim,
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
        )(rank_safe)

        suit_emb = nn.Embed(
            num_embeddings=5,   # 4 suits + invalid
            features=self.suit_emb_dim,
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
        )(suit_safe)

        position_emb = nn.Embed(
            num_embeddings=NUM_CARD_TYPES + 1,  # 6 types + invalid
            features=position_emb_dim,
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
        )(type_safe)

        # Concatenate: (B, 9, emb_dim)
        concat = jnp.concatenate([rank_emb, suit_emb, position_emb], axis=-1)

        # -----------------------------
        # Small 1-layer transformer for card relationships
        # -----------------------------
        card_repr = TransformerBlock(
            emb_dim=self.emb_dim,
            num_heads=self.num_heads,
            ffn_dim=self.ffn_dim,
        )(concat, valid_mask)  # (B, 9, 128)

        return card_repr, valid_mask
    

class ActionEncoder(nn.Module):
    """Encodes action history tensor using embeddings and transformer.

    Input: (B, 32, 8) where each action entry contains:
        [0]: actor_id (0 or 1) - categorical
        [1]: event_type (0-12) - categorical
        [2]: street_id (0-4) - categorical
        [3]: time_id (0..T-1) - categorical
        [4]: to_call - continuous
        [5]: pot_base - continuous
        [6]: effective_stack - continuous
        [7]: is_dealer (0 or 1) - categorical (NEW)

    Output: (B, 32, emb_dim)

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
    is_dealer_emb_dim: int = 8     # 2 values (dealer/bb) (NEW)

    @nn.compact
    def __call__(
        self,
        action_tensor: chex.Array,
        action_mask: Optional[chex.Array] = None,
    ) -> chex.Array:
        """
        Args:
            action_tensor: (B, ACTION_SEQ_LEN, 8) float32 tensor
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
        is_dealer = action_tensor[..., 7].astype(jnp.int32)  # NEW

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

        # is_dealer embedding (NEW)
        is_dealer_emb = nn.Embed(
            num_embeddings=2,
            features=self.is_dealer_emb_dim,
            dtype=COMPUTE_DTYPE,
            param_dtype=COMPUTE_DTYPE,
        )(is_dealer)  # (B, 32, 8)

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
        # Updated to include is_dealer_emb
        concat = jnp.concatenate(
            [actor_emb, event_emb, street_time_emb, continuous_emb, is_dealer_emb],
            axis=-1
        )  # (B, 32, 8+48+48+24+8=136)

        # Project to emb_dim
        combined = nn.Dense(
            self.emb_dim,
            dtype=COMPUTE_DTYPE,
            param_dtype=COMPUTE_DTYPE,
        )(concat)  # (B, 32, emb_dim)

        # === Transformer ===
        encoded = TransformerEncoder(
            emb_dim=self.emb_dim,
            num_heads=self.num_heads,
            ffn_dim=self.ffn_dim,
            num_layers=self.num_layers,
        )(combined, mask=action_mask)

        return encoded


class HistoryEncoder(nn.Module):
    """Encodes opponent history statistics into a context token.
    
    Input: (B, 64) opponent history features
    Output: (B, 1, emb_dim) history representation token
    
    The 64 features are:
    
    PREFLOP (8):
    - vpip_dealer, vpip_bb
    - raise_freq_dealer, raise_freq_bb
    - fold_freq_dealer, fold_freq_bb
    - open_raise_size_avg_dealer, open_raise_size_avg_bb
    
    DISCARD (2):
    - discard_high_card_freq_dealer, discard_high_card_freq_bb
    
    FLOP (14):
    - fold_freq_dealer, fold_freq_bb
    - call_freq_dealer, call_freq_bb
    - raise_freq_dealer, raise_freq_bb
    - bet_freq_dealer, bet_freq_bb
    - avg_bet_size_dealer, avg_bet_size_bb
    - check_raise_freq
    - cbet_freq_dealer, cbet_freq_bb
    - donk_bet_freq_bb
    
    TURN (12):
    - fold_freq_dealer, fold_freq_bb
    - call_freq_dealer, call_freq_bb
    - raise_freq_dealer, raise_freq_bb
    - bet_freq_dealer, bet_freq_bb
    - avg_bet_size_dealer, avg_bet_size_bb
    - double_barrel_freq_dealer, double_barrel_freq_bb
    
    RIVER (14):
    - fold_freq_dealer, fold_freq_bb
    - call_freq_dealer, call_freq_bb
    - raise_freq_dealer, raise_freq_bb
    - bet_freq_dealer, bet_freq_bb
    - avg_bet_size_dealer, avg_bet_size_bb
    - triple_barrel_freq_dealer, triple_barrel_freq_bb
    - overbet_freq_dealer, overbet_freq_bb
    
    AGGREGATED (14):
    - showdown_winrate
    - allin_showdown_winrate
    - avg_streets_seen
    - rounds_observed
    - allin_freq_dealer, allin_freq_bb
    - allin_call_freq_dealer, allin_call_freq_bb
    - went_to_showdown_dealer, went_to_showdown_bb
    - reraise_freq_dealer, reraise_freq_bb
    """
    
    emb_dim: int = EMB_DIM
    hidden_dim: int = 128
    
    @nn.compact
    def __call__(self, history_tensor: chex.Array) -> chex.Array:
        """
        Args:
            history_tensor: (B, 64) float32 opponent history features
            
        Returns:
            history_repr: (B, 1, emb_dim) history representation token
        """
        history_tensor = history_tensor.astype(COMPUTE_DTYPE)
        
        # Dense -> GELU -> Dense to emb_dim
        x = nn.Dense(
            self.hidden_dim,
            dtype=COMPUTE_DTYPE,
            param_dtype=COMPUTE_DTYPE,
        )(history_tensor)  # (B, 128)
        x = nn.gelu(x)
        
        x = nn.Dense(
            self.emb_dim,
            dtype=COMPUTE_DTYPE,
            param_dtype=COMPUTE_DTYPE,
        )(x)  # (B, emb_dim)
        
        # Add sequence dimension: (B, emb_dim) -> (B, 1, emb_dim)
        history_repr = x[:, None, :]
        
        return history_repr


# =============================================================================
# Main Model
# =============================================================================

class TossHoldemModel(nn.Module):
    """Full transformer-based model for Toss or Hold'em.

    Architecture:
        1. Card Encoder: (B, 9, 2) -> (B, 9, 128) with embeddings (concatenated)
        2. Action Encoder: (B, ACTION_SEQ_LEN, 8) -> (B, ACTION_SEQ_LEN, emb_dim) with embeddings (concatenated)
        3. History Encoder: (B, 64) -> (B, 1, emb_dim) opponent stats
        4. Concatenate: (B, CARD_SEQ_LEN + ACTION_SEQ_LEN + 1, emb_dim)
        5. Multiple Cross-Attention Fusion layers with learnable queries: (B, 6, 128)
        6. Processing MLP: Flatten -> Dense -> gelu -> Dense -> gelu -> Dense -> (B, 256)
        7. Separate Output Heads:
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
        history_tensor: Optional[chex.Array] = None,
    ) -> Tuple[chex.Array, chex.Array]:
        """
        Args:
            card_tensor: (B, 9, 2) int32 - [card_index, type_id]
            action_tensor: (B, ACTION_SEQ_LEN, 8) float32 - action history features
            action_mask: (B, ACTION_SEQ_LEN) boolean - True for valid actions, False for padding
            history_tensor: (B, 64) float32 - opponent history features (optional)

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
        # 2. Encode history (if provided)
        # =================================================================

        if history_tensor is not None:
            history_repr = HistoryEncoder(
                emb_dim=self.emb_dim,
            )(history_tensor)  # (B, 1, emb_dim)
            history_mask = jnp.ones((batch_size, 1), dtype=jnp.bool_)
        else:
            # If no history provided, use zeros (backward compatible)
            history_repr = jnp.zeros((batch_size, 1, self.emb_dim), dtype=COMPUTE_DTYPE)
            history_mask = jnp.ones((batch_size, 1), dtype=jnp.bool_)
        
        # =================================================================
        # 3. Concatenate to form context
        # =================================================================

        # Normalize both modalities to same scale before fusion
        card_repr = RMSNorm(name='card_norm')(card_repr)
        action_repr = RMSNorm(name='action_norm')(action_repr)
        history_repr = RMSNorm(name='history_norm')(history_repr)

        context = jnp.concatenate([card_repr, action_repr, history_repr], axis=1)  # (B, 9+32+1=42, emb_dim)

        # Build context mask: use card valid mask + action mask + history mask
        if action_mask is not None:
            context_mask = jnp.concatenate([card_valid_mask, action_mask, history_mask], axis=1)  # (B, 42)
        else:
            context_mask = jnp.concatenate([
                card_valid_mask,
                jnp.ones((batch_size, ACTION_SEQ_LEN), dtype=jnp.bool_),
                history_mask
            ], axis=1)

        # =================================================================
        # 4. Learnable query tokens + MULTIPLE Cross-attention fusion layers
        # =================================================================

        # Initialize learnable query tokens (will be broadcast across batch)
        query_tokens = self.param(
            'query_tokens',
            nn.initializers.lecun_normal(),
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
            )(fused, context, context_mask)  # (B, 4, 128)

        # Final layer norm after cross-attention
        fused = RMSNorm()(fused)

        # =================================================================
        # 5. Processing MLP
        # =================================================================

        # Flatten: (B, 4, 128) -> (B, 512)
        flat = fused.reshape(batch_size, -1)

        # Layer 1: (B, 512) -> (B, 512)
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
        # 6. SEPARATE Output heads (policy and value have independent MLPs)
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
        action_tensor: (B, ACTION_SEQ_LEN, 8) action history tensor

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
    dummy_history = jnp.zeros((1, HISTORY_INPUT_DIM), dtype=jnp.float32)

    params = model.init(key, dummy_cards, dummy_actions, dummy_action_mask, dummy_history)

    return model, params


def count_parameters(params: dict) -> int:
    """Count the total number of parameters in the model.
    
    Args:
        params: Model parameters dict
        
    Returns:
        Total number of parameters
    """
    return sum(x.size for x in jax.tree_util.tree_leaves(params))