"""Transformer-based AlphaHoldem model for Toss or Hold'em.

Architecture:
- Card Encoder: (B, 8, 2) -> Embeddings -> Transformer -> (B, 8, 128)
- Action Encoder: (B, 30, 7) -> Embeddings + Dense -> Transformer -> (B, 30, 128)
- Cross-Attention Fusion: Learnable queries attend to concatenated context
- Output Heads: Policy (13), Value (1)

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
CARD_SEQ_LEN = 9        # [heropriv0, heropriv1, heropriv2, flop0, flop1, oopdiscard, dealerdiscard, turn, river]
ACTION_SEQ_LEN = 30
NUM_CARD_TYPES = 9
CARD_INPUT_DIM = 2      # [dealer_id, card_index]
ACTION_INPUT_DIM = 7    # [actor_id, event_type, street_id, time_id, to_call, pot_base, effective_stack]

EMB_DIM = 128
NUM_HEADS = 4
FFN_DIM = 512
NUM_TRANSFORMER_LAYERS = 3
NUM_QUERY_TOKENS = 8
PROCESSING_DIM = 256


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
        
        q = nn.Dense(projection_dim, use_bias=False, dtype=jnp.bfloat16, param_dtype=jnp.float32)(query)
        k = nn.Dense(projection_dim, use_bias=False, dtype=jnp.bfloat16, param_dtype=jnp.float32)(key)
        v = nn.Dense(projection_dim, use_bias=False, dtype=jnp.bfloat16, param_dtype=jnp.float32)(value)
        
        q = q.reshape(batch_size, q_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, kv_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, kv_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        scale = jnp.sqrt(jnp.float32(self.head_dim))
        attn_weights = jnp.einsum('bhqd,bhkd->bhqk', q, k) / scale
        
        if mask is not None:
            mask = mask[:, None, None, :]
            attn_weights = jnp.where(mask, attn_weights, jnp.finfo(jnp.bfloat16).min)
        
        attn_weights = jax.nn.softmax(attn_weights, axis=-1)
        attn_output = jnp.einsum('bhqk,bhkd->bhqd', attn_weights, v)
        
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, q_len, projection_dim)
        output = nn.Dense(input_dim, use_bias=False, dtype=jnp.bfloat16, param_dtype=jnp.float32)(attn_output)
        
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
        head_dim = self.emb_dim // self.num_heads
        
        # Pre-norm self-attention
        residual = x
        x = nn.LayerNorm(dtype=jnp.bfloat16, param_dtype=jnp.float32)(x)
        x = MultiHeadAttention(
            num_heads=self.num_heads,
            head_dim=head_dim,
        )(x, x, x, mask)
        x = residual + x
        
        # Pre-norm FFN
        residual = x
        x = nn.LayerNorm(dtype=jnp.bfloat16, param_dtype=jnp.float32)(x)
        x = nn.Dense(
            self.ffn_dim,
            dtype=jnp.bfloat16,
            param_dtype=jnp.float32,
        )(x)
        x = nn.gelu(x)
        x = nn.Dense(
            self.emb_dim,
            dtype=jnp.bfloat16,
            param_dtype=jnp.float32,
        )(x)
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
        for _ in range(self.num_layers):
            x = TransformerBlock(
                emb_dim=self.emb_dim,
                num_heads=self.num_heads,
                ffn_dim=self.ffn_dim,
            )(x, mask)
        
        # Final layer norm
        x = nn.LayerNorm(dtype=jnp.bfloat16, param_dtype=jnp.float32)(x)
        
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
        head_dim = self.emb_dim // self.num_heads
        
        # === Cross-Attention Sub-block ===
        attn_input = nn.LayerNorm(dtype=jnp.bfloat16, param_dtype=jnp.float32)(query_tokens)
        context_normed = nn.LayerNorm(dtype=jnp.bfloat16, param_dtype=jnp.float32)(context)
        
        attn_output = MultiHeadAttention(
            num_heads=self.num_heads,
            head_dim=head_dim,
        )(attn_input, context_normed, context_normed, context_mask)
        
        post_attn = query_tokens + attn_output
        
        # === FFN Sub-block ===
        ffn_input = nn.LayerNorm(dtype=jnp.bfloat16, param_dtype=jnp.float32)(post_attn)
        
        ffn_hidden = nn.Dense(
            self.ffn_dim,
            dtype=jnp.bfloat16,
            param_dtype=jnp.float32,
        )(ffn_input)
        ffn_hidden = nn.gelu(ffn_hidden)
        
        ffn_out = nn.Dense(
            self.emb_dim,
            dtype=jnp.bfloat16,
            param_dtype=jnp.float32,
        )(ffn_hidden)
        
        output = post_attn + ffn_out
        
        return output


# =============================================================================
# Encoder Modules
# =============================================================================

class CardEncoder(nn.Module):
    """Encodes card tensor using embeddings and transformer.
    
    Input: (B, 8, 2) where each entry is [card_index, type_id]. 
    
    Position semantics:
        0-1: Player's hole cards
        2-3: Initial flop (community)
        4: OOP player's discard (non-dealer discards first)
        5: Dealer's discard
        6: Turn
        7: River
    
    Output: (B, 8, emb_dim)
    """
    
    emb_dim: int = EMB_DIM
    num_heads: int = NUM_HEADS
    ffn_dim: int = FFN_DIM
    num_layers: int = NUM_TRANSFORMER_LAYERS

    @nn.compact
    def __call__(self, card_tensor: chex.Array) -> Tuple[chex.Array, chex.Array]:
        """
        Args:
            card_tensor: (B, 9, 2) int32 tensor with [card_index, type_id]

            card_index: 0-51 valid cards, -1 invalid/undealt
            type_id: 0-8 valid types:
                0 = heropriv0
                1 = heropriv1
                2 = heropriv2
                3 = flop0
                4 = flop1
                5 = herodiscard
                6 = villaindiscard
                7 = turn
                8 = river
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

        # Type embedding: 0-8 + 9 invalid
        # (NUM_CARD_TYPES assumed = 9)
        type_safe = jnp.where(type_id >= 0, type_id, NUM_CARD_TYPES).astype(jnp.int32)

        # -----------------------------
        # Embeddings: rank + suit + type
        # -----------------------------
        rank_emb = nn.Embed(
            num_embeddings=14,  # 13 ranks + invalid
            features=self.emb_dim,
            dtype=jnp.bfloat16,
            param_dtype=jnp.float32,
        )(rank_safe)  # (B, 9, D)

        suit_emb = nn.Embed(
            num_embeddings=5,   # 4 suits + invalid
            features=self.emb_dim,
            dtype=jnp.bfloat16,
            param_dtype=jnp.float32,
        )(suit_safe)  # (B, 9, D)

        type_emb = nn.Embed(
            num_embeddings=NUM_CARD_TYPES + 1,  # 9 types + invalid
            features=self.emb_dim,
            dtype=jnp.bfloat16,
            param_dtype=jnp.float32,
        )(type_safe)  # (B, 9, D)

        combined = rank_emb + suit_emb + type_emb  # (B, 9, D)

        # -----------------------------
        # Transformer encode
        # -----------------------------
        encoded = TransformerEncoder(
            emb_dim=self.emb_dim,
            num_heads=self.num_heads,
            ffn_dim=self.ffn_dim,
            num_layers=self.num_layers,
        )(combined, mask=valid_mask)

        return encoded, valid_mask


class ActionEncoder(nn.Module):
    """Encodes action history tensor using embeddings and transformer.
    
    Input: (B, 30, 7) where each action entry contains:
        [0]: actor_id (0 or 1) - categorical
        [1]: event_type (0-12) - categorical
        [2]: street_id (0-4) - categorical
        [3]: time_id (0..T-1) - categorical
        [4]: to_call - continuous
        [5]: pot_base - continuous
        [6]: effective_stack - continuous
    
    Output: (B, 30, emb_dim)
    """
    
    emb_dim: int = EMB_DIM
    num_heads: int = NUM_HEADS
    ffn_dim: int = FFN_DIM
    num_layers: int = NUM_TRANSFORMER_LAYERS
    
    @nn.compact
    def __call__(
        self,
        action_tensor: chex.Array,
        action_mask: Optional[chex.Array] = None,
    ) -> chex.Array:
        """
        Args:
            action_tensor: (B, 30, 7) float32 tensor
            action_mask: (B, 30) boolean mask where True = valid action
            
        Returns:
            action_representation: (B, 30, emb_dim)
        """
        # === Extract components ===
        actor_id = action_tensor[..., 0].astype(jnp.int32)
        event_type = action_tensor[..., 1].astype(jnp.int32)
        street_id = action_tensor[..., 2].astype(jnp.int32)
        time_id = action_tensor[..., 3].astype(jnp.int32)
        to_call = action_tensor[..., 4:5]
        pot_base = action_tensor[..., 5:6]
        eff_stack = action_tensor[..., 6:7]
        
        # === Categorical Embeddings ===
        actor_emb = nn.Embed(
            num_embeddings=2,
            features=self.emb_dim,
            dtype=jnp.bfloat16,
            param_dtype=jnp.float32,
        )(actor_id)  # (B, 30, emb_dim)

        event_emb = nn.Embed(
            num_embeddings=13,
            features=self.emb_dim,
            dtype=jnp.bfloat16,
            param_dtype=jnp.float32,
        )(event_type)  # (B, 30, emb_dim)

        street_emb = nn.Embed(
            num_embeddings=5,
            features=self.emb_dim,
            dtype=jnp.bfloat16,
            param_dtype=jnp.float32,
        )(street_id)  # (B, 30, emb_dim)

        time_mod = jnp.mod(time_id, jnp.int32(ACTION_SEQ_LEN))
        time_emb = nn.Embed(
            num_embeddings=ACTION_SEQ_LEN,
            features=self.emb_dim,
            dtype=jnp.bfloat16,
            param_dtype=jnp.float32,
        )(time_mod)  # (B, 30, emb_dim)

        # === Continuous Features Projection ===
        eps = jnp.float32(1e-6)
        pot_base_safe = jnp.maximum(pot_base, eps)
        cont1 = jnp.log1p(to_call / pot_base_safe)
        cont2 = jnp.log1p(pot_base_safe)
        cont3 = jnp.log1p(eff_stack / pot_base_safe)
        continuous = jnp.concatenate([cont1, cont2, cont3], axis=-1)  # (B, 30, 3)
        continuous_emb = nn.Dense(
            self.emb_dim,
            dtype=jnp.bfloat16,
            param_dtype=jnp.float32,
        )(continuous.astype(jnp.float32))  # (B, 30, emb_dim)

        # === Combine: additive ===
        combined = actor_emb + event_emb + street_emb + time_emb + continuous_emb
        
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
        1. Card Encoder: (B, 8, 2) -> (B, 8, 128) with embeddings
        2. Action Encoder: (B, 30, 7) -> (B, 30, 128) with embeddings
        3. Concatenate: (B, 38, 128)
        4. Cross-Attention Fusion with learnable queries: (B, 8, 128)
        5. Processing MLP: Flatten -> Dense -> ReLU -> (B, 256)
        6. Output Heads:
           - Policy: (B, 13)
           - Value: (B,)
    """
    
    emb_dim: int = EMB_DIM
    num_heads: int = NUM_HEADS
    ffn_dim: int = FFN_DIM
    num_layers: int = NUM_TRANSFORMER_LAYERS
    num_query_tokens: int = NUM_QUERY_TOKENS
    processing_dim: int = PROCESSING_DIM
    num_actions: int = NUM_ACTIONS
    
    @nn.compact
    def __call__(
        self,
        card_tensor: chex.Array,
        action_tensor: chex.Array,
        action_mask: Optional[chex.Array] = None,
    ) -> Tuple[chex.Array, chex.Array]:
        """
        Args:
            card_tensor: (B, 8, 2) int32 - [dealer_id, card_index]
            action_tensor: (B, 30, 7) float32 - action history features
            action_mask: (B, 30) boolean - True for valid actions, False for padding
            
        Returns:
            policy_logits: (B, 13) - raw logits for action distribution
            value: (B,) - estimated state value
        """
        batch_size = card_tensor.shape[0]
        
        # =================================================================
        # 1. Encode cards and actions
        # =================================================================
        
        card_repr, card_valid_mask = CardEncoder(
            emb_dim=self.emb_dim,
            num_heads=self.num_heads,
            ffn_dim=self.ffn_dim,
            num_layers=self.num_layers,
        )(card_tensor)  # (B, 8, 128), (B, 8)
        
        action_repr = ActionEncoder(
            emb_dim=self.emb_dim,
            num_heads=self.num_heads,
            ffn_dim=self.ffn_dim,
            num_layers=self.num_layers,
        )(action_tensor, action_mask)  # (B, 30, 128)
        
        # =================================================================
        # 2. Concatenate to form context
        # =================================================================
        
        context = jnp.concatenate([card_repr, action_repr], axis=1)  # (B, 38, 128)
        
        # Build context mask: use card valid mask + action mask
        if action_mask is not None:
            context_mask = jnp.concatenate([card_valid_mask, action_mask], axis=1)  # (B, 38)
        else:
            context_mask = jnp.concatenate([
                card_valid_mask,
                jnp.ones((batch_size, ACTION_SEQ_LEN), dtype=jnp.bool_)
            ], axis=1)
        
        # =================================================================
        # 3. Learnable query tokens + Cross-attention fusion
        # =================================================================
        
        # Initialize learnable query tokens (will be broadcast across batch)
        query_tokens = self.param(
            'query_tokens',
            nn.initializers.normal(stddev=0.02),
            (self.num_query_tokens, self.emb_dim),
        )
        
        # Broadcast to batch dimension: (Q, D) -> (B, Q, D)
        query_tokens = jnp.broadcast_to(
            query_tokens[None, :, :],
            (batch_size, self.num_query_tokens, self.emb_dim),
        )
        query_tokens = query_tokens.astype(jnp.bfloat16)
        
        # Cross-attention: queries attend to context
        fused = CrossAttentionBlock(
            emb_dim=self.emb_dim,
            num_heads=self.num_heads,
            ffn_dim=self.ffn_dim,
        )(query_tokens, context, context_mask)  # (B, 8, 128)
        
        # =================================================================
        # 4. Processing MLP
        # =================================================================
        
        # Flatten: (B, 8, 128) -> (B, 1024)
        flat = fused.reshape(batch_size, -1)
        
        # Dense -> ReLU: (B, 1024) -> (B, 256)
        processed = nn.Dense(
            self.processing_dim,
            dtype=jnp.bfloat16,
            param_dtype=jnp.float32,
        )(flat)
        processed = nn.relu(processed)
        
        # =================================================================
        # 5. Output heads
        # =================================================================
        
        # Policy head: (B, 256) -> (B, 13)
        policy_logits = nn.Dense(
            self.num_actions,
            dtype=jnp.bfloat16,
            param_dtype=jnp.float32,
        )(processed)
        
        # Value head: (B, 256) -> (B, 1) -> (B,)
        value = nn.Dense(
            1,
            dtype=jnp.bfloat16,
            param_dtype=jnp.float32,
        )(processed)
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
        action_tensor: (B, 30, 7) action history tensor
        
    Returns:
        mask: (B, 30) boolean mask where True = valid action
    """
    # Sum across features - if any feature is non-zero, the action is valid
    feature_sum = jnp.sum(jnp.abs(action_tensor), axis=-1)  # (B, 30)
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
        masked_logits: (B, 13) logits with illegal actions set to -inf
    """
    neg_inf = jnp.finfo(jnp.float32).min
    masked_logits = jnp.where(legal_mask, policy_logits, neg_inf)
    return masked_logits


def init_model(key: chex.PRNGKey) -> Tuple[TossHoldemModel, dict]:
    """Initialize the model and return parameters.
    
    Args:
        key: JAX random key for initialization
        
    Returns:
        model: TossHoldemModel instance
        params: Initialized parameters
    """
    model = TossHoldemModel()
    
    # Create dummy inputs for initialization
    dummy_cards = jnp.zeros((1, CARD_SEQ_LEN, CARD_INPUT_DIM), dtype=jnp.int32)
    dummy_actions = jnp.zeros((1, ACTION_SEQ_LEN, ACTION_INPUT_DIM), dtype=jnp.float32)
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


# =============================================================================
# Example Usage / Testing
# =============================================================================

if __name__ == "__main__":
    # Test model initialization and forward pass
    key = jax.random.PRNGKey(42)
    
    print("Initializing model...")
    model, params = init_model(key)
    
    num_params = count_parameters(params)
    print(f"Total parameters: {num_params:,}")
    
    # Test forward pass
    batch_size = 4
    
    # Create dummy inputs
    key, k1, k2 = jax.random.split(key, 3)
    cards = jax.random.randint(k1, (batch_size, CARD_SEQ_LEN, CARD_INPUT_DIM), -1, 52)
    actions = jax.random.normal(k2, (batch_size, ACTION_SEQ_LEN, ACTION_INPUT_DIM))
    action_mask = create_action_mask(actions)
    
    print(f"\nInput shapes:")
    print(f"  Cards: {cards.shape}")
    print(f"  Actions: {actions.shape}")
    print(f"  Action mask: {action_mask.shape}")
    
    # Forward pass
    policy_logits, value = model.apply(params, cards, actions, action_mask)
    
    print(f"\nOutput shapes:")
    print(f"  Policy logits: {policy_logits.shape}")
    print(f"  Value: {value.shape}")
    
    # Test legal action masking
    legal_mask = jax.random.bernoulli(key, 0.5, (batch_size, NUM_ACTIONS))
    masked_policy = apply_legal_action_mask(policy_logits, legal_mask)
    print(f"\nMasked policy logits: {masked_policy.shape}")
    
    # Test JIT compilation
    print("\nTesting JIT compilation...")
    
    @jax.jit
    def forward(params, cards, actions, action_mask):
        return model.apply(params, cards, actions, action_mask)
    
    # Warm-up JIT
    _ = forward(params, cards, actions, action_mask)
    
    # Timed run
    import time
    start = time.time()
    for _ in range(100):
        _ = forward(params, cards, actions, action_mask)
    jax.block_until_ready(_)
    elapsed = time.time() - start
    
    print(f"100 forward passes: {elapsed:.3f}s ({elapsed/100*1000:.2f}ms per pass)")
    print("\nModel test complete!")