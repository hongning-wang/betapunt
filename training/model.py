"""Transformer-based AlphaHoldem model for Toss or Hold'em.

Architecture:
- Card Encoder: (B, 8, 2) -> Embeddings -> Transformer -> (B, 8, 128)
- Action Encoder: (B, 30, 18) -> Embeddings + Dense -> Transformer -> (B, 30, 128)
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
CARD_SEQ_LEN = 8
ACTION_SEQ_LEN = 30
CARD_INPUT_DIM = 2      # [dealer_id, card_index]
ACTION_INPUT_DIM = 18   # [player_id, legal_mask(13), action_taken, pot, round_idx, pos_in_round]

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
    emb_dim: int
    num_heads: int
    ffn_dim: int
    
    @nn.compact
    def __call__(
        self,
        x: chex.Array,
        mask: Optional[chex.Array] = None,
    ) -> chex.Array:
        assert self.emb_dim % self.num_heads == 0, "emb_dim must be divisible by num_heads"
        head_dim = self.emb_dim // self.num_heads
        
        # Self-attention block
        residual = x
        x = nn.LayerNorm(dtype=jnp.bfloat16, param_dtype=jnp.float32)(x)
        x = MultiHeadAttention(num_heads=self.num_heads, head_dim=head_dim)(x, x, x, mask)
        x = residual + x
        
        # FFN block
        residual = x
        x = nn.LayerNorm(dtype=jnp.bfloat16, param_dtype=jnp.float32)(x)
        x = nn.Dense(self.ffn_dim, dtype=jnp.bfloat16, param_dtype=jnp.float32)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.emb_dim, dtype=jnp.bfloat16, param_dtype=jnp.float32)(x)
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
        head_dim = self.emb_dim // self.num_heads
        
        # Cross-attention
        residual = query_tokens
        x = nn.LayerNorm(dtype=jnp.bfloat16, param_dtype=jnp.float32)(query_tokens)
        x = MultiHeadAttention(self.num_heads, head_dim)(x, context, context, context_mask)
        x = residual + x
        
        # FFN
        residual = x
        x = nn.LayerNorm(dtype=jnp.bfloat16, param_dtype=jnp.float32)(x)
        x = nn.Dense(self.ffn_dim, dtype=jnp.bfloat16, param_dtype=jnp.float32)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.emb_dim, dtype=jnp.bfloat16, param_dtype=jnp.float32)(x)
        x = residual + x
        
        return x


# =============================================================================
# Encoder Modules
# =============================================================================

class CardEncoder(nn.Module):
    """Encodes card tensor using embeddings and transformer.
    
    Input: (B, 8, 2) where each entry is [dealer_id, card_index]
        - dealer_id: 0 = non-dealer, 1 = dealer, 2 = community, -1 = invalid
        - card_index: 0-51 for cards, -1 for invalid/undealt
    
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
            card_tensor: (B, 8, 2) int32 tensor with [dealer_id, card_index]
            
        Returns:
            card_representation: (B, 8, emb_dim)
            valid_mask: (B, 8) boolean mask where True = valid card
        """
        seq_len = card_tensor.shape[1]  # 8
        
        # === Card Index Embedding ===
        # 0-51 valid cards, -1 invalid -> mapped to 52
        card_indices = card_tensor[..., 1]
        
        card_emb = nn.Embed(
            num_embeddings=53,  # 52 cards + 1 invalid
            features=self.emb_dim,
            dtype=jnp.bfloat16,
            param_dtype=jnp.float32,
        )(card_indices)  # (B, 8, emb_dim)
        
        # === Dealer ID Embedding ===
        # 0 = non-dealer, 1 = dealer, 2 = community, -1 invalid -> mapped to 3
        dealer_ids = card_tensor[..., 0]
        dealer_ids_safe = jnp.where(dealer_ids >= 0, dealer_ids, 3)
        
        dealer_emb = nn.Embed(
            num_embeddings=4,  # 0, 1, 2, 3(invalid)
            features=self.emb_dim,
            dtype=jnp.bfloat16,
            param_dtype=jnp.float32,
        )(dealer_ids_safe)  # (B, 8, emb_dim)
        
        # === Position Embedding ===
        # Encodes which slot (hole0, hole1, flop0, flop1, discard_oop, discard_dlr, turn, river)
        positions = jnp.arange(seq_len)
        pos_emb = nn.Embed(
            num_embeddings=8,
            features=self.emb_dim,
            dtype=jnp.bfloat16,
            param_dtype=jnp.float32,
        )(positions)  # (8, emb_dim)
        
        # === Combine: additive ===
        combined = card_emb + dealer_emb + pos_emb[None, :, :]
        
        # === Mask: True for valid cards (card_index >= 0) ===
        valid_mask = card_tensor[..., 1] >= 0  # (B, 8)
        
        # === Transformer ===
        encoded = TransformerEncoder(
            emb_dim=self.emb_dim,
            num_heads=self.num_heads,
            ffn_dim=self.ffn_dim,
            num_layers=self.num_layers,
        )(combined, mask=valid_mask)
        
        return encoded, valid_mask

!!! Need to merge
class ActionEncoder(nn.Module):
    """Encodes action history tensor using embeddings and transformer.
    
    Input: (B, 30, 5) where each action entry contains:
        [0]: player_id (0 or 1) - categorical
        [1]: action taken (0-12) - categorical
        [2]: pot size (normalized) - continuous
        [3]: round index (0-4) - categorical
        [4]: position in round (0-5) - categorical
    
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
            action_tensor: (B, 30, 5) float32 tensor
            action_mask: (B, 30) boolean mask where True = valid action
            
        Returns:
            action_representation: (B, 30, emb_dim)
        """
        # === Extract components ===
        player_id = action_tensor[..., 0].astype(jnp.int32)
        action_taken = action_tensor[..., 1].astype(jnp.int32)
        pot = action_tensor[..., 2]
        round_idx = action_tensor[..., 3].astype(jnp.int32)
        pos_in_round = action_tensor[..., 4].astype(jnp.int32)
        
        # === Categorical Embeddings ===
        player_emb = nn.Embed(
            num_embeddings=2,
            features=self.emb_dim,
            dtype=jnp.bfloat16,
            param_dtype=jnp.float32,
        )(player_id)  # (B, 30, emb_dim)
        
        action_emb = nn.Embed(
            num_embeddings=13,
            features=self.emb_dim,
            dtype=jnp.bfloat16,
            param_dtype=jnp.float32,
        )(action_taken)  # (B, 30, emb_dim)
        
        round_emb = nn.Embed(
            num_embeddings=5,
            features=self.emb_dim,
            dtype=jnp.bfloat16,
            param_dtype=jnp.float32,
        )(round_idx)  # (B, 30, emb_dim)
        
        pos_emb = nn.Embed(
            num_embeddings=6,
            features=self.emb_dim,
            dtype=jnp.bfloat16,
            param_dtype=jnp.float32,
        )(pos_in_round)  # (B, 30, emb_dim)
        
        # === Continuous Features Projection ===
        continuous_emb = nn.Dense(
            self.emb_dim,
            dtype=jnp.bfloat16,
            param_dtype=jnp.float32,
        )(pot.astype(jnp.float32))  # (B, 30, emb_dim)
        
        # === Combine: additive ===
        combined = player_emb + action_emb + round_emb + pos_emb + continuous_emb
        
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
        2. Action Encoder: (B, 30, 18) -> (B, 30, 128) with embeddings
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
            action_tensor: (B, 30, 18) float32 - action history features
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
        action_tensor: (B, 30, 18) action history tensor
        
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