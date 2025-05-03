import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """Layer normalization module with elementwise affine parameters."""
    
    def __init__(self, hidden_size, eps=1e-12):
        """Initialize LayerNorm.
        
        Args:
            hidden_size: Dimensionality of input features
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps
        
    def forward(self, x):
        """Apply layer normalization.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_size]
            
        Returns:
            Normalized tensor of the same shape
        """
        mean = x.mean(-1, keepdim=True)
        variance = x.var(-1, unbiased=False, keepdim=True)
        normalized = (x - mean) / torch.sqrt(variance + self.variance_epsilon)
        return self.weight * normalized + self.bias


def generate_positional_encoding(max_len, d_model):
    """Generate sinusoidal positional encoding.

    Args:
        max_len: Maximum sequence length
        d_model: Dimensionality of the model

    Returns:
        Tensor of shape [1, max_len, d_model]
    """
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
    )
    
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0)
    
    return pe


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention module."""
    
    def __init__(self, config):
        """Initialize multi-head attention.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # Ensure hidden_size is divisible by num_attention_heads
        assert config.hidden_size % config.num_attention_heads == 0
        
        # Linear layers for queries, keys, values, and output
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.output = nn.Linear(self.all_head_size, config.hidden_size)
        
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        
    def transpose_for_scores(self, x):
        """Reshape from [batch_size, seq_len, hidden_size] to [batch_size, num_heads, seq_len, head_dim]."""
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_shape)
        # Transpose to [batch_size, num_heads, seq_len, head_dim]
        return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_states, attention_mask=None):
        """Compute multi-head self-attention.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask: Optional mask of shape [batch_size, 1, 1, seq_len] or [batch_size, 1, seq_len, seq_len]
                            where 1 indicates tokens to attend to and 0 indicates tokens to ignore
                            
        Returns:
            Attention output of shape [batch_size, seq_len, hidden_size]
        """
        # Project inputs to queries, keys, and values
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        
        # Reshape for attention computation
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        
        # Take the dot product between "query" and "key" to get the raw attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Add the mask to the attention scores
            attention_scores = attention_scores + attention_mask
        
        # Normalize the attention scores to probabilities
        attention_probs = F.softmax(attention_scores, dim=-1)
        
        # Apply dropout to attention probabilities
        attention_probs = self.dropout(attention_probs)
        
        # Compute the context vector
        context_layer = torch.matmul(attention_probs, value_layer)
        
        # Reshape back to [batch_size, seq_len, hidden_size]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_shape)
        
        # Apply output projection
        output = self.output(context_layer)
        
        return output


class PositionwiseFeedForward(nn.Module):
    """Position-wise feed-forward network."""
    
    def __init__(self, config):
        """Initialize feed-forward network.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        
        self.dense1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.dense2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self, hidden_states):
        """Apply position-wise feed-forward network.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            
        Returns:
            Output tensor of shape [batch_size, seq_len, hidden_size]
        """
        hidden_states = self.dense1(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = self.dense2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        return hidden_states
        

class TransformerLayer(nn.Module):
    """Transformer decoder layer."""
    
    def __init__(self, config):
        """Initialize transformer layer.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        
        self.self_attention = MultiHeadAttention(config)
        self.norm1 = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        self.feed_forward = PositionwiseFeedForward(config)
        self.norm2 = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self, hidden_states, attention_mask=None):
        """Apply transformer layer.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask: Optional mask for self-attention
            
        Returns:
            Output tensor of shape [batch_size, seq_len, hidden_size]
        """
        # Self-attention with residual connection and layer normalization
        attention_output = self.self_attention(hidden_states, attention_mask)
        hidden_states = self.norm1(hidden_states + self.dropout(attention_output))
        
        # Feed-forward with residual connection and layer normalization
        ff_output = self.feed_forward(hidden_states)
        hidden_states = self.norm2(hidden_states + self.dropout(ff_output))
        
        return hidden_states


def create_causal_mask(seq_len, device):
    """Create a causal attention mask for autoregressive generation.
    
    Args:
        seq_len: Sequence length
        device: Device to create the mask on
        
    Returns:
        Causal mask of shape [1, 1, seq_len, seq_len]
    """
    # Create a mask that prevents attending to future tokens
    # 1 (True) is used to indicate positions to mask out
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    
    # Convert to attention mask format: 0 for positions to mask, -inf for positions to keep
    mask = torch.where(mask, -float('inf'), torch.zeros_like(mask, dtype=torch.float))
    
    # Add batch and head dimensions: [1, 1, seq_len, seq_len]
    mask = mask.unsqueeze(0).unsqueeze(0)
    
    return mask 