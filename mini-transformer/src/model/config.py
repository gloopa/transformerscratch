class MiniTransformerConfig:
    """Configuration class for the MiniTransformer model."""
    
    def __init__(
        self,
        vocab_size=50257,  # GPT-2 vocabulary size, will be adjusted based on tokenizer
        max_position_embeddings=1024,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,  # 4x hidden_size is standard
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        layer_norm_eps=1e-12,
        initializer_range=0.02,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    ):
        """Initialize the configuration.
        
        Args:
            vocab_size: Size of the vocabulary
            max_position_embeddings: Maximum sequence length
            hidden_size: Size of the hidden layer embeddings
            num_hidden_layers: Number of transformer layers
            num_attention_heads: Number of attention heads
            intermediate_size: Size of the feedforward layer
            hidden_dropout_prob: Dropout probability for hidden layers
            attention_probs_dropout_prob: Dropout probability for attention
            layer_norm_eps: Epsilon for layer normalization
            initializer_range: Standard deviation for initializing weights
            pad_token_id: ID of the padding token
            bos_token_id: ID of the beginning of sequence token
            eos_token_id: ID of the end of sequence token
        """
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        
        # Calculated properties
        self.head_dim = hidden_size // num_attention_heads
        assert self.head_dim * num_attention_heads == hidden_size, "hidden_size must be divisible by num_attention_heads"
    
    def get_params_count(self):
        """Calculate approximate parameter count based on config."""
        # Token embeddings
        embedding_params = self.vocab_size * self.hidden_size
        # Position embeddings
        position_params = self.max_position_embeddings * self.hidden_size
        
        # Each transformer layer
        layer_params = 0
        # Self-attention
        layer_params += 4 * self.hidden_size * self.hidden_size  # Q, K, V, and output projections
        # Layer norms (2 per layer)
        layer_params += 4 * self.hidden_size  # 2 gammas and 2 betas
        # FFN
        layer_params += 2 * self.hidden_size * self.intermediate_size + self.hidden_size + self.intermediate_size  # weights and biases
        
        total_params = embedding_params + position_params + (layer_params * self.num_hidden_layers)
        return total_params
    
    @classmethod
    def from_dict(cls, config_dict):
        """Initialize from a dictionary of parameters."""
        return cls(**config_dict)
    
    def to_dict(self):
        """Convert configuration to dictionary."""
        return {
            "vocab_size": self.vocab_size,
            "max_position_embeddings": self.max_position_embeddings,
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "intermediate_size": self.intermediate_size,
            "hidden_dropout_prob": self.hidden_dropout_prob,
            "attention_probs_dropout_prob": self.attention_probs_dropout_prob,
            "layer_norm_eps": self.layer_norm_eps,
            "initializer_range": self.initializer_range,
            "pad_token_id": self.pad_token_id,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
        }
    
    def __str__(self):
        """String representation with parameter count."""
        param_count = self.get_params_count()
        param_count_m = param_count / 1000000
        
        return (
            f"MiniTransformer Configuration:\n"
            f"  Vocabulary Size: {self.vocab_size}\n"
            f"  Max Sequence Length: {self.max_position_embeddings}\n"
            f"  Hidden Size: {self.hidden_size}\n"
            f"  Layers: {self.num_hidden_layers}\n"
            f"  Attention Heads: {self.num_attention_heads}\n"
            f"  Head Dimension: {self.head_dim}\n"
            f"  Intermediate Size: {self.intermediate_size}\n"
            f"  Parameters: ~{param_count_m:.2f}M\n"
        )


# Predefined model sizes
MINI_CONFIG = MiniTransformerConfig(
    vocab_size=50257,
    max_position_embeddings=1024,
    hidden_size=384,
    num_hidden_layers=6,
    num_attention_heads=6,
    intermediate_size=1536,
)  # ~11M parameters

BASE_CONFIG = MiniTransformerConfig(
    vocab_size=50257,
    max_position_embeddings=1024,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
)  # ~110M parameters 