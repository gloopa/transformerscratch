import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.config import MiniTransformerConfig
from src.model.layers import TransformerLayer, LayerNorm, generate_positional_encoding, create_causal_mask


class MiniTransformer(nn.Module):
    """A small transformer-based language model."""
    
    def __init__(self, config):
        """Initialize the model.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Position embeddings
        self.register_buffer(
            "position_embeddings",
            generate_positional_encoding(config.max_position_embeddings, config.hidden_size)
        )
        
        # Dropout for embeddings
        self.embedding_dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(config) for _ in range(config.num_hidden_layers)
        ])
        
        # Final layer normalization
        self.layer_norm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Output projection (tied with input embeddings)
        self.output_projection = lambda x: F.linear(x, self.token_embeddings.weight)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()
            
    def get_input_embeddings(self):
        """Get the input embeddings layer."""
        return self.token_embeddings
    
    def set_input_embeddings(self, embeddings):
        """Set the input embeddings layer."""
        self.token_embeddings = embeddings
            
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        labels=None,
        use_cache=False,
        output_attentions=False,
    ):
        """Forward pass through the model.
        
        Args:
            input_ids: Input token IDs of shape [batch_size, seq_len]
            attention_mask: Optional attention mask of shape [batch_size, seq_len]
                          where 1 indicates tokens to attend to and 0 indicates tokens to ignore
            position_ids: Optional position IDs of shape [batch_size, seq_len]
                        if not provided, positions will be assigned sequentially
            labels: Optional target token IDs for language modeling of shape [batch_size, seq_len]
                  If provided, the model will compute the cross-entropy loss
            use_cache: Whether to return the last hidden state for future use
            output_attentions: Whether to return attention probabilities
            
        Returns:
            A dictionary with the following keys:
            - logits: Logits for next token prediction of shape [batch_size, seq_len, vocab_size]
            - loss: Language modeling loss (optional, if labels are provided)
            - last_hidden_state: Last hidden state (optional, if use_cache is True)
            - attentions: Attention probabilities (optional, if output_attentions is True)
        """
        input_shape = input_ids.size()
        batch_size, seq_length = input_shape
        device = input_ids.device
        
        # Create position IDs if not provided
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Create causal attention mask for autoregressive modeling
        # This ensures each token can only attend to previous tokens
        causal_mask = create_causal_mask(seq_length, device)
        
        # Create padding attention mask if attention_mask is provided
        if attention_mask is not None:
            # Convert 0s to -inf and 1s to 0s in attention_mask
            # This will be added to attention scores
            padding_mask = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e4
            attention_mask = causal_mask + padding_mask
        else:
            attention_mask = causal_mask
        
        # Get token embeddings
        embeddings = self.token_embeddings(input_ids)
        
        # Add position embeddings
        position_embeddings = self.position_embeddings[:, :seq_length, :]
        hidden_states = embeddings + position_embeddings
        
        # Apply embedding dropout
        hidden_states = self.embedding_dropout(hidden_states)
        
        # Initialize lists to store attention information if needed
        all_attentions = [] if output_attentions else None
        
        # Forward pass through each transformer layer
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
            if output_attentions:
                all_attentions.append(layer.self_attention.attention_probs)
        
        # Apply final layer normalization
        hidden_states = self.layer_norm(hidden_states)
        
        # Compute logits for next token prediction
        logits = self.output_projection(hidden_states)
        
        # Compute loss if labels are provided
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1)
            )
        
        # Prepare outputs
        outputs = {
            "logits": logits,
        }
        
        if loss is not None:
            outputs["loss"] = loss
            
        if use_cache:
            outputs["last_hidden_state"] = hidden_states
            
        if output_attentions:
            outputs["attentions"] = all_attentions
            
        return outputs
    
    def generate(
        self, 
        input_ids, 
        max_length=100, 
        temperature=1.0, 
        top_k=0, 
        top_p=0.9, 
        do_sample=True,
        repetition_penalty=1.0,
    ):
        """Generate text using the model.
        
        Args:
            input_ids: Input token IDs of shape [batch_size, seq_len]
            max_length: Maximum sequence length
            temperature: Temperature for sampling
            top_k: Number of highest probability tokens to keep for sampling
            top_p: Nucleus sampling probability threshold
            do_sample: Whether to sample or use greedy decoding
            repetition_penalty: Penalty for repetition
            
        Returns:
            Generated token IDs of shape [batch_size, max_length]
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Create an empty tensor for storing generated tokens
        generated = input_ids.clone()
        
        # Keep generating tokens until we reach max_length
        cur_length = input_ids.shape[1]
        while cur_length < max_length:
            # Forward pass to get logits for the current sequence
            with torch.no_grad():
                outputs = self.forward(generated)
                logits = outputs["logits"]
                
                # Get the logits for the next token
                next_token_logits = logits[:, -1, :].clone()
                
                # Apply temperature
                next_token_logits = next_token_logits / temperature
                
                # Apply repetition penalty
                if repetition_penalty != 1.0:
                    for i in range(batch_size):
                        for token_id in set(generated[i].tolist()):
                            next_token_logits[i, token_id] /= repetition_penalty
                
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(
                        next_token_logits, top_k, dim=-1
                    )[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float("inf")
                
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(
                        next_token_logits, descending=True, dim=-1
                    )
                    cumulative_probs = torch.cumsum(
                        F.softmax(sorted_logits, dim=-1), dim=-1
                    )
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    
                    # Shift the indices to the right to keep the first token above the threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    # Scatter sorted indices back to original logits
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        dim=1, index=sorted_indices, src=sorted_indices_to_remove
                    )
                    next_token_logits[indices_to_remove] = -float("inf")
                
                # Sample next token or use greedy decoding
                if do_sample:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Append next token to the sequence
                generated = torch.cat((generated, next_token), dim=1)
                cur_length += 1
                
                # Stop if we generate an EOS token
                if (next_token == self.config.eos_token_id).any():
                    break
        
        return generated
        
    def param_count(self):
        """Count the number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def __str__(self):
        """String representation of the model."""
        param_count = self.param_count()
        param_count_m = param_count / 1000000
        
        return (
            f"MiniTransformer Model:\n"
            f"  Config: {self.config.__str__()}\n"
            f"  Parameter Count: {param_count_m:.2f}M\n"
        )


def create_model(config=None):
    """Create a MiniTransformer model.
    
    Args:
        config: Model configuration, if None, the default configuration will be used
        
    Returns:
        MiniTransformer model
    """
    if config is None:
        config = MiniTransformerConfig()
        
    model = MiniTransformer(config)
    
    # Verify parameter count
    actual_params = model.param_count()
    expected_params = config.get_params_count()
    
    print(f"Created model with {actual_params / 1e6:.2f}M parameters (expected: {expected_params / 1e6:.2f}M)")
    
    return model 