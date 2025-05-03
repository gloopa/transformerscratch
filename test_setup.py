#!/usr/bin/env python3
import os
import sys
import torch

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from src.model.config import MiniTransformerConfig
from src.model.transformer import create_model

def test_model_creation():
    """Test if we can create a model."""
    print("Testing model creation...")
    
    # Create model configuration
    config = MiniTransformerConfig(
        vocab_size=50257,
        max_position_embeddings=1024,
        hidden_size=384,
        num_hidden_layers=6,
        num_attention_heads=6,
        intermediate_size=1536,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    
    # Create model
    model = create_model(config)
    
    # Print model info
    param_count = model.param_count()
    print(f"Model parameter count: {param_count / 1e6:.2f}M")
    
    # Test a forward pass
    device = torch.device("cpu")
    model.to(device)
    
    # Create dummy input
    batch_size = 2
    seq_length = 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length), device=device)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids)
    
    print(f"Output shape: {outputs.logits.shape}")
    print("Forward pass successful!")
    
    return True

if __name__ == "__main__":
    success = test_model_creation()
    if success:
        print("All tests passed!")
    else:
        print("Tests failed!") 