#!/usr/bin/env python3
import os
import sys
import torch

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from src.model.config import MiniTransformerConfig
from src.model.transformer import create_model
from src.data.dataloader import create_dataloaders
from tokenizers import Tokenizer

def test_data_loading_and_model():
    """Test if we can load the data and create a model."""
    print("Testing data loading and model creation...")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizer directly using the Hugging Face tokenizers library
    tokenizer_path = os.path.join("data/wikitext/tokenizer")
    try:
        tokenizer_file = os.path.join(tokenizer_path, "tokenizer.json")
        if os.path.exists(tokenizer_file):
            tokenizer = Tokenizer.from_file(tokenizer_file)
            vocab_size = tokenizer.get_vocab_size()
            print(f"Loaded tokenizer with vocabulary size: {vocab_size}")
        else:
            # Fallback for vocab_size
            vocab_size = 50257  # Default GPT-2 vocab size
            print(f"Using default vocabulary size: {vocab_size}")
    except Exception as e:
        print(f"Failed to load tokenizer: {e}")
        vocab_size = 50257  # Default GPT-2 vocab size
        print(f"Using default vocabulary size: {vocab_size}")
    
    # Create model configuration
    config = MiniTransformerConfig(
        vocab_size=vocab_size,
        max_position_embeddings=512,  # Smaller for testing
        hidden_size=128,  # Smaller for testing
        num_hidden_layers=2,  # Smaller for testing
        num_attention_heads=4,  # Smaller for testing
        intermediate_size=512,  # Smaller for testing
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    
    # Create model
    model = create_model(config)
    model.to(device)
    
    # Print model info
    param_count = model.param_count()
    print(f"Model parameter count: {param_count / 1e6:.2f}M")
    
    # Create dataloaders
    try:
        dataloaders = create_dataloaders(
            data_path="data/wikitext",
            batch_size=4,  # Small batch size for testing
            context_length=512,  # Smaller context length for testing
            stride=256,
            num_workers=0,  # No multiprocessing for testing
            pad_token_id=config.pad_token_id,
        )
        
        train_dataloader = dataloaders["train"]
        val_dataloader = dataloaders["validation"]
        
        print(f"Created dataloaders with {len(train_dataloader)} training batches")
        print(f"and {len(val_dataloader)} validation batches")
        
        # Test a single batch
        train_batch = next(iter(train_dataloader))
        input_ids = train_batch["input_ids"].to(device)
        attention_mask = train_batch["attention_mask"].to(device)
        labels = train_batch["labels"].to(device)
        
        print(f"Input shape: {input_ids.shape}")
        print(f"Attention mask shape: {attention_mask.shape}")
        print(f"Labels shape: {labels.shape}")
        
        # Forward pass
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
        
        # Check if outputs is a dict with 'logits' or just the tensor directly
        if isinstance(outputs, dict) and 'logits' in outputs:
            output_tensor = outputs['logits']
        else:
            output_tensor = outputs  # Assume it's the tensor directly
            
        print(f"Output shape: {output_tensor.shape}")
        print("Forward pass successful!")
        
        return True
    except Exception as e:
        print(f"Failed to load data or run model: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_data_loading_and_model()
    if success:
        print("All tests passed!")
    else:
        print("Tests failed!") 