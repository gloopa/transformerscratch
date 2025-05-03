#!/usr/bin/env python3
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from src.model.config import MiniTransformerConfig
from src.model.transformer import create_model

class DummyDataset(Dataset):
    """Simple dummy dataset for testing."""
    
    def __init__(self, vocab_size=50257, seq_length=32, size=100):
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.size = size
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Create random input and target
        input_ids = torch.randint(0, self.vocab_size, (self.seq_length,))
        # Just shift input by 1 for target
        target_ids = torch.cat([input_ids[1:], torch.randint(0, self.vocab_size, (1,))])
        
        return {
            "input_ids": input_ids,
            "labels": target_ids
        }

def train_one_epoch(model, dataloader, optimizer, device):
    """Train the model for one epoch.
    
    Args:
        model: Model to train
        dataloader: DataLoader for training data
        optimizer: Optimizer
        device: Device to use
        
    Returns:
        Average loss
    """
    model.train()
    total_loss = 0
    
    criterion = nn.CrossEntropyLoss()
    
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(input_ids)
        
        # Get logits
        if isinstance(outputs, dict) and 'logits' in outputs:
            logits = outputs['logits']
        else:
            logits = outputs
            
        # Reshape for loss calculation
        logits = logits.view(-1, logits.size(-1))
        labels = labels.view(-1)
        
        # Calculate loss
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Update weights
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def test_training():
    """Test if we can train the model."""
    print("Testing model training...")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model configuration
    config = MiniTransformerConfig(
        vocab_size=50257,
        max_position_embeddings=1024,
        hidden_size=128,  # Smaller for faster testing
        num_hidden_layers=2,  # Smaller for faster testing
        num_attention_heads=4,  # Smaller for faster testing
        intermediate_size=512,  # Smaller for faster testing
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
    
    # Create dummy dataset and dataloader
    dataset = DummyDataset(vocab_size=config.vocab_size, seq_length=32, size=100)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=5e-5)
    
    # Train for a few iterations
    num_epochs = 2
    for epoch in range(num_epochs):
        loss = train_one_epoch(model, dataloader, optimizer, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")
    
    print("Training completed successfully!")
    return True

if __name__ == "__main__":
    success = test_training()
    if success:
        print("All tests passed!")
    else:
        print("Tests failed!") 