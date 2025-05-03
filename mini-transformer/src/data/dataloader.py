import os
import torch
import numpy as np
from tokenizers import Tokenizer


class WikiTextDataset(torch.utils.data.Dataset):
    """WikiText dataset for language modeling."""
    
    def __init__(self, data_path, split="train", context_length=1024, stride=512):
        """Initialize the dataset.
        
        Args:
            data_path: Path to the tokenized data directory
            split: Split to use (train, validation, or test)
            context_length: Maximum context length
            stride: Stride for sliding window
        """
        self.data_path = data_path
        self.split = split
        self.context_length = context_length
        self.stride = stride
        
        # Load the preprocessed token IDs
        self.token_ids = self._load_data()
        
        # Create examples with sliding window
        self.examples = self._create_examples()
        
        print(f"Loaded {split} split with {len(self.examples)} examples")
        
    def _load_data(self):
        """Load the tokenized data."""
        filename = os.path.join(self.data_path, f"{self.split}.txt")
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File not found: {filename}")
        
        token_ids = []
        with open(filename, "r") as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    ids = list(map(int, line.strip().split()))
                    token_ids.extend(ids)
        
        return token_ids
    
    def _create_examples(self):
        """Create examples with sliding window."""
        examples = []
        for i in range(0, len(self.token_ids) - self.context_length, self.stride):
            examples.append(self.token_ids[i:i + self.context_length])
        
        # Handle last example if needed
        if i + self.context_length < len(self.token_ids):
            examples.append(self.token_ids[-self.context_length:])
        
        return examples
    
    def __len__(self):
        """Return the number of examples."""
        return len(self.examples)
    
    def __getitem__(self, idx):
        """Return an example."""
        tokens = self.examples[idx]
        return {
            "input_ids": torch.tensor(tokens, dtype=torch.long),
        }


class WikiTextCollator:
    """Collator for WikiText dataset."""
    
    def __init__(self, pad_token_id=0):
        """Initialize the collator.
        
        Args:
            pad_token_id: Token ID to use for padding
        """
        self.pad_token_id = pad_token_id
    
    def __call__(self, examples):
        """Collate examples.
        
        Args:
            examples: List of examples from dataset
            
        Returns:
            Collated batch
        """
        # All examples should have the same length, but just in case
        max_length = max(len(example["input_ids"]) for example in examples)
        
        input_ids = []
        attention_mask = []
        labels = []
        
        for example in examples:
            # Convert to tensor and pad if needed
            tokens = example["input_ids"]
            padding_length = max_length - len(tokens)
            
            # Create attention mask (1 for tokens, 0 for padding)
            mask = [1] * len(tokens) + [0] * padding_length
            
            # Pad the tokens if needed
            if padding_length > 0:
                tokens = torch.cat([
                    tokens, 
                    torch.ones(padding_length, dtype=torch.long) * self.pad_token_id
                ])
            
            input_ids.append(tokens)
            attention_mask.append(torch.tensor(mask, dtype=torch.long))
            labels.append(tokens.clone())  # For language modeling, labels are the input shifted by 1
        
        # Stack tensors
        batch = {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
            "labels": torch.stack(labels),
        }
        
        return batch


def load_tokenizer(tokenizer_path):
    """Load the tokenizer.
    
    Args:
        tokenizer_path: Path to the tokenizer directory
        
    Returns:
        Tokenizer
    """
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")
    
    vocab_file = os.path.join(tokenizer_path, "vocab.json")
    merges_file = os.path.join(tokenizer_path, "merges.txt")
    
    if not os.path.exists(vocab_file) or not os.path.exists(merges_file):
        raise FileNotFoundError(f"Tokenizer files not found in {tokenizer_path}")
    
    tokenizer = Tokenizer.from_file(os.path.join(tokenizer_path, "tokenizer.json"))
    
    return tokenizer


def create_dataloaders(
    data_path, 
    batch_size=16, 
    context_length=1024, 
    stride=512, 
    num_workers=4,
    pad_token_id=0
):
    """Create dataloaders for training and evaluation.
    
    Args:
        data_path: Path to the data directory
        batch_size: Batch size
        context_length: Maximum context length
        stride: Stride for sliding window
        num_workers: Number of workers for dataloaders
        pad_token_id: Padding token ID
        
    Returns:
        Dictionary with train, validation, and test dataloaders
    """
    # Create datasets
    train_dataset = WikiTextDataset(
        data_path=data_path, 
        split="train", 
        context_length=context_length, 
        stride=stride
    )
    
    val_dataset = WikiTextDataset(
        data_path=data_path, 
        split="validation", 
        context_length=context_length, 
        stride=context_length  # Non-overlapping for validation
    )
    
    test_dataset = WikiTextDataset(
        data_path=data_path, 
        split="test", 
        context_length=context_length, 
        stride=context_length  # Non-overlapping for test
    )
    
    # Create collator
    collator = WikiTextCollator(pad_token_id=pad_token_id)
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return {
        "train": train_loader,
        "validation": val_loader,
        "test": test_loader,
    } 