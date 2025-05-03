#!/usr/bin/env python3
import os
import argparse
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer

def download_and_prepare_wikitext(dataset_name="wikitext-2-raw-v1", output_dir="data"):
    """Download and prepare WikiText dataset.
    
    Args:
        dataset_name: Name of the wikitext dataset to download (wikitext-2-raw-v1 or wikitext-103-raw-v1)
        output_dir: Output directory for the processed data
    """
    print(f"Downloading {dataset_name}...")
    
    # Create output directory
    dataset_dir = os.path.join(output_dir, dataset_name.split("-")[0])
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Download the dataset
    dataset = load_dataset(dataset_name)
    
    # Save raw text files for training a tokenizer
    print("Saving raw text files...")
    for split, data in dataset.items():
        filename = os.path.join(dataset_dir, f"{split}_raw.txt")
        with open(filename, "w", encoding="utf-8") as f:
            f.write("\n".join(data["text"]))
        print(f"  Saved {filename}")
    
    # Train tokenizer on the training data
    print("Training tokenizer...")
    train_file = os.path.join(dataset_dir, "train_raw.txt")
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=[train_file],
        vocab_size=50257,  # Same as GPT-2
        min_frequency=2,
        special_tokens=["<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>"]
    )
    
    # Save the tokenizer
    tokenizer_dir = os.path.join(dataset_dir, "tokenizer")
    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer.save_model(tokenizer_dir)
    print(f"Tokenizer saved to {tokenizer_dir}")
    
    # Tokenize and save each split
    print("Tokenizing data...")
    for split in dataset.keys():
        input_file = os.path.join(dataset_dir, f"{split}_raw.txt")
        output_file = os.path.join(dataset_dir, f"{split}.txt")
        
        with open(input_file, "r", encoding="utf-8") as f_in:
            with open(output_file, "w", encoding="utf-8") as f_out:
                for line in f_in:
                    if line.strip():  # Skip empty lines
                        encoded = tokenizer.encode(line.strip())
                        f_out.write(" ".join(map(str, encoded.ids)) + "\n")
        
        print(f"  Tokenized {split} split saved to {output_file}")
    
    print("Data preparation complete!")
    
    # Print some statistics
    with open(os.path.join(dataset_dir, "train.txt"), "r") as f:
        train_tokens = sum(len(line.split()) for line in f)
    with open(os.path.join(dataset_dir, "validation.txt"), "r") as f:
        val_tokens = sum(len(line.split()) for line in f)
    with open(os.path.join(dataset_dir, "test.txt"), "r") as f:
        test_tokens = sum(len(line.split()) for line in f)
    
    print(f"Dataset statistics:")
    print(f"  Train: {train_tokens} tokens")
    print(f"  Validation: {val_tokens} tokens")
    print(f"  Test: {test_tokens} tokens")
    print(f"  Vocabulary size: {tokenizer.get_vocab_size()}")

def main():
    parser = argparse.ArgumentParser(description="Download and prepare WikiText dataset")
    parser.add_argument("--dataset", type=str, default="wikitext-2-raw-v1", 
                        help="Dataset name (wikitext-2-raw-v1 or wikitext-103-raw-v1)")
    parser.add_argument("--output_dir", type=str, default="data", 
                        help="Output directory")
    args = parser.parse_args()
    
    download_and_prepare_wikitext(args.dataset, args.output_dir)

if __name__ == "__main__":
    main() 