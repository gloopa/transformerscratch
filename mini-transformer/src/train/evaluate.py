#!/usr/bin/env python3
import os
import sys
import argparse
import torch
import numpy as np

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.model.config import MiniTransformerConfig
from src.model.transformer import create_model
from src.data.dataloader import create_dataloaders
from src.data.tokenizer import load_tokenizer
from src.train.utils import evaluate, compute_perplexity, load_checkpoint


def evaluate_model(args):
    """Evaluate the model.
    
    Args:
        args: Command line arguments
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer_path = os.path.join(args.data_dir, "tokenizer")
    tokenizer = load_tokenizer(tokenizer_path)
    
    # Load model checkpoint
    print(f"Loading model from {args.model_path}")
    
    # Create a temporary model to get the configuration
    temp_model = create_model()
    checkpoint_data = load_checkpoint(temp_model, path=args.model_path)
    config_dict = checkpoint_data["config"]
    
    # Create model with the loaded configuration
    if config_dict is not None:
        config = MiniTransformerConfig.from_dict(config_dict)
    else:
        config = temp_model.config
        
    # Create the model
    model = create_model(config)
    model.to(device)
    
    # Load the weights
    load_checkpoint(model, path=args.model_path)
    
    # Print model info
    print(f"Model parameters: {model.param_count() / 1e6:.2f}M")
    
    # Create dataloaders
    dataloaders = create_dataloaders(
        data_path=args.data_dir,
        batch_size=args.batch_size,
        context_length=config.max_position_embeddings,
        num_workers=args.num_workers,
        pad_token_id=config.pad_token_id,
    )
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_results = evaluate(model, dataloaders["test"])
    test_loss = test_results["loss"]
    test_perplexity = test_results["perplexity"]
    
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test perplexity: {test_perplexity:.2f}")
    
    # Generate some text
    if args.generate:
        print("\nGenerating text...")
        
        # Tokenize the prompt
        if args.prompt:
            prompt_tokens = tokenizer.encode(args.prompt)
            prompt = torch.tensor([prompt_tokens], dtype=torch.long).to(device)
        else:
            # Use random tokens as prompt
            prompt = torch.randint(0, config.vocab_size, (1, 10), dtype=torch.long).to(device)
        
        # Generate text
        generated = model.generate(
            input_ids=prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            do_sample=args.do_sample,
            repetition_penalty=args.repetition_penalty,
        )
        
        # Decode generated tokens
        generated_text = tokenizer.decode(generated[0].tolist())
        
        print("Generated text:")
        print("-" * 50)
        print(generated_text)
        print("-" * 50)
        
        # Save generated text to file
        if args.output_file:
            with open(args.output_file, "w", encoding="utf-8") as f:
                f.write(generated_text)
            print(f"Generated text saved to {args.output_file}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained MiniTransformer model")
    
    # Model configuration
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--data_dir", type=str, default="data/wikitext-2", help="Data directory")
    
    # Evaluation configuration
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA")
    
    # Generation configuration
    parser.add_argument("--generate", action="store_true", help="Generate text")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt for generation")
    parser.add_argument("--max_length", type=int, default=100, help="Maximum length for generation")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p (nucleus) sampling")
    parser.add_argument("--do_sample", action="store_true", help="Use sampling (otherwise greedy)")
    parser.add_argument("--repetition_penalty", type=float, default=1.2, help="Repetition penalty")
    parser.add_argument("--output_file", type=str, default=None, help="Output file for generated text")
    
    args = parser.parse_args()
    
    evaluate_model(args)


if __name__ == "__main__":
    main() 