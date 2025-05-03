#!/usr/bin/env python3
import os
import sys
import argparse
import torch
import tqdm
import random
import numpy as np

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.model.config import MiniTransformerConfig, MINI_CONFIG
from src.model.transformer import create_model
from src.data.dataloader import create_dataloaders
from src.data.tokenizer import load_tokenizer
from src.train.scheduler import get_scheduler
from src.train.utils import (
    get_optimizer,
    train_step,
    evaluate,
    save_checkpoint,
    load_checkpoint,
    plot_training_curves,
    get_amp_scaler,
    Timer,
    MovingAverage,
)


def set_seed(seed):
    """Set random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(args):
    """Train the model.
    
    Args:
        args: Command line arguments
    """
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.plots_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizer and update vocabulary size
    tokenizer_path = os.path.join(args.data_dir, "tokenizer")
    if os.path.exists(tokenizer_path):
        tokenizer = load_tokenizer(tokenizer_path)
        vocab_size = tokenizer.get_vocab_size()
        print(f"Loaded tokenizer with vocabulary size: {vocab_size}")
    else:
        vocab_size = args.vocab_size
        print(f"No tokenizer found, using default vocabulary size: {vocab_size}")
    
    # Create model configuration
    config = MiniTransformerConfig(
        vocab_size=vocab_size,
        max_position_embeddings=args.context_length,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        intermediate_size=args.intermediate_size,
        hidden_dropout_prob=args.dropout,
        attention_probs_dropout_prob=args.dropout,
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
    dataloaders = create_dataloaders(
        data_path=args.data_dir,
        batch_size=args.batch_size,
        context_length=args.context_length,
        stride=args.stride,
        num_workers=args.num_workers,
        pad_token_id=config.pad_token_id,
    )
    
    train_dataloader = dataloaders["train"]
    val_dataloader = dataloaders["validation"]
    
    # Compute number of training steps
    steps_per_epoch = len(train_dataloader)
    total_steps = steps_per_epoch * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    # Create optimizer
    optimizer = get_optimizer(
        model,
        name=args.optimizer,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    
    # Create learning rate scheduler
    scheduler = get_scheduler(
        name=args.scheduler,
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        max_steps=total_steps,
        min_lr_ratio=args.min_lr_ratio,
    )
    
    # Enable mixed precision training if requested
    use_amp = args.mixed_precision and torch.cuda.is_available()
    scaler = get_amp_scaler(enabled=use_amp)
    
    if use_amp:
        print("Using automatic mixed precision training")
    
    # Load checkpoint if provided
    if args.resume_from:
        checkpoint_path = args.resume_from
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint_data = load_checkpoint(model, optimizer, scheduler, checkpoint_path)
        start_epoch = checkpoint_data["epoch"]
        start_step = checkpoint_data["step"]
        best_val_loss = checkpoint_data["loss"]
    else:
        start_epoch = 0
        start_step = 0
        best_val_loss = float("inf")
    
    # Initialize training metrics
    timer = Timer()
    timer.start()
    
    loss_avg = MovingAverage(window_size=100)
    grad_norm_avg = MovingAverage(window_size=100)
    
    train_losses = []
    val_losses = []
    train_perplexities = []
    val_perplexities = []
    
    # Training loop
    print(f"Starting training for {args.num_epochs} epochs")
    
    for epoch in range(start_epoch, args.num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_steps = 0
        
        progress_bar = tqdm.tqdm(
            enumerate(train_dataloader),
            total=len(train_dataloader),
            desc=f"Epoch {epoch+1}/{args.num_epochs}",
            disable=args.no_progress_bar,
        )
        
        for step, batch in progress_bar:
            global_step = epoch * steps_per_epoch + step + 1
            
            # Skip steps if resuming from checkpoint
            if global_step < start_step:
                continue
            
            # Perform training step
            step_results = train_step(
                model=model,
                batch=batch,
                optimizer=optimizer,
                scheduler=scheduler,
                max_grad_norm=args.max_grad_norm,
                use_amp=use_amp,
                scaler=scaler,
            )
            
            # Update metrics
            loss = step_results["loss"]
            perplexity = step_results["perplexity"]
            grad_norm = step_results["grad_norm"]
            lr = step_results["lr"]
            
            loss_avg.add(loss)
            if grad_norm is not None:
                grad_norm_avg.add(grad_norm)
            
            train_loss += loss
            train_steps += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{loss_avg.get():.4f}",
                "ppl": f"{perplexity:.2f}",
                "lr": f"{lr:.8f}",
                "grad": f"{grad_norm_avg.get():.2f}" if grad_norm is not None else "N/A",
            })
            
            # Save checkpoint periodically
            if global_step % args.save_steps == 0:
                checkpoint_path = os.path.join(args.checkpoint_dir, f"checkpoint-{global_step}.pt")
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step=global_step,
                    epoch=epoch,
                    loss=loss_avg.get(),
                    config=config,
                    path=checkpoint_path,
                )
            
            # Evaluate periodically
            if global_step % args.eval_steps == 0:
                # Evaluate on validation set
                val_results = evaluate(model, val_dataloader, max_steps=args.eval_max_steps)
                val_loss = val_results["loss"]
                val_perplexity = val_results["perplexity"]
                
                print(f"\nValidation results - Step {global_step}:")
                print(f"  Loss: {val_loss:.4f}")
                print(f"  Perplexity: {val_perplexity:.2f}")
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    checkpoint_path = os.path.join(args.checkpoint_dir, "best_model.pt")
                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        step=global_step,
                        epoch=epoch,
                        loss=val_loss,
                        config=config,
                        path=checkpoint_path,
                    )
                    print(f"New best model saved with validation loss: {val_loss:.4f}")
        
        # Compute average training loss for the epoch
        epoch_train_loss = train_loss / max(1, train_steps)
        epoch_train_perplexity = np.exp(min(epoch_train_loss, 100))
        
        # Evaluate after each epoch
        val_results = evaluate(model, val_dataloader)
        val_loss = val_results["loss"]
        val_perplexity = val_results["perplexity"]
        
        # Store metrics for plotting
        train_losses.append(epoch_train_loss)
        val_losses.append(val_loss)
        train_perplexities.append(epoch_train_perplexity)
        val_perplexities.append(val_perplexity)
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{args.num_epochs} summary:")
        print(f"  Train loss: {epoch_train_loss:.4f}, perplexity: {epoch_train_perplexity:.2f}")
        print(f"  Val loss: {val_loss:.4f}, perplexity: {val_perplexity:.2f}")
        print(f"  Time elapsed: {timer.elapsed_str()}")
        
        # Save checkpoint after each epoch
        checkpoint_path = os.path.join(args.checkpoint_dir, f"checkpoint-epoch-{epoch+1}.pt")
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=global_step,
            epoch=epoch+1,
            loss=val_loss,
            config=config,
            path=checkpoint_path,
        )
        
        # Plot training curves
        plot_path = os.path.join(args.plots_dir, "training_curves.png")
        plot_training_curves(
            train_losses=train_losses,
            val_losses=val_losses,
            train_perplexities=train_perplexities,
            val_perplexities=val_perplexities,
            output_path=plot_path,
        )
    
    # Save final model
    final_checkpoint_path = os.path.join(args.checkpoint_dir, "final_model.pt")
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=global_step,
        epoch=args.num_epochs,
        loss=val_loss,
        config=config,
        path=final_checkpoint_path,
    )
    
    # Print final summary
    timer.stop()
    print("\nTraining completed!")
    print(f"  Best validation loss: {best_val_loss:.4f}")
    print(f"  Final validation loss: {val_loss:.4f}")
    print(f"  Final validation perplexity: {val_perplexity:.2f}")
    print(f"  Total training time: {timer.elapsed_str()}")
    print(f"  Model parameters: {param_count / 1e6:.2f}M")
    

def main():
    parser = argparse.ArgumentParser(description="Train a MiniTransformer language model")
    
    # Model configuration
    parser.add_argument("--hidden_size", type=int, default=384, help="Hidden size")
    parser.add_argument("--num_layers", type=int, default=6, help="Number of layers")
    parser.add_argument("--num_heads", type=int, default=6, help="Number of attention heads")
    parser.add_argument("--intermediate_size", type=int, default=1536, help="Intermediate size for feed-forward layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability")
    parser.add_argument("--context_length", type=int, default=1024, help="Maximum context length")
    parser.add_argument("--vocab_size", type=int, default=50257, help="Vocabulary size (will be overridden if tokenizer is found)")
    
    # Training configuration
    parser.add_argument("--data_dir", type=str, default="data/wikitext-2", help="Data directory")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--plots_dir", type=str, default="plots", help="Plots directory")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--min_lr_ratio", type=float, default=0.1, help="Minimum learning rate ratio")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm")
    parser.add_argument("--stride", type=int, default=512, help="Stride for sliding window")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adam", "adamw", "sgd"], help="Optimizer")
    parser.add_argument("--scheduler", type=str, default="linear", choices=["linear", "cosine", "cosine_restart"], help="Learning rate scheduler")
    parser.add_argument("--mixed_precision", action="store_true", help="Use mixed precision training")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument("--save_steps", type=int, default=1000, help="Save checkpoint every X steps")
    parser.add_argument("--eval_steps", type=int, default=500, help="Evaluate every X steps")
    parser.add_argument("--eval_max_steps", type=int, default=100, help="Maximum steps for intermediate evaluation")
    parser.add_argument("--resume_from", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA")
    parser.add_argument("--no_progress_bar", action="store_true", help="Disable progress bar")
    
    args = parser.parse_args()
    
    train(args)


if __name__ == "__main__":
    main() 