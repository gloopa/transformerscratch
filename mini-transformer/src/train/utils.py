import os
import math
import time
import datetime
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.cuda.amp import GradScaler


def save_checkpoint(model, optimizer, scheduler, step, epoch, loss, config, path):
    """Save a checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer to save
        scheduler: Learning rate scheduler to save
        step: Current step
        epoch: Current epoch
        loss: Current loss
        config: Model configuration
        path: Path to save the checkpoint
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "step": step,
        "epoch": epoch,
        "loss": loss,
        "config": config.to_dict() if hasattr(config, "to_dict") else config,
    }, path)
    
    print(f"Checkpoint saved to {path}")


def load_checkpoint(model, optimizer=None, scheduler=None, path=None):
    """Load a checkpoint.
    
    Args:
        model: Model to load
        optimizer: Optimizer to load
        scheduler: Learning rate scheduler to load
        path: Path to the checkpoint
        
    Returns:
        Dictionary with loaded information
    """
    if not os.path.exists(path):
        return {
            "step": 0,
            "epoch": 0,
            "loss": float("inf"),
            "config": model.config if hasattr(model, "config") else None,
        }
        
    checkpoint = torch.load(path, map_location=torch.device("cpu"))
    
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
    if scheduler is not None and "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"] is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    return {
        "step": checkpoint.get("step", 0),
        "epoch": checkpoint.get("epoch", 0),
        "loss": checkpoint.get("loss", float("inf")),
        "config": checkpoint.get("config", None),
    }


def compute_perplexity(loss):
    """Compute perplexity from loss.
    
    Args:
        loss: Cross-entropy loss
        
    Returns:
        Perplexity
    """
    return math.exp(min(loss, 100))


def gradient_clipping(model, optimizer, max_norm=1.0):
    """Apply gradient clipping.
    
    Args:
        model: Model
        optimizer: Optimizer
        max_norm: Maximum norm
        
    Returns:
        Gradient norm
    """
    # Compute gradient norm
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    
    # Clip gradients
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for p in model.parameters():
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)
                
    return total_norm


def train_step(model, batch, optimizer, scheduler=None, max_grad_norm=1.0, use_amp=False, scaler=None):
    """Perform one training step.
    
    Args:
        model: Model to train
        batch: Batch of data
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        max_grad_norm: Maximum gradient norm for clipping
        use_amp: Whether to use automatic mixed precision
        scaler: Gradient scaler for AMP
        
    Returns:
        Dictionary with loss and other training metrics
    """
    model.train()
    optimizer.zero_grad()
    
    # Move batch to device
    device = next(model.parameters()).device
    batch = {k: v.to(device) for k, v in batch.items()}
    
    # Forward pass with or without mixed precision
    if use_amp:
        with torch.cuda.amp.autocast():
            outputs = model(**batch)
            loss = outputs["loss"]
            
        # Backward pass with scaler
        scaler.scale(loss).backward()
        grad_norm = None  # Can't compute accurate grad norm with AMP
        
        # Clip gradients and update weights with scaler
        scaler.unscale_(optimizer)
        grad_norm = gradient_clipping(model, optimizer, max_grad_norm)
        
        scaler.step(optimizer)
        scaler.update()
    else:
        # Forward pass
        outputs = model(**batch)
        loss = outputs["loss"]
        
        # Backward pass
        loss.backward()
        
        # Clip gradients
        grad_norm = gradient_clipping(model, optimizer, max_grad_norm)
        
        # Update weights
        optimizer.step()
    
    # Update learning rate
    if scheduler is not None:
        scheduler.step()
        
    # Compute perplexity
    perplexity = compute_perplexity(loss.item())
    
    return {
        "loss": loss.item(),
        "perplexity": perplexity,
        "grad_norm": grad_norm,
        "lr": optimizer.param_groups[0]["lr"] if scheduler is not None else None,
    }


def evaluate(model, dataloader, max_steps=None):
    """Evaluate the model on a dataset.
    
    Args:
        model: Model to evaluate
        dataloader: Dataloader for the dataset
        max_steps: Maximum number of steps to evaluate (None for all)
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    total_loss = 0.0
    total_steps = 0
    
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            if max_steps is not None and step >= max_steps:
                break
                
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs["loss"]
            
            # Accumulate loss
            total_loss += loss.item()
            total_steps += 1
    
    # Compute average loss and perplexity
    avg_loss = total_loss / max(1, total_steps)
    perplexity = compute_perplexity(avg_loss)
    
    return {
        "loss": avg_loss,
        "perplexity": perplexity,
    }


def get_optimizer(model, name="adamw", learning_rate=1e-4, weight_decay=0.01, **kwargs):
    """Get an optimizer.
    
    Args:
        model: Model
        name: Optimizer name (adam, adamw, or sgd)
        learning_rate: Learning rate
        weight_decay: Weight decay
        **kwargs: Additional arguments
        
    Returns:
        Optimizer
    """
    # Prepare parameter groups with weight decay
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    
    # Create optimizer
    if name.lower() == "adam":
        return torch.optim.Adam(
            optimizer_grouped_parameters,
            lr=learning_rate,
            **kwargs
        )
    elif name.lower() == "adamw":
        return torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=learning_rate,
            **kwargs
        )
    elif name.lower() == "sgd":
        return torch.optim.SGD(
            optimizer_grouped_parameters,
            lr=learning_rate,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def plot_training_curves(train_losses, val_losses, train_perplexities, val_perplexities, output_path):
    """Plot training curves.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        train_perplexities: List of training perplexities
        val_perplexities: List of validation perplexities
        output_path: Path to save the plot
    """
    plt.figure(figsize=(12, 10))
    
    # Plot losses
    plt.subplot(2, 1, 1)
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Validation")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.title("Loss Curves")
    plt.legend()
    plt.grid(True)
    
    # Plot perplexities
    plt.subplot(2, 1, 2)
    plt.plot(train_perplexities, label="Train")
    plt.plot(val_perplexities, label="Validation")
    plt.ylabel("Perplexity")
    plt.xlabel("Epochs")
    plt.title("Perplexity Curves")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    print(f"Training curves saved to {output_path}")


def get_amp_scaler(enabled=True):
    """Get an AMP gradient scaler.
    
    Args:
        enabled: Whether to enable AMP
        
    Returns:
        Gradient scaler
    """
    if enabled:
        return GradScaler()
    return None


class Timer:
    """Simple timer for tracking execution time."""
    
    def __init__(self):
        """Initialize the timer."""
        self.start_time = None
        self.end_time = None
        
    def start(self):
        """Start the timer."""
        self.start_time = time.time()
        self.end_time = None
        
    def stop(self):
        """Stop the timer."""
        self.end_time = time.time()
        
    def elapsed(self):
        """Get the elapsed time.
        
        Returns:
            Elapsed time in seconds
        """
        if self.start_time is None:
            return 0.0
            
        end_time = self.end_time if self.end_time is not None else time.time()
        return end_time - self.start_time
        
    def elapsed_str(self):
        """Get the elapsed time as a formatted string.
        
        Returns:
            Formatted elapsed time
        """
        seconds = self.elapsed()
        return str(datetime.timedelta(seconds=seconds))


class MovingAverage:
    """Moving average for tracking metrics."""
    
    def __init__(self, window_size=100):
        """Initialize the moving average.
        
        Args:
            window_size: Window size
        """
        self.window_size = window_size
        self.values = []
        self.sum = 0.0
        
    def add(self, value):
        """Add a value.
        
        Args:
            value: Value to add
        """
        self.values.append(value)
        self.sum += value
        
        if len(self.values) > self.window_size:
            self.sum -= self.values.pop(0)
            
    def get(self):
        """Get the current average.
        
        Returns:
            Current average
        """
        if not self.values:
            return 0.0
            
        return self.sum / len(self.values) 