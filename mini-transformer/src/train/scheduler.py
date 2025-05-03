import math
import numpy as np
import torch
from torch.optim.lr_scheduler import _LRScheduler


class WarmupLRScheduler(_LRScheduler):
    """Learning rate scheduler with warmup."""
    
    def __init__(
        self,
        optimizer,
        warmup_steps,
        max_steps,
        min_lr_ratio=0.1,
        last_epoch=-1,
    ):
        """Initialize the scheduler.
        
        Args:
            optimizer: PyTorch optimizer
            warmup_steps: Number of warmup steps
            max_steps: Maximum number of steps
            min_lr_ratio: Minimum learning rate ratio
            last_epoch: Last epoch
        """
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self):
        """Get the learning rate.
        
        Returns:
            Learning rate
        """
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            lr_ratio = self.last_epoch / max(1, self.warmup_steps)
        else:
            # Linear decay
            progress = (self.last_epoch - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
            lr_ratio = max(self.min_lr_ratio, 1.0 - progress)
            
        return [base_lr * lr_ratio for base_lr in self.base_lrs]


class CosineWarmupLRScheduler(_LRScheduler):
    """Cosine learning rate scheduler with warmup."""
    
    def __init__(
        self,
        optimizer,
        warmup_steps,
        max_steps,
        min_lr_ratio=0.1,
        last_epoch=-1,
    ):
        """Initialize the scheduler.
        
        Args:
            optimizer: PyTorch optimizer
            warmup_steps: Number of warmup steps
            max_steps: Maximum number of steps
            min_lr_ratio: Minimum learning rate ratio
            last_epoch: Last epoch
        """
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self):
        """Get the learning rate.
        
        Returns:
            Learning rate
        """
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            lr_ratio = self.last_epoch / max(1, self.warmup_steps)
        else:
            # Cosine decay
            progress = (self.last_epoch - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
            lr_ratio = self.min_lr_ratio + 0.5 * (1.0 - self.min_lr_ratio) * (1.0 + math.cos(math.pi * progress))
            
        return [base_lr * lr_ratio for base_lr in self.base_lrs]


class CosineWarmupRestartLRScheduler(_LRScheduler):
    """Cosine learning rate scheduler with warmup and restarts."""
    
    def __init__(
        self,
        optimizer,
        warmup_steps,
        cycle_length,
        max_steps,
        min_lr_ratio=0.1,
        restart_multiplier=0.8,
        last_epoch=-1,
    ):
        """Initialize the scheduler.
        
        Args:
            optimizer: PyTorch optimizer
            warmup_steps: Number of warmup steps
            cycle_length: Length of each cycle
            max_steps: Maximum number of steps
            min_lr_ratio: Minimum learning rate ratio
            restart_multiplier: Multiplier for the next restart
            last_epoch: Last epoch
        """
        self.warmup_steps = warmup_steps
        self.cycle_length = cycle_length
        self.max_steps = max_steps
        self.min_lr_ratio = min_lr_ratio
        self.restart_multiplier = restart_multiplier
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self):
        """Get the learning rate.
        
        Returns:
            Learning rate
        """
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            lr_ratio = self.last_epoch / max(1, self.warmup_steps)
        else:
            # Calculate which cycle we're in
            cycle_steps = self.last_epoch - self.warmup_steps
            cycle_idx = cycle_steps // self.cycle_length
            cycle_progress = (cycle_steps % self.cycle_length) / self.cycle_length
            
            # Cosine decay with restart
            amplitude = (1.0 - self.min_lr_ratio) * (self.restart_multiplier ** cycle_idx)
            lr_ratio = self.min_lr_ratio + 0.5 * amplitude * (1.0 + math.cos(math.pi * cycle_progress))
            
        return [base_lr * lr_ratio for base_lr in self.base_lrs]


def get_scheduler(name, optimizer, warmup_steps, max_steps, **kwargs):
    """Get a learning rate scheduler.
    
    Args:
        name: Scheduler name (linear, cosine, or cosine_restart)
        optimizer: PyTorch optimizer
        warmup_steps: Number of warmup steps
        max_steps: Maximum number of steps
        **kwargs: Additional arguments
        
    Returns:
        Learning rate scheduler
    """
    if name == "linear":
        return WarmupLRScheduler(optimizer, warmup_steps, max_steps, **kwargs)
    elif name == "cosine":
        return CosineWarmupLRScheduler(optimizer, warmup_steps, max_steps, **kwargs)
    elif name == "cosine_restart":
        cycle_length = kwargs.pop("cycle_length", max_steps // 4)
        return CosineWarmupRestartLRScheduler(optimizer, warmup_steps, cycle_length, max_steps, **kwargs)
    else:
        raise ValueError(f"Unknown scheduler: {name}") 