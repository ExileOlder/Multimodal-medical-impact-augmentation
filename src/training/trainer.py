"""Training pipeline manager."""

import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import json
from typing import Optional, Dict, Any
import time

# Add codes directory to path for transport module
codes_path = Path(__file__).parent.parent.parent / "codes"
sys.path.insert(0, str(codes_path))

from transport import create_transport

from .config import Config
from ..models import NextDiTWithMask_2B_patch2
from ..data import MultimodalDataset, collate_fn


class TrainingPipeline:
    """
    Training pipeline for medical image augmentation system.
    
    Key features:
    - Flow Matching / Rectified Flow loss (from codes/transport)
    - Mixed precision training (BF16/FP16)
    - Checkpoint saving and loading
    - Training loss logging
    """
    
    def __init__(self, config: Config, device: torch.device = None):
        """
        Initialize training pipeline.
        
        Args:
            config: Configuration object
            device: Training device (defaults to cuda if available)
        """
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.transport = None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        
        # Logging
        self.train_losses = []
        self.val_losses = []
        
        print(f"Training device: {self.device}")
    
    def setup_model(self):
        """Initialize model, optimizer, scheduler, and transport."""
        print("\n" + "="*70)
        print("SETTING UP MODEL")
        print("="*70)
        
        # Create model
        print("Creating NextDiTWithMask model...")
        self.model = NextDiTWithMask_2B_patch2(
            in_channels=self.config.model.in_channels,
            mask_channels=self.config.model.mask_channels,
            dim=self.config.model.hidden_size,
            n_layers=self.config.model.depth,
            n_heads=self.config.model.num_heads,
            patch_size=self.config.model.patch_size,
            learn_sigma=False,  # Simplified for medical images
        ).to(self.device)
        
        print(f"Model parameters: {self.model.parameter_count():,}")
        
        # Create optimizer
        print(f"Creating {self.config.optimizer.type} optimizer...")
        if self.config.optimizer.type == "AdamW":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                betas=self.config.optimizer.betas,
                eps=self.config.optimizer.eps,
                weight_decay=self.config.training.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer.type}")
        
        # Create learning rate scheduler
        print(f"Creating {self.config.scheduler.type} scheduler...")
        if self.config.scheduler.type == "cosine":
            from torch.optim.lr_scheduler import CosineAnnealingLR
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.num_epochs,
                eta_min=self.config.scheduler.min_lr
            )
        else:
            raise ValueError(f"Unsupported scheduler: {self.config.scheduler.type}")
        
        # Create gradient scaler for mixed precision
        if self.config.training.use_amp:
            print(f"Enabling mixed precision training ({self.config.training.amp_dtype})...")
            self.scaler = GradScaler()
        
        # ⚠️ CRITICAL: Create transport for Flow Matching Loss
        print("Creating transport for Flow Matching loss...")
        self.transport = create_transport(
            path_type="Linear",  # Linear interpolation path
            prediction="velocity",  # Velocity prediction (Flow Matching)
            loss_weight=None,  # Uniform weighting
            train_eps=0.0,  # Stable for velocity prediction
            sample_eps=0.0,
            snr_type="uniform"  # Uniform time sampling
        )
        print("✓ Transport created (Flow Matching / Rectified Flow)")
        
        print("="*70 + "\n")
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{self.config.training.num_epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            images = batch['images'].to(self.device)
            masks = batch['masks'].to(self.device)
            captions = batch['captions']  # List of strings
            text_only_flags = batch['text_only_flags']
            
            # TODO: Encode captions to features (requires text encoder)
            # For now, create dummy caption features
            batch_size = images.shape[0]
            cap_feats = torch.randn(batch_size, 77, 5120).to(self.device)
            cap_mask = torch.ones(batch_size, 77).to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Mixed precision training
            if self.config.training.use_amp:
                with autocast(dtype=torch.bfloat16 if self.config.training.amp_dtype == "bfloat16" else torch.float16):
                    # Forward pass through model
                    # Note: transport expects x1 (target) as input
                    model_kwargs = {
                        'cap_feats': cap_feats,
                        'cap_mask': cap_mask,
                        'condition_mask': masks
                    }
                    
                    # ⚠️ CRITICAL: Use transport.training_losses for Flow Matching
                    # This computes the velocity prediction loss
                    loss_dict = self.transport.training_losses(
                        model=lambda xt, t: self.model(
                            xt, t,
                            cap_feats=model_kwargs['cap_feats'],
                            cap_mask=model_kwargs['cap_mask'],
                            condition_mask=model_kwargs['condition_mask']
                        ),
                        x1=images,  # Target images
                        model_kwargs=None  # Already passed via lambda
                    )
                    
                    loss = loss_dict['loss'].mean()
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.training.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.gradient_clip
                    )
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard training without AMP
                model_kwargs = {
                    'cap_feats': cap_feats,
                    'cap_mask': cap_mask,
                    'condition_mask': masks
                }
                
                loss_dict = self.transport.training_losses(
                    model=lambda xt, t: self.model(
                        xt, t,
                        cap_feats=model_kwargs['cap_feats'],
                        cap_mask=model_kwargs['cap_mask'],
                        condition_mask=model_kwargs['condition_mask']
                    ),
                    x1=images,
                    model_kwargs=None
                )
                
                loss = loss_dict['loss'].mean()
                loss.backward()
                
                if self.config.training.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.gradient_clip
                    )
                
                self.optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Log periodically
            if self.global_step % self.config.training.log_every == 0:
                self.train_losses.append({
                    'epoch': epoch,
                    'step': self.global_step,
                    'loss': loss.item(),
                    'lr': self.optimizer.param_groups[0]['lr']
                })
        
        avg_loss = epoch_loss / num_batches
        return avg_loss
    
    def save_checkpoint(self, epoch: int, loss: float, is_best: bool = False):
        """
        Save training checkpoint.
        
        Args:
            epoch: Current epoch
            loss: Current loss
            is_best: Whether this is the best model so far
        """
        checkpoint_dir = Path(self.config.training.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'best_loss': self.best_loss,
            'config': self.config.__dict__,
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")
        
        # Save best model
        if is_best:
            best_path = checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"Saved best model: {best_path}")
        
        # Save latest checkpoint
        latest_path = checkpoint_dir / "latest.pth"
        torch.save(checkpoint, latest_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load training checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        
        print(f"Resumed from epoch {self.current_epoch}, step {self.global_step}")
    
    def save_training_log(self):
        """Save training losses to JSON file."""
        log_dir = Path(self.config.training.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_path = log_dir / "training_log.json"
        with open(log_path, 'w') as f:
            json.dump({
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
            }, f, indent=2)
        
        print(f"Saved training log: {log_path}")
