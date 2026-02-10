"""Training script for medical image augmentation system."""

import argparse
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from src.training.config import load_config, print_config
from src.training.trainer import TrainingPipeline
from src.data import load_jsonl_folder, validate_paths, MultimodalDataset, collate_fn, log_data_statistics


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train medical image augmentation model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yaml",
        help="Path to training configuration file"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    return parser.parse_args()


def create_dataloaders(config):
    """
    Create training and validation dataloaders.
    
    Args:
        config: Configuration object
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)
    
    # Load training data
    print(f"\nLoading training data from: {config.data.train_data_path}")
    train_data = load_jsonl_folder(config.data.train_data_path)
    train_data = validate_paths(train_data)
    log_data_statistics(train_data, "Training Set")
    
    # Load validation data
    print(f"\nLoading validation data from: {config.data.val_data_path}")
    val_data = load_jsonl_folder(config.data.val_data_path)
    val_data = validate_paths(val_data)
    log_data_statistics(val_data, "Validation Set")
    
    # Create datasets
    train_dataset = MultimodalDataset(
        train_data,
        image_size=(config.data.image_size, config.data.image_size),
        normalize=True,
        auto_caption=True
    )
    
    val_dataset = MultimodalDataset(
        val_data,
        image_size=(config.data.image_size, config.data.image_size),
        normalize=True,
        auto_caption=True
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print("="*70 + "\n")
    
    return train_loader, val_loader


def main():
    """Main training function."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    print_config(config)
    
    # Override resume path if provided
    if args.resume:
        config.training.resume_from = args.resume
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(config)
    
    # Create training pipeline
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pipeline = TrainingPipeline(config, device=device)
    
    # Setup model, optimizer, scheduler
    pipeline.setup_model()
    
    # Resume from checkpoint if specified
    if config.training.resume_from:
        pipeline.load_checkpoint(config.training.resume_from)
    
    # Training loop
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)
    print(f"Total epochs: {config.training.num_epochs}")
    print(f"Starting from epoch: {pipeline.current_epoch + 1}")
    print(f"Device: {device}")
    print(f"Mixed precision: {config.training.use_amp} ({config.training.amp_dtype})")
    print(f"Batch size: {config.data.batch_size}")
    print(f"Learning rate: {config.training.learning_rate}")
    print("="*70 + "\n")
    
    try:
        for epoch in range(pipeline.current_epoch + 1, config.training.num_epochs + 1):
            # Train for one epoch
            train_loss = pipeline.train_epoch(train_loader, epoch)
            
            print(f"\nEpoch {epoch}/{config.training.num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Learning Rate: {pipeline.optimizer.param_groups[0]['lr']:.6f}")
            
            # Update learning rate
            pipeline.scheduler.step()
            
            # Save checkpoint
            if epoch % config.training.save_every == 0:
                is_best = train_loss < pipeline.best_loss
                if is_best:
                    pipeline.best_loss = train_loss
                
                pipeline.save_checkpoint(epoch, train_loss, is_best=is_best)
            
            # Save training log
            pipeline.save_training_log()
        
        print("\n" + "="*70)
        print("TRAINING COMPLETED!")
        print("="*70)
        print(f"Best loss: {pipeline.best_loss:.4f}")
        print(f"Total steps: {pipeline.global_step}")
        print(f"Checkpoints saved to: {config.training.checkpoint_dir}")
        print(f"Logs saved to: {config.training.log_dir}")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        print("Saving checkpoint...")
        pipeline.save_checkpoint(
            epoch=pipeline.current_epoch,
            loss=train_loss if 'train_loss' in locals() else float('inf'),
            is_best=False
        )
        print("Checkpoint saved. You can resume training with --resume")
    
    except Exception as e:
        print(f"\n\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
