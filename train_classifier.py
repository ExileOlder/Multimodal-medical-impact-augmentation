"""
Downstream classification experiment to evaluate augmentation value.

This script trains a ResNet-50 classifier on:
1. Original data only
2. Original + augmented data

The accuracy improvement demonstrates the value of the augmentation system.
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from pathlib import Path
import json
from PIL import Image
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class FundusDataset(Dataset):
    """Dataset for fundus images with DR grade labels."""
    
    def __init__(self, data_list, transform=None):
        """
        Args:
            data_list: List of dicts with 'image_path' and 'label' (DR grade 0-4)
            transform: Image transformations
        """
        self.data_list = data_list
        self.transform = transform
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        item = self.data_list[idx]
        
        # Load image
        image = Image.open(item['image_path']).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        label = int(item['label'])
        
        return image, label


def create_transforms(image_size=224):
    """Create train and val transforms."""
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


def create_model(num_classes=5, pretrained=True):
    """Create ResNet-50 classifier."""
    model = models.resnet50(pretrained=pretrained)
    
    # Replace final layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    return model


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc="Training")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
    """Evaluate model."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc, all_preds, all_labels


def run_experiment(
    train_data,
    val_data,
    experiment_name,
    num_epochs=20,
    batch_size=32,
    learning_rate=1e-4,
    device=None
):
    """
    Run classification experiment.
    
    Args:
        train_data: Training data list
        val_data: Validation data list
        experiment_name: Name for this experiment
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Training device
        
    Returns:
        Dict with results
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {experiment_name}")
    print(f"{'='*70}")
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"Device: {device}")
    print(f"{'='*70}\n")
    
    # Create datasets and dataloaders
    train_transform, val_transform = create_transforms()
    
    train_dataset = FundusDataset(train_data, transform=train_transform)
    val_dataset = FundusDataset(val_data, transform=val_transform)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    model = create_model(num_classes=5, pretrained=True).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Training loop
    best_val_acc = 0.0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc, val_preds, val_labels = evaluate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"results/{experiment_name}_best.pth")
    
    # Final evaluation
    print(f"\n{'='*70}")
    print(f"FINAL RESULTS: {experiment_name}")
    print(f"{'='*70}")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(val_labels, val_preds, target_names=[f"Grade {i}" for i in range(5)]))
    
    # Confusion matrix
    cm = confusion_matrix(val_labels, val_preds)
    print("\nConfusion Matrix:")
    print(cm)
    
    results = {
        'experiment_name': experiment_name,
        'best_val_acc': best_val_acc,
        'final_train_acc': history['train_acc'][-1],
        'final_val_acc': history['val_acc'][-1],
        'history': history,
        'num_train_samples': len(train_data),
        'num_val_samples': len(val_data)
    }
    
    return results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Downstream classification experiment")
    parser.add_argument(
        "--original_data",
        type=str,
        required=True,
        help="Path to original training data manifest (JSONL)"
    )
    parser.add_argument(
        "--augmented_data",
        type=str,
        default=None,
        help="Path to augmented data manifest (JSONL)"
    )
    parser.add_argument(
        "--val_data",
        type=str,
        required=True,
        help="Path to validation data manifest (JSONL)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    args = parser.parse_args()
    
    # Create results directory
    Path("results").mkdir(exist_ok=True)
    
    # Load data
    print("Loading data...")
    
    # Load original data
    with open(args.original_data, 'r') as f:
        original_data = [json.loads(line) for line in f]
    
    # Load validation data
    with open(args.val_data, 'r') as f:
        val_data = [json.loads(line) for line in f]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Experiment 1: Original data only
    print("\n" + "="*70)
    print("EXPERIMENT 1: Original Data Only")
    print("="*70)
    
    results_original = run_experiment(
        train_data=original_data,
        val_data=val_data,
        experiment_name="original_only",
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=device
    )
    
    # Experiment 2: Original + Augmented data (if provided)
    if args.augmented_data:
        print("\n" + "="*70)
        print("EXPERIMENT 2: Original + Augmented Data")
        print("="*70)
        
        # Load augmented data
        with open(args.augmented_data, 'r') as f:
            augmented_data = [json.loads(line) for line in f]
        
        # Combine datasets
        combined_data = original_data + augmented_data
        
        results_augmented = run_experiment(
            train_data=combined_data,
            val_data=val_data,
            experiment_name="original_plus_augmented",
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device
        )
        
        # Compare results
        print("\n" + "="*70)
        print("COMPARISON")
        print("="*70)
        print(f"Original Only - Best Val Acc: {results_original['best_val_acc']:.4f}")
        print(f"Original + Augmented - Best Val Acc: {results_augmented['best_val_acc']:.4f}")
        
        improvement = (results_augmented['best_val_acc'] - results_original['best_val_acc']) * 100
        print(f"\nAccuracy Improvement: {improvement:+.2f}%")
        
        if improvement > 0:
            print("✓ Augmentation improved classification accuracy!")
        else:
            print("⚠ Augmentation did not improve accuracy (may need more data or tuning)")
        
        # Save comparison results
        comparison = {
            'original_only': results_original,
            'original_plus_augmented': results_augmented,
            'improvement_percent': improvement
        }
        
        with open("results/downstream_evaluation.json", 'w') as f:
            json.dump(comparison, f, indent=2)
        
        print(f"\nResults saved to: results/downstream_evaluation.json")
    
    else:
        print("\n⚠ No augmented data provided. Only ran baseline experiment.")
        print("To compare with augmented data, provide --augmented_data argument")
        
        # Save baseline results
        with open("results/downstream_evaluation.json", 'w') as f:
            json.dump({'original_only': results_original}, f, indent=2)


if __name__ == "__main__":
    main()
