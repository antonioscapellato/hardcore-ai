from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Subset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import logging

from bitbybit.utils.models import get_backbone
from bitbybit.utils.data import get_loaders, CIFAR10_MEAN, CIFAR10_STD, CIFAR100_MEAN, CIFAR100_STD
import bitbybit as bb
from bitbybit.config.resnet20 import resnet20_full_patch_config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent / "submission_checkpoints"
TENSORBOARD_DIR = Path(__file__).parent / "runs"

def get_split_loaders(dataset_name, data_dir, batch_size, mean, std, num_workers, pin_memory):
    """Create train, validation, and test loaders with a validation split."""
    train_loader, test_loader = get_loaders(
        dataset_name=dataset_name,
        data_dir=data_dir,
        batch_size=batch_size,
        mean=mean,
        std=std,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    # Create validation split from training data
    train_dataset = train_loader.dataset
    indices = list(range(len(train_dataset)))
    train_indices, val_indices = train_test_split(
        indices, test_size=0.1, random_state=42, stratify=train_dataset.targets
    )
    
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(train_dataset, val_indices)
    
    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader

def compute_topk_accuracy(outputs, targets, k=5):
    """Compute top-k accuracy."""
    with torch.no_grad():
        _, pred = outputs.topk(k, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        return correct_k.mul_(100.0 / targets.size(0)).item()

def train_epoch(model, train_loader, criterion, optimizer, device, epoch, writer, model_name, kernel_type):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    top5_correct = 0
    
    pbar = tqdm(train_loader, desc=f"Training {model_name} ({kernel_type}) Epoch {epoch+1}")
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        top5_correct += compute_topk_accuracy(outputs, targets, k=5)
        
        pbar.set_postfix({
            'loss': total_loss / (pbar.n + 1),
            'top1_acc': 100. * correct / total,
            'top5_acc': 100. * top5_correct / (pbar.n + 1)
        })
    
    avg_loss = total_loss / len(train_loader)
    top1_acc = 100. * correct / total
    top5_acc = 100. * top5_correct / len(train_loader)
    
    writer.add_scalar(f"{model_name}/{kernel_type}/Train_Loss", avg_loss, epoch)
    writer.add_scalar(f"{model_name}/{kernel_type}/Train_Top1_Acc", top1_acc, epoch)
    writer.add_scalar(f"{model_name}/{kernel_type}/Train_Top5_Acc", top5_acc, epoch)
    
    return avg_loss, top1_acc, top5_acc

def evaluate(model, loader, criterion, device, epoch, writer, model_name, kernel_type, split="Val"):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    top5_correct = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc=f"{split} {model_name} ({kernel_type})"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            top5_correct += compute_topk_accuracy(outputs, targets, k=5)
    
    avg_loss = total_loss / len(loader)
    top1_acc = 100. * correct / total
    top5_acc = 100. * top5_correct / len(loader)
    
    writer.add_scalar(f"{model_name}/{kernel_type}/{split}_Loss", avg_loss, epoch)
    writer.add_scalar(f"{model_name}/{kernel_type}/{split}_Top1_Acc", top1_acc, epoch)
    writer.add_scalar(f"{model_name}/{kernel_type}/{split}_Top5_Acc", top5_acc, epoch)
    
    return avg_loss, top1_acc, top5_acc

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TENSORBOARD_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Training hyperparameters
    num_epochs = 5
    learning_rate = 0.1
    weight_decay = 5e-4
    hash_length = 4096
    patience = 10  # Early stopping patience
    warmup_epochs = 5  # Learning rate warmup epochs

    cifar_10_train_loader, cifar_10_val_loader, cifar_10_test_loader = get_split_loaders(
        dataset_name="CIFAR10",
        data_dir=Path(__file__).parent / "data",
        batch_size=128,
        mean=CIFAR10_MEAN,
        std=CIFAR10_STD,
        num_workers=2,
        pin_memory=True,
    )

    cifar_100_train_loader, cifar_100_val_loader, cifar_100_test_loader = get_split_loaders(
        dataset_name="CIFAR100",
        data_dir=Path(__file__).parent / "data",
        batch_size=128,
        mean=CIFAR100_MEAN,
        std=CIFAR100_STD,
        num_workers=2,
        pin_memory=True,
    )

    models = [
        ("cifar10_resnet20", get_backbone("cifar10_resnet20"), cifar_10_train_loader, cifar_10_val_loader, cifar_10_test_loader, 10),
        ("cifar100_resnet20", get_backbone("cifar100_resnet20"), cifar_100_train_loader, cifar_100_val_loader, cifar_100_test_loader, 100),
    ]

    for model_name, model, train_loader, val_loader, test_loader, num_classes in models:
        logger.info(f"\nTraining {model_name}")
        model = model.to(device)
        
        # Create configurations for both kernels
        random_config = resnet20_full_patch_config.copy()
        random_config["hash_kernel_type"] = "random_projection"
        random_config["hash_length"] = hash_length
        
        learned_config = resnet20_full_patch_config.copy()
        learned_config["hash_kernel_type"] = "learned_projection"
        learned_config["hash_length"] = hash_length
        
        for kernel_type, config in [("RandomProjKernel", random_config), ("LearnedProjKernel", learned_config)]:
            logger.info(f"\nTraining with {kernel_type}")
            writer = SummaryWriter(TENSORBOARD_DIR / f"{model_name}_{kernel_type.lower()}")
            
            # Initialize model
            patched_model = bb.patch_model(model, config)
            patched_model = patched_model.to(device)
            
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(patched_model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
            scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs)
            
            best_val_acc = 0
            epochs_no_improve = 0
            best_checkpoint_path = OUTPUT_DIR / f"{model_name}_{kernel_type.lower()}.pth"
            
            for epoch in range(num_epochs):
                # Learning rate warmup
                if epoch < warmup_epochs:
                    lr = learning_rate * (epoch + 1) / warmup_epochs
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
                
                train_loss, train_top1_acc, train_top5_acc = train_epoch(
                    patched_model, train_loader, criterion, optimizer, device, epoch, writer, model_name, kernel_type
                )
                val_loss, val_top1_acc, val_top5_acc = evaluate(
                    patched_model, val_loader, criterion, device, epoch, writer, model_name, kernel_type, "Val"
                )
                
                logger.info(f"Epoch {epoch+1}/{num_epochs} - "
                           f"Train Loss: {train_loss:.4f}, Train Top1 Acc: {train_top1_acc:.2f}%, Train Top5 Acc: {train_top5_acc:.2f}%, "
                           f"Val Loss: {val_loss:.4f}, Val Top1 Acc: {val_top1_acc:.2f}%, Val Top5 Acc: {val_top5_acc:.2f}%")
                
                # Save best model based on validation accuracy
                if val_top1_acc > best_val_acc:
                    best_val_acc = val_top1_acc
                    epochs_no_improve = 0
                    torch.save(patched_model.state_dict(), best_checkpoint_path)
                    logger.info(f"Saved best model at {best_checkpoint_path} with Val Top1 Acc: {best_val_acc:.2f}%")
                else:
                    epochs_no_improve += 1
                
                # Early stopping
                if epochs_no_improve >= patience:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
                
                # Update scheduler after warmup
                if epoch >= warmup_epochs:
                    scheduler.step()
            
            # Evaluate on test set with best model
            logger.info(f"Loading best model for {model_name} ({kernel_type}) for final test evaluation")
            patched_model.load_state_dict(torch.load(best_checkpoint_path))
            test_loss, test_top1_acc, test_top5_acc = evaluate(
                patched_model, test_loader, criterion, device, epoch, writer, model_name, kernel_type, "Test"
            )
            logger.info(f"Final Test Results - Test Loss: {test_loss:.4f}, Test Top1 Acc: {test_top1_acc:.2f}%, Test Top5 Acc: {test_top5_acc:.2f}%")
            
            writer.close()

if __name__ == "__main__":
    main()