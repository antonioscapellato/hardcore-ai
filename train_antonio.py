from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from bitbybit.utils.models import get_backbone
from bitbybit.utils.data import get_loaders, CIFAR10_MEAN, CIFAR10_STD, CIFAR100_MEAN, CIFAR100_STD
import bitbybit as bb
from bitbybit.config.resnet20 import resnet20_full_patch_config

OUTPUT_DIR = Path(__file__).parent / "submission_checkpoints"

def train_model(model, train_loader, test_loader, model_name, epochs=5, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Train and evaluate the model, then save the hashed model.
    
    Args:
        model: The PyTorch model to train.
        train_loader: DataLoader for training data.
        test_loader: DataLoader for test data.
        model_name: Name of the model for saving the checkpoint.
        epochs: Number of training epochs.
        device: Device to run the training on (cuda or cpu).
    """
    print(f"\n{'='*50}")
    print(f"Starting training for {model_name}")
    print(f"Training on device: {device}")
    print(f"Number of epochs: {epochs}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"{'='*50}\n")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Training phase
        model.train()
        running_loss = 0.0
        batch_count = len(train_loader)
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == batch_count:
                print(f"Batch [{batch_idx + 1}/{batch_count}] - Loss: {loss.item():.4f}")
        
        epoch_loss = running_loss / len(train_loader.dataset)
        
        # Evaluation phase
        print("\nEvaluating model...")
        model.eval()
        correct = 0
        total = 0
        test_loss = 0.0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        avg_test_loss = test_loss / len(test_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs} Summary:")
        print(f"  Training Loss: {epoch_loss:.4f}")
        print(f"  Test Loss: {avg_test_loss:.4f}")
        print(f"  Test Accuracy: {accuracy:.2f}%")
        print(f"  Correct/Total: {correct}/{total}")
        
        scheduler.step()

    print(f"\n{'='*50}")
    print(f"Training completed for {model_name}")
    print(f"Final Test Accuracy: {accuracy:.2f}%")
    print(f"Saving model to {OUTPUT_DIR / f'{model_name}.pth'}")
    print(f"{'='*50}\n")

    # Store hashed model
    hashed_model = bb.patch_model(model, config=resnet20_full_patch_config)
    torch.save(hashed_model.state_dict(), OUTPUT_DIR / f"{model_name}.pth")

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cifar_10_train_loader, cifar_10_test_loader = get_loaders(
        dataset_name="CIFAR10",
        data_dir=Path(__file__).parent / "data",
        batch_size=128,
        mean=CIFAR10_MEAN,
        std=CIFAR10_STD,
        num_workers=2,
        pin_memory=True,
    )

    cifar_100_train_loader, cifar_100_test_loader = get_loaders(
        dataset_name="CIFAR100",
        data_dir=Path(__file__).parent / "data",
        batch_size=128,     
        mean=CIFAR100_MEAN,
        std=CIFAR100_STD,
        num_workers=2,
        pin_memory=True,
    )

    models = [
        ("cifar10_resnet20", get_backbone("cifar10_resnet20"), cifar_10_train_loader, cifar_10_test_loader),
        ("cifar100_resnet20", get_backbone("cifar100_resnet20"), cifar_100_train_loader, cifar_100_test_loader),
    ]

    for model_name, model, train_loader, test_loader in models:
        train_model(model, train_loader, test_loader, model_name)

if __name__ == "__main__":
    main()