"""
Training script for ResNet-20 on CIFAR-10 and CIFAR-100 with checkpoint saving.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import logging
import time
import sys
import os
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)

# Define ResNet-20 architecture
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = torch.nn.functional.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet20(num_classes=10):
    return ResNet(BasicBlock, [3, 3, 3], num_classes)

def train_model(model, train_loader, criterion, optimizer, device, num_epochs=5):
    logger.info(f"Starting training for {num_epochs} epochs on {device}")
    model.train()
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        running_loss = 0.0
        total_batches = len(train_loader)
        
        for i, (images, labels) in enumerate(train_loader):
            batch_start_time = time.time()
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if (i + 1) % 100 == 0:
                avg_loss = running_loss / 100
                batch_time = time.time() - batch_start_time
                logger.info(f"Epoch [{epoch+1}/{num_epochs}], "
                          f"Step [{i+1}/{total_batches}], "
                          f"Loss: {avg_loss:.4f}, "
                          f"Batch Time: {batch_time:.2f}s")
                running_loss = 0.0
        
        epoch_time = time.time() - epoch_start_time
        logger.info(f"Epoch {epoch+1} completed in {epoch_time:.2f} seconds")
    
    return model

def evaluate_model(model, test_loader, device):
    logger.info("Starting model evaluation")
    model.eval()
    correct = 0
    total = 0
    eval_start_time = time.time()
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if (i + 1) % 100 == 0:
                logger.debug(f"Processed {i+1} test batches")
    
    accuracy = 100 * correct / total
    eval_time = time.time() - eval_start_time
    logger.info(f"Evaluation completed in {eval_time:.2f} seconds")
    logger.info(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy

def save_checkpoint(model, filename, checkpoint_dir='submission_checkpoints'):
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(model.state_dict(), filepath)
    logger.info(f"Saved checkpoint: {filepath}")

def main():
    logger.info("Starting ResNet-20 training script for CIFAR-10 and CIFAR-100")
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Define data transformations for CIFAR
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    logger.info("Defined data transformations")
    
    # Train and evaluate on CIFAR-10
    logger.info("Loading CIFAR-10 dataset")
    train_dataset_cifar10 = torchvision.datasets.CIFAR10(
        root='./data', train=True, transform=transform, download=True
    )
    test_dataset_cifar10 = torchvision.datasets.CIFAR10(
        root='./data', train=False, transform=transform
    )
    train_loader_cifar10 = DataLoader(dataset=train_dataset_cifar10, batch_size=128, shuffle=True)
    test_loader_cifar10 = DataLoader(dataset=test_dataset_cifar10, batch_size=128, shuffle=False)
    
    logger.info("Initializing ResNet-20 for CIFAR-10")
    model_cifar10 = ResNet20(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model_cifar10.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    
    logger.info("Training on CIFAR-10")
    model_cifar10 = train_model(model_cifar10, train_loader_cifar10, criterion, optimizer, device, num_epochs=5)
    save_checkpoint(model_cifar10, 'cifar10_resnet20.pth')
    evaluate_model(model_cifar10, test_loader_cifar10, device)
    
    # Train and evaluate on CIFAR-100
    logger.info("Loading CIFAR-100 dataset")
    train_dataset_cifar100 = torchvision.datasets.CIFAR100(
        root='./data', train=True, transform=transform, download=True
    )
    test_dataset_cifar100 = torchvision.datasets.CIFAR100(
        root='./data', train=False, transform=transform
    )
    train_loader_cifar100 = DataLoader(dataset=train_dataset_cifar100, batch_size=128, shuffle=True)
    test_loader_cifar100 = DataLoader(dataset=test_dataset_cifar100, batch_size=128, shuffle=False)
    
    logger.info("Initializing ResNet-20 for CIFAR-100")
    model_cifar100 = ResNet20(num_classes=100).to(device)
    optimizer = optim.SGD(model_cifar100.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    
    logger.info("Training on CIFAR-100")
    model_cifar100 = train_model(model_cifar100, train_loader_cifar100, criterion, optimizer, device, num_epochs=5)
    save_checkpoint(model_cifar100, 'cifar100_resnet20.pth')
    evaluate_model(model_cifar100, test_loader_cifar100, device)
    
    logger.info("Training script completed")

if __name__ == "__main__":
    main()
