"""
Training script for a DeepCAM CNN model using LearnedProjKernel for MNIST classification.
This script implements a CNN architecture that uses Locality-Sensitive Hashing (LSH)
with learned projections for efficient similarity computation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import math
import logging
import time
import sys
from pathlib import Path

# Add the src directory to Python path
src_path = str(Path(__file__).parent / "src")
if src_path not in sys.path:
    sys.path.append(src_path)

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

from bitbybit.kernel.learned import LearnedProjKernel

class DeepCAMCNN(nn.Module):
    """
    DeepCAM CNN model that combines convolutional layers with LearnedProjKernel.
    
    This model uses a CNN backbone to extract features from images, then applies
    Locality-Sensitive Hashing with learned projections to compute similarities
    between features and class prototypes.
    
    Architecture:
    - Two convolutional layers with ReLU and max pooling
    - Feature normalization to unit vectors
    - LSH-based similarity computation using learned projections
    """
    
    def __init__(self, in_channels=1, num_classes=10, hash_length=256):
        """
        Initialize the DeepCAM CNN model.
        
        Args:
            in_channels (int): Number of input channels (1 for MNIST)
            num_classes (int): Number of output classes
            hash_length (int): Length of binary hash codes for LSH
        """
        logger.info(f"Initializing DeepCAMCNN with in_channels={in_channels}, "
                   f"num_classes={num_classes}, hash_length={hash_length}")
        super(DeepCAMCNN, self).__init__()
        
        # First convolutional layer: 1->32 channels, 3x3 kernel
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        logger.debug("Initialized first conv layer: 1->32 channels")
        
        # Second convolutional layer: 32->64 channels, 3x3 kernel
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        logger.debug("Initialized second conv layer: 32->64 channels")
        
        # Calculate input features for LearnedProjKernel
        # For MNIST (28x28): after two pooling layers -> 7x7 spatial dimension
        self.in_features = 64 * 7 * 7
        self.hash_length = hash_length
        logger.debug(f"Feature dimension after CNN: {self.in_features}")
        
        # Initialize LSH kernel with learned projections
        self.lsh = LearnedProjKernel(
            in_features=self.in_features,
            out_features=num_classes,
            hash_length=hash_length
        )
        logger.info("Initialized LearnedProjKernel for LSH")
        
        # Initialize learnable class prototypes
        self.class_prototypes = nn.Parameter(torch.randn(num_classes, hash_length))
        logger.debug("Initialized learnable class prototypes")
        
        # L2 norm parameter for scaling the output
        self.l2_norm = nn.Parameter(torch.ones(1), requires_grad=True)
        logger.debug("Initialized L2 norm parameter")

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        logger.debug(f"Forward pass with input shape: {x.shape}")
        
        # Feature extraction through CNN layers
        x = self.pool(self.relu(self.conv1(x)))
        logger.debug(f"After first conv block shape: {x.shape}")
        
        x = self.pool(self.relu(self.conv2(x)))
        logger.debug(f"After second conv block shape: {x.shape}")
        
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, features)
        logger.debug(f"Flattened features shape: {x.shape}")
        
        # Normalize features to unit vectors for LSH
        unit_vectors = x / (torch.norm(x, p=2, dim=1, keepdim=True) + 1e-8)
        logger.debug(f"Normalized unit vectors shape: {unit_vectors.shape}")
        
        # Compute hash codes for input features
        codes = self.lsh._compute_codes_internal(unit_vectors)
        logger.debug(f"Generated hash codes shape: {codes.shape}")
        
        # Use learned class prototypes
        codes_2 = torch.sigmoid(self.class_prototypes)  # Convert to [0,1] range
        codes_2 = codes_2.unsqueeze(0).expand(x.size(0), -1, -1)  # Expand to batch size
        logger.debug(f"Class prototype codes shape: {codes_2.shape}")
        
        # Estimate cosine similarity between features and class prototypes
        cosine_theta = self.lsh._estimate_cosine_internal(codes, codes_2)
        logger.debug(f"Computed cosine similarities shape: {cosine_theta.shape}")
        
        # Scale the output using L2 norm
        output = cosine_theta * self.l2_norm
        logger.debug(f"Final output shape: {output.shape}")
        
        return output

def train_model(model, train_loader, criterion, optimizer, device, num_epochs=10):
    """
    Train the model for specified number of epochs.
    
    Args:
        model (nn.Module): The model to train
        train_loader (DataLoader): Training data loader
        criterion: Loss function
        optimizer: Optimizer for updating model parameters
        device: Device to train on (CPU/GPU)
        num_epochs (int): Number of training epochs
    """
    logger.info(f"Starting training for {num_epochs} epochs on {device}")
    model.train()
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        running_loss = 0.0
        total_batches = len(train_loader)
        
        logger.info(f"Epoch {epoch+1}/{num_epochs} started")
        
        for i, (images, labels) in enumerate(train_loader):
            batch_start_time = time.time()
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Log training progress
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

def evaluate_model(model, test_loader, device):
    """
    Evaluate the model on the test set.
    
    Args:
        model (nn.Module): The model to evaluate
        test_loader (DataLoader): Test data loader
        device: Device to evaluate on (CPU/GPU)
        
    Returns:
        float: Test accuracy percentage
    """
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

def main():
    """Main execution function for training and evaluating the model."""
    logger.info("Starting DeepCAM CNN training script")
    
    # Set up device (GPU if available, else CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Define data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    logger.info("Defined data transformations")
    
    # Load MNIST dataset
    logger.info("Loading MNIST dataset")
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, transform=transform, download=True
    )
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, transform=transform
    )
    logger.info(f"Training set size: {len(train_dataset)}")
    logger.info(f"Test set size: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
    logger.info("Created data loaders")
    
    # Initialize model, loss function, and optimizer
    logger.info("Initializing model and training components")
    model = DeepCAMCNN(in_channels=1, num_classes=10, hash_length=256).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train and evaluate the model
    logger.info("Starting model training")
    train_model(model, train_loader, criterion, optimizer, device, num_epochs=10)
    
    logger.info("Starting model evaluation")
    evaluate_model(model, test_loader, device)
    
    logger.info("Training script completed")

if __name__ == "__main__":
    main()