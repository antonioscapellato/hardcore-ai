from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from bitbybit.utils.models import get_backbone
from bitbybit.utils.data import get_loaders, CIFAR10_MEAN, CIFAR10_STD, CIFAR100_MEAN, CIFAR100_STD
import bitbybit as bb

# Define the output directory for saving model checkpoints, located in the same directory as this script
OUTPUT_DIR = Path(__file__).parent / "submission_checkpoints"

class LearnedProjKernel(nn.Module):
    """
    A custom PyTorch module implementing a learned projection kernel for Locality Sensitive Hashing (LSH).
    This kernel approximates dot-products using learned projections and hashing for energy-efficient inference.
    """
    def __init__(self, input_dim, k):
        """
        Initialize the learned projection kernel with a learnable projection matrix.

        Args:
            input_dim (int): Dimensionality of the input features.
            k (int): Number of hash bits used for LSH approximation.
        """
        super().__init__()
        # Initialize a learnable projection matrix P with random values, scaled for stability
        self.P = nn.Parameter(torch.randn(k, input_dim) / torch.sqrt(torch.tensor(input_dim, dtype=torch.float)))
        self.k = k  # Store the number of hash bits

    def forward(self, W, x):
        """
        Forward pass to approximate the dot-product using LSH with learned projections.

        Args:
            W (torch.Tensor): Weight matrix of shape [output_dim, input_dim].
            x (torch.Tensor): Input vector of shape [input_dim].

        Returns:
            torch.Tensor: Approximated dot-product of shape [output_dim].
        """
        # Normalize the weight matrix rows and input vector to unit length for consistent hashing
        W_norm = W / torch.norm(W, dim=1, keepdim=True)
        x_norm = x / torch.norm(x)
        
        # Compute binary hash codes for weights and input using the learned projection matrix
        h_W = torch.sign(W_norm @ self.P.T)  # Shape: [output_dim, k]
        h_x = torch.sign(self.P @ x_norm)    # Shape: [k]
        
        # Calculate similarity as the normalized dot-product of hash codes
        s = (h_W @ h_x) / self.k  # Shape: [output_dim]
        
        # Approximate the original dot-product using a cosine function based on similarity
        return torch.cos((torch.pi / 2) * (1 - s))

def train_model(model, train_loader, test_loader, epochs=10):
    """
    Train the provided model using the specified data loaders and evaluate its performance.

    Args:
        model (nn.Module): The neural network model to train.
        train_loader (DataLoader): DataLoader providing training data.
        test_loader (DataLoader): DataLoader providing test data.
        epochs (int): Number of training epochs (default is 10).

    Returns:
        nn.Module: The trained model.
    """
    # Define the loss function as cross-entropy, suitable for classification tasks
    criterion = nn.CrossEntropyLoss()
    
    # Initialize the Adam optimizer with a learning rate of 0.001 for parameter updates
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Select device (GPU if available, otherwise CPU) and move the model to that device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training loop over the specified number of epochs
    for epoch in range(epochs):
        model.train()  # Set the model to training mode (enables dropout, batch norm, etc.)
        for inputs, labels in train_loader:
            # Move input data and labels to the selected device
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()  # Reset gradients to zero for a fresh optimization step
            outputs = model(inputs)  # Forward pass through the model
            loss = criterion(outputs, labels)  # Compute the loss
            loss.backward()  # Backward pass to compute gradients
            optimizer.step()  # Update model parameters using the optimizer

        # Evaluation phase after each epoch
        model.eval()  # Set the model to evaluation mode (disables dropout, etc.)
        correct, total = 0, 0  # Track correct predictions and total samples
        with torch.no_grad():  # Disable gradient computation for efficiency during evaluation
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)  # Forward pass on test data
                _, predicted = outputs.max(1)  # Get the index of the max log-probability (predicted class)
                total += labels.size(0)  # Increment total by batch size
                correct += predicted.eq(labels).sum().item()  # Count correct predictions
        accuracy = 100. * correct / total  # Calculate accuracy as a percentage
        print(f"Epoch {epoch+1}, Accuracy: {accuracy:.2f}%")  # Report progress
    return model  # Return the trained model

def main():
    """
    Main function to orchestrate data loading, model patching, training, and checkpoint saving.
    """
    # Ensure the output directory exists, creating it if necessary
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Set up data loaders for CIFAR-10 dataset with predefined mean and std for normalization
    cifar_10_train_loader, cifar_10_test_loader = get_loaders(
        dataset_name="CIFAR10",
        data_dir=Path(__file__).parent / "leaderboard" / "data",
        batch_size=128,  # Batch size for efficient training
        mean=CIFAR10_MEAN,  # Mean for normalization
        std=CIFAR10_STD,    # Standard deviation for normalization
        num_workers=2,      # Number of subprocesses for data loading
        pin_memory=True     # Speed up data transfer to GPU
    )
    
    # Set up data loaders for CIFAR-100 dataset with its specific mean and std
    cifar_100_train_loader, cifar_100_test_loader = get_loaders(
        dataset_name="CIFAR100",
        data_dir=Path(__file__).parent / "leaderboard" / "data",
        batch_size=128,
        mean=CIFAR100_MEAN,
        std=CIFAR100_STD,
        num_workers=2,
        pin_memory=True
    )

    # Define a list of tuples containing model names, backbones, and their respective data loaders
    models = [
        ("cifar10_resnet20", get_backbone("cifar10_resnet20"), cifar_10_train_loader, cifar_10_test_loader),
        ("cifar100_resnet20", get_backbone("cifar100_resnet20"), cifar_100_train_loader, cifar_100_test_loader),
    ]

    # Iterate over each model configuration
    for model_name, model, train_loader, test_loader in models:
        # Patch the model with the custom LearnedProjKernel for energy-efficient inference
        hashed_model = bb.patch_model(model)  # Assumes LearnedProjKernel is registered in bitbybit
        
        # Train the patched model using the provided data loaders
        trained_model = train_model(hashed_model, train_loader, test_loader)
        
        # Save the trained model's state dictionary to a file in the output directory
        torch.save(trained_model.state_dict(), OUTPUT_DIR / f"{model_name}.pth")
        
        # Print a confirmation message indicating the checkpoint has been saved
        print(f"Saved {model_name} checkpoint")

# Execute the main function if this script is run directly
if __name__ == "__main__":
    main()