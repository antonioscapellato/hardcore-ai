from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from bitbybit.utils.models import get_backbone
from bitbybit.utils.data import get_loaders, CIFAR10_MEAN, CIFAR10_STD, CIFAR100_MEAN, CIFAR100_STD
import bitbybit as bb

OUTPUT_DIR = Path(__file__).parent / "submission_checkpoints"

class LearnedProjKernel(nn.Module):
    def __init__(self, input_dim, k):
        super().__init__()
        self.P = nn.Parameter(torch.randn(k, input_dim) / torch.sqrt(torch.tensor(input_dim, dtype=torch.float)))
        self.k = k

    def forward(self, W, x):
        W_norm = W / torch.norm(W, dim=1, keepdim=True)
        x_norm = x / torch.norm(x)
        h_W = torch.sign(W_norm @ self.P.T)
        h_x = torch.sign(self.P @ x_norm)
        s = (h_W @ h_x) / self.k
        return torch.cos((torch.pi / 2) * (1 - s))

def train_model(model, train_loader, test_loader, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Evaluate
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        accuracy = 100. * correct / total
        print(f"Epoch {epoch+1}, Accuracy: {accuracy:.2f}%")
    return model

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Data loaders
    cifar_10_train_loader, cifar_10_test_loader = get_loaders(
        dataset_name="CIFAR10", data_dir=Path(__file__).parent / "leaderboard" / "data",
        batch_size=128, mean=CIFAR10_MEAN, std=CIFAR10_STD, num_workers=2, pin_memory=True
    )
    cifar_100_train_loader, cifar_100_test_loader = get_loaders(
        dataset_name="CIFAR100", data_dir=Path(__file__).parent / "leaderboard" / "data",
        batch_size=128, mean=CIFAR100_MEAN, std=CIFAR100_STD, num_workers=2, pin_memory=True
    )

    models = [
        ("cifar10_resnet20", get_backbone("cifar10_resnet20"), cifar_10_train_loader, cifar_10_test_loader),
        ("cifar100_resnet20", get_backbone("cifar100_resnet20"), cifar_100_train_loader, cifar_100_test_loader),
    ]

    for model_name, model, train_loader, test_loader in models:
        hashed_model = bb.patch_model(model)  # Assumes LearnedProjKernel is registered
        trained_model = train_model(hashed_model, train_loader, test_loader)
        torch.save(trained_model.state_dict(), OUTPUT_DIR / f"{model_name}.pth")
        print(f"Saved {model_name} checkpoint")

if __name__ == "__main__":
    main()