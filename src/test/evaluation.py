import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from bitbybit.utils.score import calculate_submission_score

def evaluate_accuracy(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    max_batches: int = None,
) -> float:
    """
    Evaluate model accuracy on the test dataset.

    Args:
        model: Model to evaluate
        test_loader: DataLoader for the test set
        device: Device to run evaluation on
        max_batches: Max number of batches to evaluate (None = full dataset)

    Returns:
        Accuracy as a percentage (0-100)
    """
    model.to(device)
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100.0 * correct / total
    return accuracy

def compute_score(
    original_model: nn.Module,
    hashed_model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    config: dict | None = None,
) -> tuple[float, float, float]:
    """
    Compute original accuracy, hashed accuracy, and submission score.

    Args:
        original_model: The unmodified pretrained model.
        hashed_model: The model after applying hash kernels and training.
        test_loader: DataLoader for the test set.
        device: Device to run evaluation on.
        config: (Optional) Configuration used when patching hashed_model, for reproducibility.

    Returns:
        A tuple: (original_accuracy, hashed_accuracy, score)
    """
    # Evaluate original model accuracy
    orig_acc = evaluate_accuracy(original_model, test_loader, device)

    # Evaluate hashed model accuracy
    hashed_acc = evaluate_accuracy(hashed_model, test_loader, device)

    # Compute accuracy drop (as fraction)
    acc_drop = (orig_acc - hashed_acc) / 100.0

    # Compute submission score
    score = calculate_submission_score(hashed_model, acc_drop=acc_drop)

    return orig_acc, hashed_acc, score