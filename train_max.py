import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from pathlib import Path
import torch
import copy
from src.bitbybit.utils.models import get_backbone
from src.bitbybit.utils.data import (
    get_loaders,
    CIFAR10_MEAN,
    CIFAR10_STD,
    CIFAR100_MEAN,
    CIFAR100_STD,
)
from src import bitbybit as bb
from src.bitbybit.config.resnet20 import resnet20_full_patch_config

from src.train.logger import get_writer
from src.train.trainer import train_model, evaluate_model

# Import evaluation utilities
from src.test.evaluation import evaluate_accuracy, compute_score

def main():
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Output directories
    BASE_DIR = Path(__file__).parent
    OUTPUT_DIR = BASE_DIR / "submission_checkpoints"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Data loaders for CIFAR10
    cifar_10_train_loader, cifar_10_test_loader = get_loaders(
        dataset_name="CIFAR10",
        data_dir=BASE_DIR / "data",
        batch_size=128,
        mean=CIFAR10_MEAN,
        std=CIFAR10_STD,
        num_workers=2,
        pin_memory=True,
    )

    # Data loaders for CIFAR100
    cifar_100_train_loader, cifar_100_test_loader = get_loaders(
        dataset_name="CIFAR100",
        data_dir=BASE_DIR / "data",
        batch_size=128,
        mean=CIFAR100_MEAN,
        std=CIFAR100_STD,
        num_workers=2,
        pin_memory=True,
    )

    # Define models to train
    models = [
        ("cifar10_resnet20", get_backbone("cifar10_resnet20"), cifar_10_train_loader, cifar_10_test_loader),
        ("cifar100_resnet20", get_backbone("cifar100_resnet20"), cifar_100_train_loader, cifar_100_test_loader),
    ]

    # Common hyperparameters
    num_epochs = 10
    learning_rate = 0.001

    for model_name, model, train_loader, test_loader in models:
        print(f"Starting training for {model_name} on device {device}")

        # Patch the model with hash kernels
        hashed_model = bb.patch_model(model, config=resnet20_full_patch_config)
        hashed_model.to(device)
    
        # Initialize loss and optimizer
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(hashed_model.parameters(), lr=learning_rate)
    
        # Initialize TensorBoard writer
        writer = get_writer(str(OUTPUT_DIR), model_name)
    
        # Train and evaluate (now including test_loader)
        train_model(
            hashed_model,
            train_loader,
            test_loader,
            criterion,
            optimizer,
            num_epochs,
            device,
            writer,
            log_interval=50,  # Log every 50 batches
        )

        # Evaluate after final epoch
        accuracy = evaluate_model(
            model=hashed_model,
            test_loader=test_loader,
            criterion=criterion,
            device=device,
            writer=writer,
            epoch=num_epochs,
        )

        # Compute hashed_model accuracy (redundant with evaluate_model's logged accuracy),
        # but use evaluate_accuracy for consistency in scoring
        hashed_accuracy = evaluate_accuracy(hashed_model, test_loader, device)
        print(f"[{model_name}] Hashed Model Accuracy (post-hashing/training): {hashed_accuracy:.2f}%")

        # Compute submission score, using orig_accuracy computed earlier
        _, _, submission_score = compute_score(original_model, hashed_model, test_loader, device)
        print(f"[{model_name}] Submission Score: {submission_score:.4f}")
        # Log hashed_accuracy and submission_score to TensorBoard at final epoch
        writer.add_scalar("Evaluation/Hashed_Accuracy", hashed_accuracy, num_epochs)
        writer.add_scalar("Score/Submission_Score", submission_score, num_epochs)

        # Save the hashed model checkpoint
        checkpoint_path = OUTPUT_DIR / f"{model_name}.pth"
        torch.save(hashed_model.state_dict(), checkpoint_path)
        print(f"Saved model checkpoint to {checkpoint_path} with final accuracy: {accuracy:.2f}%")

        # Close the TensorBoard writer
        writer.close()

if __name__ == "__main__":
    main()