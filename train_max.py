import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from pathlib import Path
import torch
from src.bitbybit.utils.models import get_backbone
from src.bitbybit.utils.data import (
    get_loaders,
    CIFAR10_MEAN,
    CIFAR10_STD,
    CIFAR100_MEAN,
    CIFAR100_STD,
)
from src import bitbybit as bb

from src.train.logger import get_writer
from src.train.trainer import train_model, evaluate_model

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
        hashed_model = bb.patch_model(model)
        hashed_model.to(device)

        # Initialize loss and optimizer
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(hashed_model.parameters(), lr=learning_rate)

        # Initialize TensorBoard writer
        writer = get_writer(str(OUTPUT_DIR), model_name)

        # Train and evaluate
        train_model(
            model=hashed_model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=num_epochs,
            device=device,
            writer=writer,
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

        # Save the hashed model checkpoint
        checkpoint_path = OUTPUT_DIR / f"{model_name}.pth"
        torch.save(hashed_model.state_dict(), checkpoint_path)
        print(f"Saved model checkpoint to {checkpoint_path} with final accuracy: {accuracy:.2f}%")

        # Close the TensorBoard writer
        writer.close()

if __name__ == "__main__":
    main()