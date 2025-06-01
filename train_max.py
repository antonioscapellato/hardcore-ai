import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from pathlib import Path
import torch
import copy
from bitbybit.utils.models import get_backbone
from bitbybit.utils.data import (
    get_loaders,
    CIFAR10_MEAN,
    CIFAR10_STD,
    CIFAR100_MEAN,
    CIFAR100_STD,
)
import bitbybit as bb
from bitbybit.config.resnet20 import submission_config_cifar10, submission_config_cifar100
from train.genetic_trainer import genetic_train
from test.evaluation import evaluate_accuracy, compute_score

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
        ("cifar100_resnet20", get_backbone("cifar100_resnet20"), cifar_100_train_loader, cifar_100_test_loader,
         submission_config_cifar100),
        ("cifar10_resnet20", get_backbone("cifar10_resnet20"), cifar_10_train_loader, cifar_10_test_loader, submission_config_cifar10),
    ]

    for model_name, model, train_loader, test_loader, model_patch in models:
        print(f"Starting genetic training for {model_name} on device {device}")

        # Keep a copy of the original pretrained model for scoring
        original_model = copy.deepcopy(model)
        original_model.to(device)

        # Run genetic algorithm
        best_model, best_score = genetic_train(
            model_name=model_name,
            original_model=original_model,
            train_loader=train_loader,
            test_loader=test_loader,
            output_dir=OUTPUT_DIR,
            device=device,
            base_config=model_patch,
            num_generations=10,
            population_size=8,
            tournament_size=3,
            mutation_rate=0.1,
            max_epochs=8,
        )

        # Evaluate final model
        hashed_accuracy = evaluate_accuracy(best_model, test_loader, device)
        print(f"[{model_name}] Best Hashed Model Accuracy: {hashed_accuracy:.2f}%")
        print(f"[{model_name}] Best Submission Score: {best_score:.4f}")

if __name__ == "__main__":
    print(">>> Launching train_max.py with genetic algorithm", flush=True)
    main()