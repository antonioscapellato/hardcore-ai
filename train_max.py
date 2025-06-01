import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from pathlib import Path
import torch
import copy
import argparse

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
from test.evaluation import evaluate_accuracy
from train.trainer import train_model
from train.logger import get_writer

def main():
    parser = argparse.ArgumentParser(
        description="Train CIFAR-10/100 ResNet20 models in simple mode"
    )
    parser.add_argument(
        "--simple_epochs",
        type=int,
        default=7,
        help="Number of epochs to train (default: 7)",
    )
    parser.add_argument(
        "--init_pth_cifar10",
        type=str,
        default=None,
        help="Path to .pth file to initialize weights for cifar10_resnet20 model",
    )
    parser.add_argument(
        "--init_pth_cifar100",
        type=str,
        default=None,
        help="Path to .pth file to initialize weights for cifar100_resnet20 model",
    )
    args = parser.parse_args()

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

    # Define models to train (always simple mode)
    models = [
        (
            "cifar100_resnet20",
            get_backbone("cifar100_resnet20"),
            cifar_100_train_loader,
            cifar_100_test_loader,
            submission_config_cifar100,
        ),
        (
            "cifar10_resnet20",
            get_backbone("cifar10_resnet20"),
            cifar_10_train_loader,
            cifar_10_test_loader,
            submission_config_cifar10,
        ),
    ]

    for model_name, model, train_loader, test_loader, model_patch in models:
        print(f"Starting simple training for {model_name} on device {device}")

        # Patch and move model
        patched_model = bb.patch_model(copy.deepcopy(model), model_patch)
        patched_model.to(device)

        # Freeze all parameters
        for param in patched_model.parameters():
            param.requires_grad = False

        # Unfreeze parameters in LearnedProjKernel
        from bitbybit.kernel.learned import LearnedProjKernel
        for module in patched_model.modules():
            if isinstance(module, LearnedProjKernel):
                for param in module.parameters():
                    param.requires_grad = True

        # Setup TensorBoard writer
        writer = get_writer(str(OUTPUT_DIR), f"{model_name}_simple")
        criterion = torch.nn.CrossEntropyLoss()
        learning_rate = 0.001
        # Create optimizer only on learnable parameters
        learnable_params = [param for param in patched_model.parameters() if param.requires_grad]
        optimizer = torch.optim.Adam(learnable_params, lr=learning_rate)

        # Train for specified epochs
        train_model(
            patched_model,
            copy.deepcopy(model).to(device),
            train_loader,
            test_loader,
            criterion,
            optimizer,
            args.simple_epochs,
            device,
            writer,
            OUTPUT_DIR,
            f"{model_name}_simple",
            log_interval=50,
        )

        # Save and report final accuracy
        final_acc = evaluate_accuracy(patched_model, test_loader, device)
        checkpoint_path = OUTPUT_DIR / f"{model_name}_simple.pth"
        torch.save(patched_model.state_dict(), checkpoint_path)
        print(f"[{model_name}] Simple Mode Final Accuracy: {final_acc:.2f}%")
        print(f"[{model_name}] Simple Mode Model saved to {checkpoint_path}")
        writer.close()

if __name__ == "__main__":
    main()