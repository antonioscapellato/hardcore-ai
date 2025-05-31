import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import numpy as np
import torch.nn as nn
from typing import Dict
import math

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



def next_pow2(x: int) -> int:
    return 1 << (x - 1).bit_length()

def update_hash_lengths(
    model: nn.Module,
    cfg: dict,
    collision: int = 4,           # desired (# real params)/(# buckets)
    min_len: int = 256,
    max_len: int = 16_384,
) -> dict:
    """
    Return a *new* config with hash_length auto-set per layer.
    """
    new_cfg = copy.deepcopy(cfg)
    base    = cfg.get("common_params", {})

    for name, mod in model.named_modules():
        if not isinstance(mod, (nn.Conv2d, nn.Linear)):
            continue

        P = mod.weight.numel()                                # real parameter count
        L = max(min_len, min(max_len, P // collision))        # bucket count target
        L = next_pow2(L)                                      # round to power-of-two

        layer_cfg = new_cfg.get(name, {}).copy()              # keep other fields
        # inherit untouched fields from common_params if missing
        for k in ("hash_kernel_type", "input_tile_size", "output_tile_size"):
            layer_cfg.setdefault(k, base.get(k))
        layer_cfg["hash_length"] = L
        new_cfg[name] = layer_cfg

    return new_cfg


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

        # Keep a copy of the original pretrained model for scoring
        original_model = copy.deepcopy(model)
        original_model.to(device)
    
        # Patch the model with hash kernels
        # updated_conf = update_hash_lengths(model, resnet20_full_patch_config)
        hashed_model = bb.patch_model(model, config=resnet20_full_patch_config)
        hashed_model.to(device)
    
        # Initialize loss and optimizer
        criterion = torch.nn.CrossEntropyLoss()
        # Freeze all parameters except learned projection matrices
        for param in hashed_model.parameters():
            param.requires_grad = False
        # Unfreeze learned projection matrices
        for module in hashed_model.modules():
            if isinstance(module, bb.LearnedProjKernel):
                module.projection_matrix.requires_grad = True
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, hashed_model.parameters()), lr=learning_rate)
    
        # Initialize TensorBoard writer
        writer = get_writer(str(OUTPUT_DIR), model_name)
    
        # Train and evaluate (now passing original_model to compute scores)
        train_model(
            hashed_model,
            original_model,
            train_loader,
            test_loader,
            criterion,
            optimizer,
            num_epochs,
            device,
            writer,
            OUTPUT_DIR,
            model_name,
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

        # Close the TensorBoard writer
        writer.close()

if __name__ == "__main__":
    print(">>> Launching train_max.py", flush=True)
    main()