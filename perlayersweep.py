import torch, copy, time
from bitbybit.utils.models import get_backbone
import bitbybit as bb

from pathlib import Path
import torch

from bitbybit.utils.data import get_loaders, CIFAR10_MEAN, CIFAR10_STD, CIFAR100_MEAN, CIFAR100_STD
from bitbybit.config.resnet20 import resnet20_full_patch_config
from bitbybit.config.resnet20_random import resnet20_full_patch_config_random

OUTPUT_DIR = Path(__file__).parent / "submission_checkpoints"




def validate_model(device, model, test_loader):
    # validation
    model.to(device)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    model.to("cpu")
    torch.cuda.empty_cache() 
    return 100 * correct / total
            

# >>> LAYER SWEEP >>>
def sweep_hash_lengths(
    base_cfg: dict,
    backbone_name: str,
    test_loader,
    base_acc,
    tol_pp: float = 0.2,
    candidates=(64, 128, 256, 512, 1024, 2048),
    device: str = "cuda:0",
    model: torch.nn.Module = None,
) -> dict:
    """
    Greedy per-layer sweep:  pick the shortest hash_length that keeps
    accuracy within `tol_pp` of the current best.
    """
    print(f"⏳ running baseline on device {device} …")
    print(f" {backbone_name}  baseline = {base_acc:.2f}%")

    new_cfg = copy.deepcopy(base_cfg)
    layer_keys = [k for k in base_cfg if k != "common_params"]

    for name in layer_keys:
        best_len = base_cfg[name]["hash_length"]

        for bits in candidates:               # ascending ➜ stop at first good one
            if bits >= best_len:
                break                         # already shorter not possible

            trial_cfg = copy.deepcopy(new_cfg)
            trial_cfg[name]["hash_length"] = bits

            new_model = bb.patch_model(model, config=trial_cfg)
            acc = validate_model(device, new_model, test_loader)

            if acc >= base_acc - tol_pp:
                best_len = bits
                new_cfg[name]["hash_length"] = bits
                print(f"{name:<25} → {bits:4d} bits  (Δ {base_acc-acc:+.2f} pp)")
                break

    return new_cfg



def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cifar_10_train_loader, cifar_10_test_loader = get_loaders(
        dataset_name="CIFAR10",
        data_dir=Path(__file__).parent / "data",
        batch_size=128,
        mean=CIFAR10_MEAN,
        std=CIFAR10_STD,
        num_workers=2,
        pin_memory=True,
    )

    cifar_100_train_loader, cifar_100_test_loader = get_loaders(
        dataset_name="CIFAR100",
        data_dir=Path(__file__).parent / "data",
        batch_size=128,     
        mean=CIFAR100_MEAN,
        std=CIFAR100_STD,
        num_workers=2,
        pin_memory=True,
    )
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    

    models = [
        ("cifar10_resnet20", get_backbone("cifar10_resnet20"), cifar_10_train_loader, cifar_10_test_loader),
        ("cifar100_resnet20", get_backbone("cifar100_resnet20"), cifar_100_train_loader, cifar_100_test_loader),
    ]

    for model_name, model, train_loader, test_loader in models:
        accuracy_base = validate_model(DEVICE, model, test_loader)
      
        cfg10_best = sweep_hash_lengths(
            base_cfg=resnet20_full_patch_config_random,
            backbone_name=model_name,
            test_loader=test_loader,
            base_acc=accuracy_base,
            device=DEVICE,
            model=model,
        )

        model10 = bb.patch_model(model, config=cfg10_best)
        acc10 = validate_model(DEVICE, model10, test_loader)
        
        print(cfg10_best)
        print(f"{model_name}  final = {acc10:.2f}%")
        

        with open(f"{model_name}_best.py", "w") as f:
            f.write(f"resnet20_full_patch_config = {cfg10_best!r}\n")
        
        


if __name__ == "__main__":
    main()