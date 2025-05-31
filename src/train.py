from pathlib import Path
import torch

from bitbybit.utils.models import get_backbone
from bitbybit.utils.data import get_loaders, CIFAR10_MEAN, CIFAR10_STD, CIFAR100_MEAN, CIFAR100_STD
import bitbybit as bb
from bitbybit.config.resnet20 import resnet20_full_patch_config
from bitbybit.config.resnet20_learned import resnet20_full_patch_config_learned 

from bitbybit.config.conf import conf
OUTPUT_DIR = Path(__file__).parent / "submission_checkpoints"

def validate_model(device, model, test_loader):
    # validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100 * correct / total
            

def train_model(device, model, train_loader, test_loader):
    criterion = torch.nn.CrossEntropyLoss()
        # optimizer = torch.optim.AdamW(model.parameters(), lr=conf.lr)
    optimizer = torch.optim.SGD(model.parameters(), lr=conf.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=conf.epochs, eta_min=0) 
    train_loss = 0
    model.to(device)
    model.train()
    for epoch in range(conf.epochs):
        # training
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(images)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        accuracy = validate_model(device, model, test_loader)
        scheduler.step()
        print(f"Epoch [{epoch+1}/{conf.epochs}], Train Loss: {train_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    return model
        
        
    
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

    models = [
        ("cifar10_resnet20", get_backbone("cifar10_resnet20"), cifar_10_train_loader, cifar_10_test_loader),
        ("cifar100_resnet20", get_backbone("cifar100_resnet20"), cifar_100_train_loader, cifar_100_test_loader),
    ]

    for model_name, model, train_loader, test_loader in models:
        # <train model>
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        accuracy_base = validate_model(device, model, test_loader)
        print(f"Base model accuracy: {accuracy_base:.2f}%")
        
        hashed_model = bb.patch_model(model, config=resnet20_full_patch_config)
        trained_model = train_model(device, hashed_model, train_loader, test_loader)
        # Store model

        torch.save(trained_model.state_dict(), OUTPUT_DIR / f"{model_name}.pth")
        print(f"Saved {model_name} checkpoint")



        # <evaluate model>




if __name__ == "__main__":
    main()