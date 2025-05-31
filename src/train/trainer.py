import time
import torch
from torch.utils.tensorboard import SummaryWriter
from src.test.evaluation import compute_score
from pathlib import Path
import os

from warnings import warn

def train_model(model: torch.nn.Module,
                original_model: torch.nn.Module,
                train_loader: torch.utils.data.DataLoader,
                test_loader: torch.utils.data.DataLoader,
                criterion: torch.nn.Module,
                optimizer: torch.optim.Optimizer,
                num_epochs: int,
                device: torch.device,
                writer: SummaryWriter,
                output_dir: Path,
                model_name: str,
                log_interval: int = 1) -> None:
    """
    Train the model, logging per-batch loss, batch accuracy, batch time, and per-epoch metrics
    to both console and TensorBoard. Also evaluates on test set and logs submission score
    after each epoch.
    """
    model.train()
    global_step = 0
    best_score = float('-inf')
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_start_time = time.time()
        for batch_idx, (inputs, labels) in enumerate(train_loader, start=1):
            batch_start_time = time.time()
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            # Compute batch accuracy
            _, predicted = torch.max(outputs.data, 1)
            correct_batch = (predicted == labels).sum().item()
            batch_acc = 100.0 * correct_batch / labels.size(0)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_end_time = time.time()
            batch_time = batch_end_time - batch_start_time

            batch_loss = loss.item()
            epoch_loss += batch_loss
            global_step += 1

            # Log to console every log_interval batches
            if batch_idx % log_interval == 0 or batch_idx == len(train_loader):
                print(f"[Epoch {epoch+1}/{num_epochs}] "
                      f"Batch {batch_idx}/{len(train_loader)} - "
                      f"Loss: {batch_loss:.4f} - "
                      f"Acc: {batch_acc:.2f}% - "
                      f"Time: {batch_time:.3f}s")
            # Write per-batch loss, accuracy, and time to TensorBoard
            writer.add_scalar("Train/Batch_Loss", batch_loss, global_step)
            writer.add_scalar("Train/Batch_Accuracy", batch_acc, global_step)
            writer.add_scalar("Train/Batch_Time", batch_time, global_step)
            warn(f"[Epoch {epoch+1}/{num_epochs}] "
                      f"Batch {batch_idx}/{len(train_loader)} - "
                      f"Loss: {batch_loss:.4f} - "
                      f"Acc: {batch_acc:.2f}% - "
                      f"Time: {batch_time:.3f}s")

        avg_epoch_loss = epoch_loss / len(train_loader)
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1} completed in {epoch_time:.2f}s - "
              f"Average Loss: {avg_epoch_loss:.4f}")

        # Log the average training loss per epoch
        writer.add_scalar("Train/Epoch_Avg_Loss", avg_epoch_loss, epoch + 1)

        # Evaluate on test set after each epoch
        evaluate_model(model, test_loader, criterion, device, writer, epoch + 1)

        # Compute and log submission score
        orig_acc, hashed_acc, score = compute_score(original_model, model, test_loader, device)
        print(f"[Epoch {epoch+1}/{num_epochs}] Submission Score: {score:.4f}")
        writer.add_scalar("Score/Submission_Score", score, epoch + 1)
        
        # Write submission score to log file
        log_dir = Path(os.path.dirname(writer.log_dir))
        log_file = log_dir / "accuracy_log.txt"
        with open(log_file, "a") as f:
            f.write(f"Epoch {epoch+1}: Score = {score:.4f}\n")

        # Save checkpoint on first epoch or if submission score improves
        if epoch == 0 or score > best_score:
            best_score = score
            checkpoint_path = output_dir / f"{model_name}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"[Epoch {epoch+1}] New best model. Saved checkpoint to {checkpoint_path}")

def evaluate_model(model: torch.nn.Module,
                   test_loader: torch.utils.data.DataLoader,
                   criterion: torch.nn.Module,
                   device: torch.device,
                   writer: SummaryWriter,
                   epoch: int) -> float:
    """
    Evaluate the model on the test set, logging loss and accuracy to both console and TensorBoard.
    Returns the accuracy as a percentage.
    """
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_test_loss = test_loss / len(test_loader)
    accuracy = 100.0 * correct / total

    print(f"Validation at Epoch {epoch}: "
          f"Loss: {avg_test_loss:.4f}, Accuracy: {accuracy:.2f}%")

    # Log validation loss and accuracy to TensorBoard
    writer.add_scalar("Validation/Loss", avg_test_loss, epoch)
    writer.add_scalar("Test/Loss", avg_test_loss, epoch)
    writer.add_scalar("Validation/Accuracy", accuracy, epoch)
    
    # Write accuracy to log file
    log_dir = Path(os.path.dirname(writer.log_dir))
    log_file = log_dir / "accuracy_log.txt"
    with open(log_file, "a") as f:
        f.write(f"Epoch {epoch}: Accuracy = {accuracy:.2f}%\n")

    return accuracy