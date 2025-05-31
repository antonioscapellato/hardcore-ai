import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

def get_writer(base_dir: str, model_name: str) -> SummaryWriter:
    """
    Creates a TensorBoard SummaryWriter with a timestamped subdirectory under base_dir/logs/model_name.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(base_dir, "logs", f"{model_name}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    return writer