import torch
import os

def save_checkpoint(model, optimizer=None, scheduler=None, epoch=0,
                    checkpoint_dir="./checkpoints", filename="checkpoint.pth",
                    best_val_acc=None, extra_info=None):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, filename)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict()
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    if best_val_acc is not None:
        checkpoint["best_val_acc"] = best_val_acc
    if extra_info is not None:
        checkpoint.update(extra_info)

    # Save atomically
    temp_path = path + ".tmp"
    torch.save(checkpoint, temp_path)
    os.replace(temp_path, path)
    print(f"Checkpoint saved at: {path}")


def load_checkpoint(model, optimizer=None, scheduler=None, path=None, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint
