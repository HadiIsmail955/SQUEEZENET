import os
import json
import torch
from tqdm import tqdm
from torchinfo import summary
from models.squeezenet import SqueezeNet
from utils.topk_accuracy import topk_accuracy
from utils.checkpoint import save_checkpoint
from contextlib import redirect_stdout

def train_model(data_module_instance, device, experiment_DIR, num_classes=200, num_epochs=32, learning_rate=4e-2,shortcut=None):
    print("Starting training process...")
    CHECKPOINT_DIR = f"{experiment_DIR}/checkpoints"
    LOG_DIR = f"{experiment_DIR}/logs"
    os.makedirs(CHECKPOINT_DIR, exist_ok=False)
    os.makedirs(LOG_DIR, exist_ok=False)
    print(f"Created experiment directories: {CHECKPOINT_DIR}, {LOG_DIR}")

    train_loader = data_module_instance.train_dataloader()
    val_loader = data_module_instance.val_dataloader()

    model = SqueezeNet(num_classes=num_classes,shortcut=shortcut).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    log_file_path = os.path.join(LOG_DIR, "model_log.txt")
    with open(log_file_path, "w",encoding="utf-8") as f:
        f.write(f"Experiment logs for SqueezeNet\n")
        f.write(f"Num classes: {num_classes}, Num epochs: {num_epochs}, LR: {learning_rate}\n")
        f.write(f"Model architecture:\n{model}\n\n")
        with redirect_stdout(f):
            summary(model, input_size=(1,3,224,224), verbose=1)

    best_val_acc = 0.0
    train_loss_history = []
    top1_train_history = []
    top5_train_history = []
    top1_val_history = []
    top5_val_history = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        total=0
        top1_train_acc=0.0
        top5_train_acc=0.0

        loop= tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}",leave=False)

        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            top1_train_acc += topk_accuracy(outputs, labels, k=1) * labels.size(0)
            top5_train_acc += topk_accuracy(outputs, labels, k=5) * labels.size(0)
            total += labels.size(0)
            loop.set_postfix(loss=loss.item())

        top1_train_acc /= total
        top5_train_acc /= total
        scheduler.step()
        epoch_loss = running_loss / len(train_loader.dataset)
        train_loss_history.append(epoch_loss)

        model.eval()
        with torch.no_grad():
            total=0
            top1_val_acc=0.0
            top5_val_acc=0.0

            val_loop = tqdm(val_loader, desc=f"Validating {epoch+1}/{num_epochs}", leave=False)

            for images, labels in val_loop:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                top1_val_acc += topk_accuracy(outputs, labels, k=1) * labels.size(0)
                top5_val_acc += topk_accuracy(outputs, labels, k=5) * labels.size(0)
                total += labels.size(0)
                val_loop.set_postfix({
                    "top1": f"{top1_val_acc/total:.4f}",
                    "top5": f"{top5_val_acc/total:.4f}"
                })
            top1_val_acc /= total
            top5_val_acc /= total

            top1_train_history.append(top1_train_acc)
            top5_train_history.append(top5_train_acc)
            top1_val_history.append(top1_val_acc)
            top5_val_history.append(top5_val_acc)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Top-1 Train Acc: {top1_train_acc:.4f}, Top-5 Train Acc: {top5_train_acc:.4f}, Top-1 Val Acc: {top1_val_acc:.4f}, Top-5 Val Acc: {top5_val_acc:.4f}")

        if top1_val_acc > best_val_acc:
            save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    checkpoint_dir=CHECKPOINT_DIR,
                    filename=f"epoch_{epoch}.pth",
                    best_val_acc=best_val_acc,
                    extra_info={"train_loss": epoch_loss, "top1_val_acc": top1_val_acc, "top5_val_acc": top5_val_acc}
                )
            
    history_path = os.path.join(LOG_DIR, "training_history.json")
    with open(history_path, "w") as f:
        json.dump({
            "train_loss": train_loss_history,
            "top1_train": top1_train_history,
            "top5_train": top5_train_history,
            "top1_val": top1_val_history,
            "top5_val": top5_val_history
        }, f, indent=4)


