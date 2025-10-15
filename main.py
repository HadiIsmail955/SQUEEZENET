import os
import datetime
import scripts.download_tinyimagenet as download_script
import scripts.data_module 
import torch
import torchvision
import matplotlib.pyplot as plt

DATA_DIR = "./data/imagenet10_split" #"./data/tiny-imagenet-200"
BATCH_SIZE = 120
IMAGE_SIZE = 224 #224 or 64
NUM_WORKERS = 4
NUM_CLASSES = 10
NUM_channels = 3
NUM_EPOCHS = 30
LEARNING_RATE = 4e-2 #4e-2
TRAIN=True
SHOW_IMAGES=False
SHORTCUT="complex"  # None, "simple", "complex"

def main():
    experiment_DIR=f"./experiments/runs/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}"
    print("Starting main...")
    # check if data set is available, if not download it
    # DATA_DIR=download_script.download_and_extract_tiny_imagenet()
    print(f"Dataset is available at: {DATA_DIR}")
    data_module_instance = scripts.data_module.DataModule(data_dir=DATA_DIR, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE, num_workers=NUM_WORKERS)
    print("Setting up data module...")
    data_module_instance.setup()
    print("Data module is set up.")
    # train_loader = data_module_instance.train_dataloader()
    # for images, labels in train_loader:
    #     NUM_channels = images.shape[1]
    #     if SHOW_IMAGES:    
    #         img_grid = torchvision.utils.make_grid(images[:8], nrow=4, normalize=True)
    #         plt.figure(figsize=(8, 8))
    #         plt.imshow(img_grid.permute(1, 2, 0))
    #         plt.title("Training Samples")
    #         plt.axis("off")
    #         plt.show()
    #     break
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nExperiment imformation:")
    print(f"Using device: {device}")
    print(f"  Data directory: {DATA_DIR}")
    print(f"  Experiment directory: {experiment_DIR}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Image size: {IMAGE_SIZE}")
    print(f"  Number of channels: {NUM_channels}")
    print(f"  Number of workers: {NUM_WORKERS}")
    print(f"  Number of classes: {NUM_CLASSES}")
    print(f"  Number of epochs: {NUM_EPOCHS}")
    print(f"  Learning rate: {LEARNING_RATE}\n")
    print("Finshed experiment setup.")

    if TRAIN:
        from experiments.train import train_model
        train_model(data_module_instance, device, experiment_DIR, num_classes=NUM_CLASSES, num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE,shortcut=SHORTCUT)
    

if __name__ == '__main__':
    main()