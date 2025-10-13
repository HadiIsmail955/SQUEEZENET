import scripts.download_tinyimagenet as download_script
import scripts.data_module 
import torchvision
import matplotlib.pyplot as plt
def main():
    show_images=True
    print("Starting main...")
    # check if data set is available, if not download it
    dataset_Path=download_script.download_and_extract_tiny_imagenet()
    print(f"Dataset is available at: {dataset_Path}")
    data_module_instance = scripts.data_module.DataModule(data_dir=dataset_Path, batch_size=32, image_size=64, num_workers=4)
    print("Setting up data module...")
    data_module_instance.setup()
    print("Data module is set up.")
    train_loader = data_module_instance.train_dataloader()
    val_loader = data_module_instance.val_dataloader()
    test_loader = data_module_instance.test_dataloader()
    for images, labels in train_loader:
        print(f"Batch of images shape: {images.shape}")
        print(f"Batch of labels shape: {labels.shape}")
        if show_images:    
            img_grid = torchvision.utils.make_grid(images[:8], nrow=4, normalize=True)
            plt.figure(figsize=(8, 8))
            plt.imshow(img_grid.permute(1, 2, 0))
            plt.title("Training Samples")
            plt.axis("off")
            plt.show()
        break
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    print(f"Number of test batches: {len(test_loader)}")





if __name__ == '__main__':
    main()