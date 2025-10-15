import os
from PIL import Image
from tqdm import tqdm
from torchvision import datasets
from torch.utils.data import DataLoader
from scripts.transforms import get_train_transforms, get_val_transforms
from scripts.TinyImageNetValDataset import TinyImageNetValDataset

class DataModule:
    def __init__(self, data_dir="./data/tiny-imagenet-200", batch_size=32, image_size=224, num_workers=4):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_workers = num_workers

        self.train_transforms = get_train_transforms(image_size)
        self.val_transforms = get_val_transforms(image_size)

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self):
        train_dir = os.path.join(self.data_dir, "train")
        val_dir = os.path.join(self.data_dir, "val")

        self.train_dataset = datasets.ImageFolder(root=train_dir, transform=self.train_transforms)
        # train_class_to_idx = self.train_dataset.class_to_idx

        # self.val_dataset = TinyImageNetValDataset(val_dir, transform=self.val_transforms, class_to_idx=train_class_to_idx)
        self.val_dataset = datasets.ImageFolder(root=val_dir, transform=self.val_transforms)
        

    def setupTest(self):
        test_dir = os.path.join(self.data_dir, "test")
        self.test_dataset = datasets.ImageFolder(root=test_dir, transform=self.val_transforms)

    def train_dataloader(self):
        # verify_images(self.train_dataset)
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
    
    def val_dataloader(self):
        # verify_images(self.val_dataset)
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

def verify_images(dataset):
    bad_images = []
    for idx in tqdm(range(len(dataset))):
        path, _ = dataset.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"Bad image at {path}: {e}")
            bad_images.append(path)
    print(f"\nFound {len(bad_images)} bad images.")
    return bad_images
