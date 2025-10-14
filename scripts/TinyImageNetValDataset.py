from torch.utils.data import Dataset
from PIL import Image
import os

class TinyImageNetValDataset(Dataset):
    def __init__(self, val_dir, transform=None, class_to_idx=None):
        self.val_dir = val_dir
        self.transform = transform
        self.img_dir = os.path.join(val_dir, "images")
        annotations_path = os.path.join(val_dir, "val_annotations.txt")

        self.imgs = []
        self.class_to_idx = {}
        self.idx_to_class = {}

        # If training class_to_idx mapping is provided, use it as the ground truth
        if class_to_idx is not None:
            self.class_to_idx = class_to_idx
        else:
            # Create a new mapping only if none provided
            with open(annotations_path, 'r') as f:
                for line in f:
                    _, class_name = line.strip().split('\t')[:2]
                    if class_name not in self.class_to_idx:
                        self.class_to_idx[class_name] = len(self.class_to_idx)

        # Build (image, label) pairs
        with open(annotations_path, 'r') as f:
            for line in f:
                img_name, class_name = line.strip().split('\t')[:2]
                if class_name not in self.class_to_idx:
                    raise ValueError(f"Class {class_name} not found in provided class_to_idx mapping.")
                label = self.class_to_idx[class_name]
                self.imgs.append((img_name, label))

        # Build reverse lookup (optional)
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_name, label = self.imgs[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
