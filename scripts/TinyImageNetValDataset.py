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

        # If provided, reuse the training mapping
        if class_to_idx is not None:
            self.class_to_idx = class_to_idx
        else:
            self.class_to_idx = {}
            current_idx = 0

        with open(annotations_path, 'r') as f:
            for line in f:
                img_name, class_name = line.strip().split('\t')[:2]
                if class_name not in self.class_to_idx:
                    # Only assign new IDs if not in existing mapping
                    self.class_to_idx[class_name] = len(self.class_to_idx)
                label = self.class_to_idx[class_name]
                self.imgs.append((img_name, label))
