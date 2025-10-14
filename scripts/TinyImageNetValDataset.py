from torch.utils.data import Dataset
from PIL import Image
import os

class TinyImageNetValDataset(Dataset):
    def __init__(self, val_dir, transform=None):
        self.val_dir = val_dir
        self.transform = transform

        self.img_dir = os.path.join(val_dir, "images")
        annotations_path = os.path.join(val_dir, "val_annotations.txt")

        # Read the mapping from file
        self.imgs = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        current_idx = 0

        with open(annotations_path, 'r') as f:
            for line in f:
                img_name, class_name = line.strip().split('\t')[:2]
                if class_name not in self.class_to_idx:
                    self.class_to_idx[class_name] = current_idx
                    self.idx_to_class[current_idx] = class_name
                    current_idx += 1
                label = self.class_to_idx[class_name]
                self.imgs.append((img_name, label))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_name, label = self.imgs[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label
