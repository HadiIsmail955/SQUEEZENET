from torchvision import transforms
from PIL import Image

def convert_to_rgb(image):
    return image.convert("RGB")

def get_train_transforms(image_size=224):
    return transforms.Compose([
        transforms.Lambda(convert_to_rgb),
        transforms.Resize((image_size, image_size)), 
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

def get_val_transforms(image_size=224):
    return transforms.Compose([
        transforms.Lambda(convert_to_rgb),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
