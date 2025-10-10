import os
from utils.download_with_progress import download_with_progress
from utils.extract_zip_with_progress import extract_zip_with_progress

def download_and_extract_tiny_imagenet(data_dir="./data"):
    os.makedirs(data_dir, exist_ok=True)
    url="http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    zip_path = os.path.join(data_dir, "tiny-imagenet-200.zip")
    target_path = os.path.join(data_dir, "tiny-imagenet-200")
    if os.path.exists(target_path):
        print("Tiny ImageNet data already exists.")
        return target_path
    print("Downloading Tiny ImageNet data...")
    download_with_progress(url, zip_path)
    print("Extracting Tiny ImageNet data...")
    extract_zip_with_progress(zip_path, data_dir)
    os.remove(zip_path)
    print("Tiny ImageNet data downloaded and extracted.")
    return target_path