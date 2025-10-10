import zipfile
import os
from tqdm import tqdm

def extract_zip_with_progress(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Get a list of files to extract
        members = zip_ref.infolist()
        total_files = len(members)

        print(f"📦 Extracting {total_files} files from {os.path.basename(zip_path)}...")

        for member in tqdm(members, desc="Extracting", unit="file"):
            zip_ref.extract(member, path=extract_to)

    print(f"✅ Extraction complete: {extract_to} with size {os.path.getsize(extract_to)} bytes")
