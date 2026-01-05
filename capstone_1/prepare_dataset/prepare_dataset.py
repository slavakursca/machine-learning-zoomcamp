#!/usr/bin/env python
# coding: utf-8

import os
import shutil
import cv2
import numpy as np
from PIL import Image
import imagehash
import splitfolders

# -----------------------------
# Configuration
# -----------------------------
SOURCE_DIR = "SUNRGBD"
OUTPUT_DIR = "dataset"
SPLITTED_OUTPUT_DIR = "../splitted_dataset"

VALID_CLASSES = {
    "bedroom": "bedroom",
    "bathroom": "bathroom",
    "kitchen": "kitchen",
    "living_room": "living_room"
}

# -----------------------------
# Main Script
# -----------------------------
def main():
    
    # Step 1: Download and extract SUNRGBD (skip if already done)
    print("Step 1: Downloading and extracting SUNRGBD dataset...")
    if not os.path.exists("SUNRGBD.zip"):
        print("Downloading SUNRGBD.zip (~6.7 GB)... This may take a while.")
        os.system('wget https://rgbd.cs.princeton.edu/data/SUNRGBD.zip')
    else:
        print("SUNRGBD.zip already exists, skipping download.")

    if not os.path.exists(SOURCE_DIR):
        print("Extracting SUNRGBD.zip...")
        os.system('unzip SUNRGBD.zip -d .')
    else:
        print(f"{SOURCE_DIR} folder already exists, skipping extraction.")

    # Step 2: Extract one representative image per valid scene
    print("\nStep 2: Extracting representative images from valid scenes...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    extracted_count = 0

    for root, dirs, files in os.walk(SOURCE_DIR):
        if "scene.txt" in files:
            scene_path = os.path.join(root, "scene.txt")
            with open(scene_path, "r") as f:
                label = f.read().strip().lower()

            if label not in VALID_CLASSES:
                continue

            image_dir = os.path.join(root, "image")
            if not os.path.exists(image_dir):
                continue

            images = [img for img in os.listdir(image_dir) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if not images:
                continue

            img_path = os.path.join(image_dir, images[0])
            out_dir = os.path.join(OUTPUT_DIR, label)
            os.makedirs(out_dir, exist_ok=True)

            new_name = f"{os.path.basename(root)}.jpg"
            shutil.copy(img_path, os.path.join(out_dir, new_name))
            extracted_count += 1

    print(f"Extraction complete: {extracted_count} images saved to '{OUTPUT_DIR}'.")

    # Step 3: Remove corrupted images
    print("\nStep 3: Removing corrupted images...")
    removed_corrupted = 0
    for root, _, files in os.walk(OUTPUT_DIR):
        for f in files:
            path = os.path.join(root, f)
            try:
                img = Image.open(path)
                img.verify()
            except Exception:
                os.remove(path)
                removed_corrupted += 1
                print(f"  Removed corrupted: {path}")
    print(f"Removed {removed_corrupted} corrupted images.")

    # Step 4: Remove too dark images
    print("\nStep 4: Removing overly dark images...")
    removed_dark = 0
    def is_too_dark(img_path, threshold=25):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        return img is None or img.mean() < threshold

    for cls in os.listdir(OUTPUT_DIR):
        cls_path = os.path.join(OUTPUT_DIR, cls)
        if not os.path.isdir(cls_path):
            continue
        for img in os.listdir(cls_path):
            img_path = os.path.join(cls_path, img)
            if is_too_dark(img_path):
                os.remove(img_path)
                removed_dark += 1
                print(f"  Removed too dark: {img_path}")
    print(f"Removed {removed_dark} overly dark images.")

    # Step 5: Remove near-duplicates using perceptual hash
    print("\nStep 5: Removing near-duplicate images...")
    hashes = {}
    removed_duplicates = 0
    for cls in os.listdir(OUTPUT_DIR):
        cls_path = os.path.join(OUTPUT_DIR, cls)
        if not os.path.isdir(cls_path):
            continue
        for img in os.listdir(cls_path):
            path = os.path.join(cls_path, img)
            try:
                h = imagehash.phash(Image.open(path))
                if h in hashes:
                    os.remove(path)
                    removed_duplicates += 1
                    print(f"  Removed duplicate: {path}")
                else:
                    hashes[h] = path
            except Exception:
                os.remove(path)
                removed_duplicates += 1
    print(f"Removed {removed_duplicates} near-duplicate images.")

    # Step 6: Split into train/val/test
    print("\nStep 6: Splitting dataset into train (70%), val (20%), test (10%)...")
    splitfolders.ratio(
        OUTPUT_DIR,
        output=SPLITTED_OUTPUT_DIR,
        seed=42,
        ratio=(0.7, 0.2, 0.1),
        move=True
    )
    print(f"Dataset split complete! Final dataset is in: {SPLITTED_OUTPUT_DIR}")

    print("\nAll done! You can now use the dataset for training.")

if __name__ == "__main__":
    main()