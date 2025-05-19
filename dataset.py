# COCO Dataset Preparation Script
import os
import sys
import subprocess
import zipfile
import shutil
import json
import random
import numpy as np
from PIL import Image
import torch
import requests
from tqdm import tqdm

# COCO 2017 Dataset URLs
COCO_TRAIN_IMAGES_URL = "http://images.cocodataset.org/zips/train2017.zip"
COCO_VAL_IMAGES_URL = "http://images.cocodataset.org/zips/val2017.zip"
COCO_ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

# Create a download function with progress bar
def download_file(url, save_path):
    """Download a file from a URL with progress bar."""
    print(f"Downloading {url} to {save_path}")
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
    
    with open(save_path, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    
    progress_bar.close()
    
    if total_size != 0 and progress_bar.n != total_size:
        print("ERROR, something went wrong with the download")
        return False
    return True

def download_coco_dataset():
    """Download the COCO 2017 dataset."""
    print("Starting download of COCO 2017 dataset...")
    
    # Create directories
    coco_dir = "./coco_dataset"
    os.makedirs(coco_dir, exist_ok=True)
    os.makedirs(f"{coco_dir}/images", exist_ok=True)
    os.makedirs(f"{coco_dir}/annotations", exist_ok=True)
    
    # Download annotations
    annotations_zip = f"{coco_dir}/annotations_trainval2017.zip"
    if not os.path.exists(annotations_zip):
        print("Downloading annotations...")
        download_file(COCO_ANNOTATIONS_URL, annotations_zip)
    else:
        print("Annotations zip already exists, skipping download")
    
    # Extract annotations
    if not os.path.exists(f"{coco_dir}/annotations/captions_train2017.json"):
        print("Extracting annotations...")
        with zipfile.ZipFile(annotations_zip, 'r') as zip_ref:
            zip_ref.extractall(coco_dir)
        print("Annotations extracted")
    else:
        print("Annotations already extracted")
    
    # Download and extract train images
    train_zip = f"{coco_dir}/train2017.zip"
    if not os.path.exists(f"{coco_dir}/images/train2017") or len(os.listdir(f"{coco_dir}/images/train2017")) < 100:
        if not os.path.exists(train_zip):
            print("Downloading training images (large file, ~18GB)...")
            if not download_file(COCO_TRAIN_IMAGES_URL, train_zip):
                print("Failed to download training images")
                return False
        
        print("Extracting training images...")
        os.makedirs(f"{coco_dir}/images/train2017", exist_ok=True)
        with zipfile.ZipFile(train_zip, 'r') as zip_ref:
            zip_ref.extractall(f"{coco_dir}/images")
        print("Training images extracted")
        
        # Remove zip to save space
        if os.path.exists(train_zip):
            os.remove(train_zip)
            print("Removed training images zip file to save space")
    else:
        print("Training images already extracted")
    
    # Download and extract validation images
    val_zip = f"{coco_dir}/val2017.zip"
    if not os.path.exists(f"{coco_dir}/images/val2017") or len(os.listdir(f"{coco_dir}/images/val2017")) < 100:
        if not os.path.exists(val_zip):
            print("Downloading validation images (~1GB)...")
            if not download_file(COCO_VAL_IMAGES_URL, val_zip):
                print("Failed to download validation images")
                return False
        
        print("Extracting validation images...")
        os.makedirs(f"{coco_dir}/images/val2017", exist_ok=True)
        with zipfile.ZipFile(val_zip, 'r') as zip_ref:
            zip_ref.extractall(f"{coco_dir}/images")
        print("Validation images extracted")
        
        # Remove zip to save space
        if os.path.exists(val_zip):
            os.remove(val_zip)
            print("Removed validation images zip file to save space")
    else:
        print("Validation images already extracted")
    
    return True

def process_coco_annotations():
    """Process COCO annotations into the format expected by our model, preserving multiple captions per image."""
    coco_dir = "./coco_dataset"
    
    # Process training captions
    train_captions_path = f"{coco_dir}/annotations/captions_train2017.json"
    processed_train_path = f"{coco_dir}/processed_train_captions.json"
    
    if os.path.exists(train_captions_path):
        print("Processing training captions...")
        with open(train_captions_path, 'r') as f:
            train_data = json.load(f)
        
        # Create image id to filename mapping
        train_id_to_filename = {}
        for img in train_data['images']:
            train_id_to_filename[img['id']] = f"train2017/{img['file_name']}"
        
        # Group captions by image_id - industry standard approach
        train_images = {}
        for ann in train_data['annotations']:
            image_id = ann['image_id']
            if image_id in train_id_to_filename:  # Only process if we have the image
                if image_id not in train_images:
                    train_images[image_id] = {
                        "image_id": image_id,
                        "image_path": f"{coco_dir}/images/{train_id_to_filename[image_id]}",
                        "captions": []
                    }
                train_images[image_id]["captions"].append(ann['caption'])
        
        # Convert to list for saving
        train_processed = list(train_images.values())
        
        # Save processed annotations
        with open(processed_train_path, 'w') as f:
            json.dump(train_processed, f)
        print(f"Processed {len(train_processed)} training images with multiple captions")
        
        # Calculate average captions per image
        total_captions = sum(len(item["captions"]) for item in train_processed)
        avg_captions = total_captions / len(train_processed) if train_processed else 0
        print(f"Average of {avg_captions:.2f} captions per image")
    
    # Process validation captions
    val_captions_path = f"{coco_dir}/annotations/captions_val2017.json"
    processed_val_path = f"{coco_dir}/processed_val_captions.json"
    
    if os.path.exists(val_captions_path):
        print("Processing validation captions...")
        with open(val_captions_path, 'r') as f:
            val_data = json.load(f)
        
        # Create image id to filename mapping
        val_id_to_filename = {}
        for img in val_data['images']:
            val_id_to_filename[img['id']] = f"val2017/{img['file_name']}"
        
        # Group captions by image_id
        val_images = {}
        for ann in val_data['annotations']:
            image_id = ann['image_id']
            if image_id in val_id_to_filename:  # Only process if we have the image
                if image_id not in val_images:
                    val_images[image_id] = {
                        "image_id": image_id,
                        "image_path": f"{coco_dir}/images/{val_id_to_filename[image_id]}",
                        "captions": []
                    }
                val_images[image_id]["captions"].append(ann['caption'])
        
        # Convert to list for saving
        val_processed = list(val_images.values())
        
        # Save processed annotations
        with open(processed_val_path, 'w') as f:
            json.dump(val_processed, f)
        print(f"Processed {len(val_processed)} validation images with multiple captions")
        
        # Calculate average captions per image
        total_captions = sum(len(item["captions"]) for item in val_processed)
        avg_captions = total_captions / len(val_processed) if val_processed else 0
        print(f"Average of {avg_captions:.2f} captions per image")
    
    # Return paths to processed files
    return {
        "train_images_dir": f"{coco_dir}/images/train2017",
        "train_captions_file": processed_train_path,
        "val_images_dir": f"{coco_dir}/images/val2017",
        "val_captions_file": processed_val_path,
        "test_images_dir": f"{coco_dir}/images/val2017",  # Use val as test
        "test_captions_file": processed_val_path
    }

def filter_captions(data):
    """Filter out suspicious captions from a dataset.
    
    Args:
        data (list): List of image-caption pairs where each item has a "captions" key
                    with a list of caption strings.
                    
    Returns:
        list: Filtered dataset with suspicious captions removed
    """
    print(f"Filtering captions...")
    
    # Define suspicious token list for filtering
    suspicious_tokens = [
        "struct", "connector", "att", "class=", "target=", "rel=", 
        "<", ">", "[", "]", "gui", "#", "function", "return", "//",
        "import", "def ", "var ", "let ", "const ", "public ", "private "
    ]
    
    filtered_data = []
    total_captions_before = 0
    total_filtered = 0
    images_processed = 0
    
    for item in data:
        images_processed += 1
        
        # Skip if image has no captions
        if "captions" not in item or not item["captions"]:
            continue
            
        total_captions_before += len(item["captions"])
        
        # Apply basic quality filtering:
        # 1. Remove empty captions
        # 2. Filter out captions with suspicious tokens (likely code or markup)
        # 3. Filter out captions that are too short or too long
        filtered_captions = []
        for caption in item["captions"]:
            # Skip empty captions
            if not caption or not caption.strip():
                total_filtered += 1
                continue
                
            # Skip captions with suspicious tokens
            if any(token in caption.lower() for token in suspicious_tokens):
                total_filtered += 1
                continue
                
            # Skip captions that are too short or too long
            words = caption.split()
            if len(words) < 5 or len(words) > 25:
                total_filtered += 1
                continue
                
            # Passed all filters
            filtered_captions.append(caption)
        
        # Only include images with at least one valid caption
        if filtered_captions:
            new_item = item.copy()
            new_item["captions"] = filtered_captions
            filtered_data.append(new_item)
    
    print(f"Images processed: {images_processed}")
    print(f"Images after filtering: {len(filtered_data)}")
    print(f"Total captions before filtering: {total_captions_before}")
    print(f"Filtered out: {total_filtered} captions")
    
    # Calculate total captions
    total_captions = sum(len(item["captions"]) for item in filtered_data)
    print(f"Total captions after filtering: {total_captions}")
    print(f"Average captions per image: {total_captions/len(filtered_data):.2f}" if filtered_data else "No valid images remain")
    
    return filtered_data

# Keep the existing filter_captions function but rename it to filter_captions_file
def filter_captions_file(input_file, output_file, max_samples=None):
    """Filter out suspicious captions and optionally limit dataset size.
    Preserves all other captions for each image."""
    print(f"Processing captions from {input_file}...")
    
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    print(f"Total images before processing: {len(data)}")
    
    # Use the new filter_captions function for consistency
    filtered_data = filter_captions(data)
    
    # Limit dataset size if requested - sample by image, not by caption
    if max_samples and max_samples < len(filtered_data):
        random.seed(42)  # For reproducibility
        filtered_data = random.sample(filtered_data, max_samples)
        print(f"Limited to {max_samples} images")
        
        # Recalculate total captions
        total_captions = sum(len(item["captions"]) for item in filtered_data)
        print(f"Total captions after sampling: {total_captions}")
    
    # Save filtered data
    with open(output_file, 'w') as f:
        json.dump(filtered_data, f)
    
    print(f"Saved processed data to {output_file}")
    return output_file

def create_data_splits(processed_file, output_dir="./data_splits", test_size=5000, val_size=5000):
    """Create train/val/test splits from processed captions, preserving multiple captions per image."""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Creating data splits from {processed_file}...")
    
    with open(processed_file, 'r') as f:
        data = json.load(f)
    
    # Shuffle data with fixed seed for reproducibility
    random.seed(42)
    random.shuffle(data)
    
    # Create splits
    test_data = data[:test_size]
    val_data = data[test_size:test_size+val_size]
    train_data = data[test_size+val_size:]
    
    # Save splits
    train_path = f"{output_dir}/train_captions.json"
    val_path = f"{output_dir}/val_captions.json"
    test_path = f"{output_dir}/test_captions.json"
    
    with open(train_path, 'w') as f:
        json.dump(train_data, f)
    
    with open(val_path, 'w') as f:
        json.dump(val_data, f)
    
    with open(test_path, 'w') as f:
        json.dump(test_data, f)
    
    # Calculate statistics for each split
    train_images = len(train_data)
    train_captions = sum(len(item["captions"]) for item in train_data)
    
    val_images = len(val_data)
    val_captions = sum(len(item["captions"]) for item in val_data)
    
    test_images = len(test_data)
    test_captions = sum(len(item["captions"]) for item in test_data)
    
    print(f"Created splits:")
    print(f"  - Training: {train_images} images, {train_captions} captions ({train_captions/train_images:.2f} per image)")
    print(f"  - Validation: {val_images} images, {val_captions} captions ({val_captions/val_images:.2f} per image)")
    print(f"  - Testing: {test_images} images, {test_captions} captions ({test_captions/test_images:.2f} per image)")
    
    # Return paths
    return {
        "train_captions_file": train_path,
        "val_captions_file": val_path,
        "test_captions_file": test_path
    }

def setup_coco_dataset():
    """Main function to download and set up the COCO dataset."""
    print("===== COCO 2017 Dataset Setup =====")
    
    # Download the dataset
    if not download_coco_dataset():
        print("Download failed. Exiting.")
        return False
    
    # Process annotations
    processed_paths = process_coco_annotations()
    
    # Filter captions and create a more manageable subset for training
    # Limiting to ~100K training samples for efficiency
    filtered_train = filter_captions_file(
        processed_paths["train_captions_file"], 
        "./coco_dataset/filtered_train_captions.json",
        max_samples=100000  # Limit to 100K images for faster training
    )
    
    filtered_val = filter_captions_file(
        processed_paths["val_captions_file"], 
        "./coco_dataset/filtered_val_captions.json"
    )
    
    # Create final splits
    split_paths = create_data_splits(
        filtered_train,
        output_dir="./coco_dataset/splits",
        test_size=5000,
        val_size=5000
    )
    
    # Create final dataset_paths.json with all paths
    dataset_config = {
        "train_images_dir": processed_paths["train_images_dir"],
        "train_captions_file": split_paths["train_captions_file"],
        "val_images_dir": processed_paths["val_images_dir"],
        "val_captions_file": split_paths["val_captions_file"],
        "test_images_dir": processed_paths["test_images_dir"],
        "test_captions_file": split_paths["test_captions_file"]
    }
    
    # Save paths for the model training
    with open("dataset_paths.json", "w") as f:
        json.dump(dataset_config, f, indent=2)
    
    print("\n✅ COCO dataset setup complete!")
    print("Paths saved to dataset_paths.json")
    
    # Print some stats
    print("\nDataset Statistics:")
    for split in ["train", "val", "test"]:
        with open(dataset_config[f"{split}_captions_file"], 'r') as f:
            data = json.load(f)
        print(f"  - {split.capitalize()} set: {len(data)} images, {sum(len(item['captions']) for item in data)} captions")
    
    return True

if __name__ == "__main__":
    setup_coco_dataset()