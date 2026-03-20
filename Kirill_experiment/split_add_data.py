"""
Script to split add_data into two folds with original images and random crops.
"""

import os
import random
import shutil
from pathlib import Path
from PIL import Image


DATASET_FOLDS_DIR = Path('dataset_folds')


def parse_yolo_label(line: str) -> tuple:
    """Parse a YOLO label line."""
    parts = list(map(float, line.strip().split()))
    return int(parts[0]), parts[1], parts[2], parts[3], parts[4]


def format_yolo_label(class_id: int, x_center: float, y_center: float, width: float, height: float) -> str:
    """Format a YOLO label line"""
    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def crop_bounding_box(x_center: float, y_center: float, width: float, height: float,
                      crop_x: float, crop_y: float, crop_w: float, crop_h: float,
                      img_w: int, img_h: int) -> tuple | None:
    """Recalculate bounding box coordinates for a cropped image."""
    # Convert normalized coords to absolute pixels (original image)
    x_min = (x_center - width / 2) * img_w
    x_max = (x_center + width / 2) * img_w
    y_min = (y_center - height / 2) * img_h
    y_max = (y_center + height / 2) * img_h
    
    # Crop boundaries in absolute pixels
    crop_x_abs = crop_x * img_w
    crop_y_abs = crop_y * img_h
    crop_w_abs = crop_w * img_w
    crop_h_abs = crop_h * img_h
    crop_x_max = crop_x_abs + crop_w_abs
    crop_y_max = crop_y_abs + crop_h_abs
    
    # Intersect bbox with crop area
    new_x_min = max(x_min, crop_x_abs)
    new_x_max = min(x_max, crop_x_max)
    new_y_min = max(y_min, crop_y_abs)
    new_y_max = min(y_max, crop_y_max)
    
    # Check if bbox is completely outside crop
    if new_x_min >= new_x_max or new_y_min >= new_y_max:
        return None
    
    # Convert back to normalized coordinates relative to the crop
    new_x_center = ((new_x_min + new_x_max) / 2 - crop_x_abs) / crop_w_abs
    new_y_center = ((new_y_min + new_y_max) / 2 - crop_y_abs) / crop_h_abs
    new_width = (new_x_max - new_x_min) / crop_w_abs
    new_height = (new_y_max - new_y_min) / crop_h_abs
    
    # Clamp values to valid range
    new_x_center = max(0, min(1, new_x_center))
    new_y_center = max(0, min(1, new_y_center))
    new_width = max(0, min(1, new_width))
    new_height = max(0, min(1, new_height))
    
    return new_x_center, new_y_center, new_width, new_height


def create_random_crop(img_path: Path, crop_size_ratio: tuple = (0.5, 0.7)) -> tuple:
    """Create a random crop of an image."""
    img = Image.open(img_path)
    img_w, img_h = img.size
    
    # Random crop size
    crop_ratio = random.uniform(*crop_size_ratio)
    crop_w_abs = int(img_w * crop_ratio)
    crop_h_abs = int(img_h * crop_ratio)
    
    # Random crop position
    max_x = img_w - crop_w_abs
    max_y = img_h - crop_h_abs
    crop_x_abs = random.randint(0, max_x) if max_x > 0 else 0
    crop_y_abs = random.randint(0, max_y) if max_y > 0 else 0
    
    # Create crop
    crop_img = img.crop((crop_x_abs, crop_y_abs, crop_x_abs + crop_w_abs, crop_y_abs + crop_h_abs))
    
    # Normalized crop coordinates
    crop_x = crop_x_abs / img_w
    crop_y = crop_y_abs / img_h
    crop_w = crop_w_abs / img_w
    crop_h = crop_h_abs / img_h
    
    return crop_img, crop_x, crop_y, crop_w, crop_h


def process_labels(label_path: Path, crop_x: float, crop_y: float, crop_w: float, crop_h: float,
                   img_w: int, img_h: int) -> list:
    """Process labels for a cropped image."""
    if not label_path.exists():
        return []
    
    new_labels = []
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            class_id, x_center, y_center, width, height = parse_yolo_label(line)
            
            # Recalculate bbox for crop
            result = crop_bounding_box(x_center, y_center, width, height,
                                       crop_x, crop_y, crop_w, crop_h, img_w, img_h)
            
            if result is not None:
                new_x_center, new_y_center, new_width, new_height = result
                new_labels.append(format_yolo_label(class_id, new_x_center, new_y_center, new_width, new_height))
    
    return new_labels


def copy_validation_data(fold_idx: int, output_dir: Path):
    """Copy validation data from dataset_folds/fold_{fold_idx}/val/ to output fold."""
    # Source validation directories
    src_val_images = DATASET_FOLDS_DIR / f'fold_{fold_idx}' / 'images' / 'val'
    src_val_labels = DATASET_FOLDS_DIR / f'fold_{fold_idx}' / 'labels' / 'val'
    
    # Destination directories
    dst_val_images = output_dir / 'val' / 'images'
    dst_val_labels = output_dir / 'val' / 'labels'
    
    # Create destination directories
    dst_val_images.mkdir(parents=True, exist_ok=True)
    dst_val_labels.mkdir(parents=True, exist_ok=True)
    
    # Copy validation images and labels
    if not src_val_images.exists():
        print(f"  Warning: Validation images not found: {src_val_images}")
        return 0
    
    val_images = list(src_val_images.iterdir())
    for img_path in val_images:
        shutil.copy(img_path, dst_val_images / img_path.name)
        
        # Copy corresponding label
        label_name = img_path.stem + '.txt'
        src_label = src_val_labels / label_name
        if src_label.exists():
            shutil.copy(src_label, dst_val_labels / label_name)
    
    print(f"  Copied {len(val_images)} validation images from dataset_folds/fold_{fold_idx}/val/")
    return len(val_images)


def split_and_create_folds(add_data_dir: Path, output_dir: Path, seed: int = 42):
    """Split add_data into two folds with original images and random crops."""
    random.seed(seed)

    images_dir = add_data_dir / 'add_images'
    labels_dir = add_data_dir / 'add_labels'

    # Get all image files
    image_files = sorted([f for f in images_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])

    print(f"Found {len(image_files)} images")

    # Shuffle and split into two folds
    random.shuffle(image_files)
    mid = len(image_files) // 2
    fold1_images = image_files[:mid]
    fold2_images = image_files[mid:]

    print(f"Fold 1: {len(fold1_images)} images")
    print(f"Fold 2: {len(fold2_images)} images")

    # Create output directories with train/val structure
    for fold_name in ['fold1', 'fold2']:
        (output_dir / fold_name / 'train' / 'images').mkdir(parents=True, exist_ok=True)
        (output_dir / fold_name / 'train' / 'labels').mkdir(parents=True, exist_ok=True)
        (output_dir / fold_name / 'val' / 'images').mkdir(parents=True, exist_ok=True)
        (output_dir / fold_name / 'val' / 'labels').mkdir(parents=True, exist_ok=True)

    # Process each fold
    for fold_idx, (fold_name, fold_images) in enumerate([('fold1', fold1_images), ('fold2', fold2_images)], start=1):
        print(f"\nProcessing {fold_name}...")
        
        train_images_dir = output_dir / fold_name / 'train' / 'images'
        train_labels_dir = output_dir / fold_name / 'train' / 'labels'

        for img_path in fold_images:
            # Get corresponding label file
            label_name = img_path.stem + '.txt'
            label_path = labels_dir / label_name

            # Get image dimensions
            img = Image.open(img_path)
            img_w, img_h = img.size

            # Copy original image to train
            shutil.copy(img_path, train_images_dir / img_path.name)

            # Copy original labels to train
            if label_path.exists():
                shutil.copy(label_path, train_labels_dir / label_name)

            # Create random crop
            crop_img, crop_x, crop_y, crop_w, crop_h = create_random_crop(img_path)

            # Save cropped image to train
            crop_name = f"{img_path.stem}_crop{img_path.suffix}"
            crop_img.save(train_images_dir / crop_name)

            # Process and save cropped labels to train
            crop_labels = process_labels(label_path, crop_x, crop_y, crop_w, crop_h, img_w, img_h)
            crop_label_name = f"{img_path.stem}_crop.txt"

            with open(train_labels_dir / crop_label_name, 'w') as f:
                for label_line in crop_labels:
                    f.write(label_line + '\n')

        print(f"  Saved {len(fold_images)} original + {len(fold_images)} cropped images to train/")
        
        # Copy validation data from dataset_folds
        val_count = copy_validation_data(fold_idx, output_dir / fold_name)

    print(f"\nDone! Output saved to: {output_dir}")
    print(f"Total images per fold: {len(fold1_images) * 2} train (original + crop) + val from dataset_folds")
    print(f"Total labels per fold: {len(fold1_images) * 2} train (original + crop) + val from dataset_folds")


if __name__ == '__main__':
    # Paths
    add_data_dir = Path(__file__).parent / 'add_data'
    output_dir = Path(__file__).parent / 'add_data_folds'
    
    # Run split
    split_and_create_folds(add_data_dir, output_dir, seed=42)
