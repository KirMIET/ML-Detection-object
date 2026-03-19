import cv2
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random

# Настройки
CLASS_MAP = {0: 'Customer', 1: 'Staff'} 
OUTPUT_DIR = Path('classification_dataset')
IMG_DIR = Path(r'yolo_dataset/yolo_dataset/train/images')
LBL_DIR = Path(r'yolo_dataset/yolo_dataset/train/labels')

# Параметры для класса Other
MEAN_W, STD_W = 98.64, 89.95
MEAN_H, STD_H = 164.86, 121.07

def calculate_iou(box1, box2):
    """ box: [x1, y1, x2, y2] """
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area
    return intersection_area / (union_area + 1e-6)

def create_folders():
    for name in ['Staff', 'Customer']:
        (OUTPUT_DIR / name).mkdir(parents=True, exist_ok=True)

def process():
    create_folders()
    image_paths = list(IMG_DIR.glob('*.jpg'))
    
    for img_path in tqdm(image_paths):
        image = cv2.imread(str(img_path))
        if image is None: continue
        img_h, img_w, _ = image.shape
        
        label_path = LBL_DIR / (img_path.stem + '.txt')
        if not label_path.exists(): continue

        gt_boxes = [] # [x1, y1, x2, y2] 
        
        with open(label_path, 'r') as f:
            lines = f.readlines()
            
        for i, line in enumerate(lines):
            cls, xc, yc, w, h = map(float, line.split())
            
            abs_w, abs_h = w * img_w, h * img_h
            x1 = (xc * img_w) - abs_w / 2
            y1 = (yc * img_h) - abs_h / 2
            x2, y2 = x1 + abs_w, y1 + abs_h
            gt_boxes.append([x1, y1, x2, y2])
            
            margin = random.uniform(0.05, 0.10)
            pad_w = abs_w * margin
            pad_h = abs_h * margin
            
            x1_m = int(max(0, x1 - pad_w))
            y1_m = int(max(0, y1 - pad_h))
            x2_m = int(min(img_w, x2 + pad_w))
            y2_m = int(min(img_h, y2 + pad_h))
            
            crop = image[y1_m:y2_m, x1_m:x2_m]
            if crop.size > 0:
                class_name = CLASS_MAP.get(int(cls), 'Unknown')
                if class_name != 'Unknown':
                    cv2.imwrite(str(OUTPUT_DIR / class_name / f"{img_path.stem}_{i}.jpg"), crop)

if __name__ == "__main__":
    process()