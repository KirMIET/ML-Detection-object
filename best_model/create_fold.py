import os
import cv2
import random
import shutil
import glob
from pathlib import Path
from sklearn.model_selection import KFold
from tqdm import tqdm

# --- НАСТРОЙКИ ---
DATASET_DIR = 'dataset'       
OUTPUT_DIR = 'dataset_folds' 
NUM_FOLDS = 5
CROP_SCALE_MIN = 0.6          # Минимальный размер обрезка (60% от оригинала)
CROP_SCALE_MAX = 0.9          # Максимальный размер обрезка (90% от оригинала)

def get_all_data(dataset_path):
    """Собирает все пути к изображениям и их меткам из train и val."""
    images = []
    
    # Ищем во всех папках изображений (train и val)
    for split in ['train', 'val']:
        img_dir = os.path.join(dataset_path, 'images', split)
        lbl_dir = os.path.join(dataset_path, 'labels', split)
        
        if not os.path.exists(img_dir): continue
        
        for img_path in glob.glob(os.path.join(img_dir, '*.*')):
            if not img_path.lower().endswith(('.png', '.jpg', '.jpeg')): continue
            
            filename = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(lbl_dir, filename + '.txt')
            
            # Если метки нет, создаем пустой список (изображение без объектов)
            images.append({'img': img_path, 'lbl': label_path if os.path.exists(label_path) else None})
            
    return images

def yolo_to_abs(yolo_box, img_w, img_h):
    """Конвертирует YOLO формат (x_center, y_center, w, h) в абсолютные координаты (xmin, ymin, xmax, ymax)."""
    cls_id, xc, yc, w, h = yolo_box
    xmin = (xc - w / 2) * img_w
    ymin = (yc - h / 2) * img_h
    xmax = (xc + w / 2) * img_w
    ymax = (yc + h / 2) * img_h
    return int(cls_id), xmin, ymin, xmax, ymax

def abs_to_yolo(abs_box, img_w, img_h):
    """Конвертирует абсолютные координаты обратно в YOLO формат."""
    cls_id, xmin, ymin, xmax, ymax = abs_box
    xc = ((xmin + xmax) / 2) / img_w
    yc = ((ymin + ymax) / 2) / img_h
    w = (xmax - xmin) / img_w
    h = (ymax - ymin) / img_h
    return int(cls_id), xc, yc, w, h

def random_crop(img, bboxes):
    """Случайно обрезает изображение и пересчитывает bounding boxes."""
    img_h, img_w = img.shape[:2]
    
    # Случайный размер обрезка
    crop_w = int(img_w * random.uniform(CROP_SCALE_MIN, CROP_SCALE_MAX))
    crop_h = int(img_h * random.uniform(CROP_SCALE_MIN, CROP_SCALE_MAX))
    
    # Случайные координаты верхнего левого угла обрезка
    crop_x = random.randint(0, img_w - crop_w)
    crop_y = random.randint(0, img_h - crop_h)
    
    # Обрезаем изображение
    cropped_img = img[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
    
    cropped_bboxes = []
    for box in bboxes:
        cls_id, xmin, ymin, xmax, ymax = yolo_to_abs(box, img_w, img_h)
        
        # Смещаем координаты относительно обрезка и обрезаем по краям
        n_xmin = max(xmin, crop_x) - crop_x
        n_ymin = max(ymin, crop_y) - crop_y
        n_xmax = min(xmax, crop_x + crop_w) - crop_x
        n_ymax = min(ymax, crop_y + crop_h) - crop_y
        
        # Если после обрезки бокс имеет площадь (не исчез), сохраняем его
        if n_xmax > n_xmin and n_ymax > n_ymin:
            yolo_b = abs_to_yolo((cls_id, n_xmin, n_ymin, n_xmax, n_ymax), crop_w, crop_h)
            cropped_bboxes.append(yolo_b)
            
    return cropped_img, cropped_bboxes

def read_labels(label_path):
    bboxes = []
    if label_path and os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    bboxes.append([float(x) for x in parts])
    return bboxes

def write_labels(label_path, bboxes):
    with open(label_path, 'w') as f:
        for box in bboxes:
            f.write(f"{int(box[0])} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n")

def process_and_save(data_item, save_img_dir, save_lbl_dir, do_crop=True):
    """Копирует оригинал и (опционально) создает его обрезанную версию."""
    img_path = data_item['img']
    lbl_path = data_item['lbl']
    filename = os.path.basename(img_path)
    name, ext = os.path.splitext(filename)
    
    # 1. Оригинал (копируем всегда)
    orig_img_save_path = os.path.join(save_img_dir, filename)
    orig_lbl_save_path = os.path.join(save_lbl_dir, name + '.txt')
    
    shutil.copy(img_path, orig_img_save_path)
    if lbl_path:
        shutil.copy(lbl_path, orig_lbl_save_path)
    else:
        open(orig_lbl_save_path, 'w').close() # Создаем пустой txt
        
    # Если аугментация не нужна, прерываем функцию здесь
    if not do_crop:
        return
        
    # 2. Обрезанная копия (Аугментация)
    img = cv2.imread(img_path)
    if img is None: return # Защита от битых изображений
    
    bboxes = read_labels(lbl_path)
    cropped_img, cropped_bboxes = random_crop(img, bboxes)
    
    crop_filename = f"{name}_crop{ext}"
    crop_img_save_path = os.path.join(save_img_dir, crop_filename)
    crop_lbl_save_path = os.path.join(save_lbl_dir, f"{name}_crop.txt")
    
    cv2.imwrite(crop_img_save_path, cropped_img)
    write_labels(crop_lbl_save_path, cropped_bboxes)

def main():
    print("Собираем данные...")
    data = get_all_data(DATASET_DIR)
    print(f"Найдено {len(data)} изображений.")
    
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(data), 1):
        print(f"\n--- Создание Фолда {fold_idx}/{NUM_FOLDS} ---")
        
        fold_dir = os.path.join(OUTPUT_DIR, f'fold_{fold_idx}')
        
        # Создаем структуру папок
        for split in ['train', 'val']:
            os.makedirs(os.path.join(fold_dir, 'images', split), exist_ok=True)
            os.makedirs(os.path.join(fold_dir, 'labels', split), exist_ok=True)
            
        # Копируем data.yaml если он есть
        yaml_path = os.path.join(DATASET_DIR, 'data.yaml')
        if os.path.exists(yaml_path):
            shutil.copy(yaml_path, os.path.join(fold_dir, 'data.yaml'))
            
        # Обработка Train (оригинал + crop)
        print("Обработка train сета...")
        for i in tqdm(train_idx):
            process_and_save(
                data[i], 
                os.path.join(fold_dir, 'images', 'train'),
                os.path.join(fold_dir, 'labels', 'train'),
                do_crop=True  
            )
            
        # Обработка Val (ТОЛЬКО оригиналы)
        print("Обработка val сета...")
        for i in tqdm(val_idx):
            process_and_save(
                data[i], 
                os.path.join(fold_dir, 'images', 'val'),
                os.path.join(fold_dir, 'labels', 'val'),
                do_crop=False 
            )

    print("\nГотово! Все 5 датасетов сгенерированы в папку:", OUTPUT_DIR)

if __name__ == '__main__':
    main()