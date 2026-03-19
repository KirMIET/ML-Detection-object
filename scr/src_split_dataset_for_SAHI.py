import os
import shutil
import random
import cv2
from tqdm import tqdm
from pathlib import Path

IMG_DIR = Path(r"yolo_dataset/yolo_dataset/train/images")
LBL_DIR = Path(r"yolo_dataset/yolo_dataset/train/labels")

OUTPUT_DIR = "dataset_SAHI" 
TRAIN_RATIO = 0.8 

SLICE_W = 640
SLICE_H = 640
OVERLAP = 0.2

def get_yolo_bboxes(label_path):
    """Читает YOLO файл, возвращает список рамок, меняя класс на 0"""
    bboxes = []
    if not os.path.exists(label_path):
        return bboxes
    with open(label_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) == 5:
                bboxes.append([0, float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
    return bboxes

def yolo_to_abs(bbox, img_w, img_h):
    """(x_c, y_c, w, h) -> (xmin, ymin, xmax, ymax)"""
    _, x_c, y_c, w, h = bbox
    xmin = int((x_c - w / 2) * img_w)
    xmax = int((x_c + w / 2) * img_w)
    ymin = int((y_c - h / 2) * img_h)
    ymax = int((y_c + h / 2) * img_h)
    return xmin, ymin, xmax, ymax

def abs_to_yolo(xmin, ymin, xmax, ymax, slice_w, slice_h):
    """(xmin, ymin, xmax, ymax) -> (x_c, y_c, w, h)"""
    x_c = ((xmin + xmax) / 2.0) / slice_w
    y_c = ((ymin + ymax) / 2.0) / slice_h
    w = (xmax - xmin) / float(slice_w)
    h = (ymax - ymin) / float(slice_h)
    return [0, x_c, y_c, w, h] # Класс всегда 0

def slice_image_and_labels(img_name, bboxes, src_img_path, dest_img_dir, dest_lbl_dir):
    """Режет изображение и пересчитывает рамки"""
    img = cv2.imread(src_img_path)
    if img is None: return
    img_h, img_w = img.shape[:2]
    
    # Шаг 
    step_x = int(SLICE_W * (1 - OVERLAP))
    step_y = int(SLICE_H * (1 - OVERLAP))

    slice_count = 0

    for y in range(0, img_h, step_y):
        for x in range(0, img_w, step_x):
            # Корректируем края, чтобы не выйти за пределы картинки
            xmin_slice = min(x, img_w - SLICE_W) if img_w > SLICE_W else 0
            ymin_slice = min(y, img_h - SLICE_H) if img_h > SLICE_H else 0
            xmax_slice = xmin_slice + SLICE_W
            ymax_slice = ymin_slice + SLICE_H

            # Если картинка меньше слайса, берем как есть
            xmax_slice = min(xmax_slice, img_w)
            ymax_slice = min(ymax_slice, img_h)
            actual_slice_w = xmax_slice - xmin_slice
            actual_slice_h = ymax_slice - ymin_slice

            slice_bboxes = []
            
            # Проверяем каждую рамку, попала ли она в текущий слайс
            for bbox in bboxes:
                b_xmin, b_ymin, b_xmax, b_ymax = yolo_to_abs(bbox, img_w, img_h)

                # Находим пересечение рамки и слайса
                inter_xmin = max(xmin_slice, b_xmin)
                inter_ymin = max(ymin_slice, b_ymin)
                inter_xmax = min(xmax_slice, b_xmax)
                inter_ymax = min(ymax_slice, b_ymax)

                # Если есть пересечение (человек в кадре)
                if inter_xmin < inter_xmax and inter_ymin < inter_ymax:
                    # Переводим координаты относительно левого верхнего угла слайса
                    new_xmin = inter_xmin - xmin_slice
                    new_ymin = inter_ymin - ymin_slice
                    new_xmax = inter_xmax - xmin_slice
                    new_ymax = inter_ymax - ymin_slice

                    # Отсекаем огрызки (если от человека осталось меньше 10 пикселей)
                    if (new_xmax - new_xmin) > 10 and (new_ymax - new_ymin) > 10:
                        new_yolo_bbox = abs_to_yolo(new_xmin, new_ymin, new_xmax, new_ymax, actual_slice_w, actual_slice_h)
                        slice_bboxes.append(new_yolo_bbox)

            # Сохраняем слайс ТОЛЬКО если на нем есть люди (чтобы не плодить пустой фон)
            if len(slice_bboxes) > 0:
                slice_img = img[ymin_slice:ymax_slice, xmin_slice:xmax_slice]
                base_name = os.path.splitext(img_name)[0]
                slice_filename = f"{base_name}_slice_{slice_count}"
                
                # Сохраняем картинку
                cv2.imwrite(os.path.join(dest_img_dir, slice_filename + ".jpg"), slice_img)
                
                # Сохраняем лейблы
                with open(os.path.join(dest_lbl_dir, slice_filename + ".txt"), 'w') as f:
                    for b in slice_bboxes:
                        f.write(f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")
                
                slice_count += 1

def main():
    # Создаем структуру папок
    for split in ['train', 'val']:
        os.makedirs(os.path.join(OUTPUT_DIR, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, 'labels', split), exist_ok=True)

    print("Анализ датасета и подсчет людей...")
    image_files = [f for f in os.listdir(IMG_DIR) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    dataset_info = []
    total_people = 0

    for img_name in tqdm(image_files):
        base_name = os.path.splitext(img_name)[0]
        lbl_path = os.path.join(LBL_DIR, base_name + ".txt")
        
        bboxes = get_yolo_bboxes(lbl_path)
        people_count = len(bboxes)
        total_people += people_count
        
        dataset_info.append({
            'img_name': img_name,
            'bboxes': bboxes,
            'count': people_count
        })

    # Рандомное перемешивание для честного распределения
    random.seed(42)
    random.shuffle(dataset_info)

    train_target = int(total_people * TRAIN_RATIO)
    train_current_count = 0

    train_data = []
    val_data = []

    # Распределение 80/20 по количеству рамок
    for data in dataset_info:
        if train_current_count < train_target:
            train_data.append(data)
            train_current_count += data['count']
        else:
            val_data.append(data)

    print(f"\nРазбиение завершено!")
    print(f"Всего людей: {total_people}")
    print(f"В Train: {train_current_count} людей ({len(train_data)} фото)")
    print(f"В Val: {total_people - train_current_count} людей ({len(val_data)} фото)")

    # Функция для обработки сплита (копирование оригиналов + нарезка слайсов)
    def process_split(split_data, split_name):
        dest_img_dir = os.path.join(OUTPUT_DIR, 'images', split_name)
        dest_lbl_dir = os.path.join(OUTPUT_DIR, 'labels', split_name)

        print(f"\nОбработка {split_name} (Копирование оригиналов и Slicing)...")
        for data in tqdm(split_data):
            img_name = data['img_name']
            bboxes = data['bboxes']
            src_img = os.path.join(IMG_DIR, img_name)
            
            # 1. Копируем оригинал
            shutil.copy(src_img, os.path.join(dest_img_dir, img_name))
            
            # Создаем/перезаписываем лейбл для оригинала (чтобы класс точно стал 0)
            base_name = os.path.splitext(img_name)[0]
            with open(os.path.join(dest_lbl_dir, base_name + ".txt"), 'w') as f:
                for b in bboxes:
                    f.write(f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")

            # 2. Делаем нарезку и дополняем папку
            slice_image_and_labels(img_name, bboxes, src_img, dest_img_dir, dest_lbl_dir)

    # Запускаем обработку
    process_split(train_data, 'train')
    process_split(val_data, 'val')

    print(f"\nГотово! Новый датасет лежит в папке: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()