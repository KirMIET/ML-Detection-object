"""
Скрипт для аугментации датасета с целью увеличения в 2-2.5 раза.

Использование:
    python augment_dataset.py --fold 1 --target-size 2.5
"""

import os
import cv2
import random
import shutil
import argparse
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import yaml
import numpy as np


# НАСТРОЙКИ
SOURCE_DIR = Path(r"dataset")
DEST_BASE = Path(r"augmented_datasets")

# Параметры аугментации
IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}

# Multi-scale кроп параметры
CROP_SCALES = [1.0, 0.75, 0.5]  # 100%, 75%, 50% от оригинала
CROP_OVERLAP = 0.3  # Перекрытие между кропами
MAX_CROPS_PER_SCALE = 2  # Максимум кропов на масштаб

# Flip
FLIP_PROB = 0.5  # 50% изображений будут перевернуты

# Color jitter параметры
COLOR_JITTER_PROB = 0.3  # 30% изображений получат цветовые вариации
BRIGHTNESS_LIMIT = 0.1
CONTRAST_LIMIT = 0.1
HUE_SHIFT_LIMIT = 10
SATURATION_LIMIT = 0.2

# Copy-Paste параметры
COPY_PASTE_TARGET_MULTIPLIER = 2.5  # Во сколько раз увеличить количество employee
MIN_VISIBILITY = 0.5  # Минимальная видимость объекта после вставки


def parse_yolo_label(line, img_w, img_h):
    """Парсит YOLO лейбл и возвращает bbox в абсолютных координатах."""
    parts = line.strip().split()
    if len(parts) != 5:
        return None
    
    try:
        class_id = int(parts[0])
        cx = float(parts[1]) * img_w
        cy = float(parts[2]) * img_h
        bw = float(parts[3]) * img_w
        bh = float(parts[4]) * img_h
        
        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2
        
        return {
            'class_id': class_id,
            'bbox': [x1, y1, x2, y2],
        }
    except (ValueError, IndexError):
        return None


def bbox_to_yolo(bbox, img_w, img_h):
    """Конвертирует bbox из Pascal VOC в YOLO формат."""
    x1, y1, x2, y2 = bbox
    
    # Ограничиваем координаты
    x1 = max(0, min(img_w, x1))
    y1 = max(0, min(img_h, y1))
    x2 = max(0, min(img_w, x2))
    y2 = max(0, min(img_h, y2))
    
    if x2 <= x1 or y2 <= y1:
        return None
    
    cx = (x1 + x2) / 2 / img_w
    cy = (y1 + y2) / 2 / img_h
    bw = (x2 - x1) / img_w
    bh = (y2 - y1) / img_h
    
    # Дополнительная проверка
    if bw <= 0 or bh <= 0 or cx < 0 or cx > 1 or cy < 0 or cy > 1:
        return None
    
    return [cx, cy, bw, bh]


def apply_horizontal_flip(image, bboxes):
    """Применяет горизонтальный flip к изображению и bbox."""
    img_h, img_w = image.shape[:2]
    
    flipped_image = cv2.flip(image, 1)
    
    flipped_bboxes = []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        new_x1 = img_w - x2
        new_x2 = img_w - x1
        flipped_bboxes.append([new_x1, y1, new_x2, y2])
    
    return flipped_image, flipped_bboxes


def apply_color_jitter(image):
    """Применяет случайные цветовые искажения."""
    result = image.copy().astype(np.float32)
    
    # Brightness
    if random.random() < 0.5:
        factor = random.uniform(1 - BRIGHTNESS_LIMIT, 1 + BRIGHTNESS_LIMIT)
        result = result * factor
    
    # Contrast
    if random.random() < 0.5:
        factor = random.uniform(1 - CONTRAST_LIMIT, 1 + CONTRAST_LIMIT)
        result = result * factor + (1 - factor) * 128
    
    # Hue и Saturation (через HSV)
    if random.random() < 0.5:
        hsv = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        
        # Hue shift
        h_shift = random.uniform(-HUE_SHIFT_LIMIT, HUE_SHIFT_LIMIT)
        hsv[:, :, 0] = (hsv[:, :, 0] + h_shift) % 180
        
        # Saturation scale
        s_scale = random.uniform(1 - SATURATION_LIMIT, 1 + SATURATION_LIMIT)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * s_scale, 0, 255)
        
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result


def crop_and_filter_bboxes(bboxes, x_offset, y_offset, crop_w, crop_h, img_w, img_h):
    """Кропает bbox и фильтрует те, что не попадают в кроп."""
    cropped_bboxes = []
    
    for item in bboxes:
        bx1, by1, bx2, by2 = item['bbox']
        class_id = item['class_id']
        
        # Проверяем пересечение с кропом
        inter_x1 = max(bx1, x_offset)
        inter_y1 = max(by1, y_offset)
        inter_x2 = min(bx2, x_offset + crop_w)
        inter_y2 = min(by2, y_offset + crop_h)
        
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            continue  # bbox не пересекается с кропом
        
        # Вычисляем площадь пересечения и оригинальную площадь
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        orig_area = (bx2 - bx1) * (by2 - by1)
        
        if orig_area == 0:
            continue
        
        visibility = inter_area / orig_area
        
        if visibility >= MIN_VISIBILITY:
            # Трансформируем bbox в координаты кропа
            new_bbox = [
                bx1 - x_offset,
                by1 - y_offset,
                bx2 - x_offset,
                by2 - y_offset
            ]
            cropped_bboxes.append({
                'class_id': class_id,
                'bbox': new_bbox
            })
    
    return cropped_bboxes


def save_sample(image, bboxes, images_dir, labels_dir, filename):
    """Сохраняет изображение и лейблы."""
    img_h, img_w = image.shape[:2]
    
    img_path = images_dir / f"{filename}.jpg"
    lbl_path = labels_dir / f"{filename}.txt"
    
    # Сохраняем изображение
    cv2.imwrite(str(img_path), image)
    
    # Сохраняем лейблы в YOLO формате
    with open(lbl_path, 'w') as f:
        for item in bboxes:
            yolo_bbox = bbox_to_yolo(item['bbox'], img_w, img_h)
            if yolo_bbox:
                f.write(f"{item['class_id']} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}\n")
    
    return img_path, lbl_path


class AugmentedDatasetGenerator:
    """Генератор аугментированного датасета."""

    def __init__(self, source_dir, dest_dir, target_multiplier=2.5, seed=42):
        self.source_dir = Path(source_dir)
        self.dest_dir = Path(dest_dir)
        self.target_multiplier = target_multiplier
        self.seed = seed

        random.seed(seed)
        np.random.seed(seed)

        self.source_images = self.source_dir / 'images' / 'train'
        self.source_labels = self.source_dir / 'labels' / 'train'
        self.val_images = self.source_dir / 'images' / 'val'
        self.val_labels = self.source_dir / 'labels' / 'val'

    def create_output_structure(self):
        """Создаёт структуру папок для выходного датасета."""
        for split in ['train', 'val']:
            (self.dest_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
            (self.dest_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    def copy_validation_set(self):
        """Копирует validation set без изменений."""
        print("Копирование validation set...")

        for img_path in tqdm(list(self.val_images.iterdir())):
            if img_path.suffix.lower() not in IMG_EXTENSIONS:
                continue
            
            label_path = self.val_labels / (img_path.stem + '.txt')
            
            dst_img = self.dest_dir / 'images' / 'val' / img_path.name
            dst_label = self.dest_dir / 'labels' / 'val' / (img_path.stem + '.txt')
            
            shutil.copy2(img_path, dst_img)
            if label_path.exists():
                shutil.copy2(label_path, dst_label)
        
        print(f"  Скопировано {len(list(self.val_images.iterdir()))} изображений")
    
    def load_bboxes(self, label_path, img_w, img_h):
        """Загружает bbox из лейбл файла."""
        bboxes = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parsed = parse_yolo_label(line, img_w, img_h)
                    if parsed:
                        bboxes.append(parsed)
        return bboxes
    
    def generate_multi_scale_crops(self, image, bboxes, output_dirs, base_name):
        """Генерирует multi-scale кропы."""
        img_h, img_w = image.shape[:2]
        saved_count = 0
        
        for scale_idx, scale in enumerate(CROP_SCALES):
            if scale == 1.0:
                # Оригинальный размер
                cropped_bboxes = bboxes  # Все bbox подходят
                if len(cropped_bboxes) > 0:
                    save_sample(image, cropped_bboxes, output_dirs['images'], output_dirs['labels'], f"{base_name}_scale{scale_idx}_orig")
                    saved_count += 1
            else:
                # Кропы меньшего размера
                crop_w = int(img_w * scale)
                crop_h = int(img_h * scale)
                
                if crop_w <= 0 or crop_h <= 0:
                    continue
                
                # Генерируем позиции для кропов
                positions = []
                
                # Центр
                cx, cy = img_w // 2, img_h // 2
                positions.append((max(0, cx - crop_w // 2), max(0, cy - crop_h // 2)))
                
                # Левый верхний угол
                positions.append((0, 0))
                
                # Правый нижний угол
                positions.append((max(0, img_w - crop_w), max(0, img_h - crop_h)))
                
                # Случайные позиции
                while len(positions) < MAX_CROPS_PER_SCALE + 1:
                    rx = random.randint(0, max(0, img_w - crop_w))
                    ry = random.randint(0, max(0, img_h - crop_h))
                    positions.append((rx, ry))
                
                # Убираем дубликаты
                positions = list(set(positions))[:MAX_CROPS_PER_SCALE]
                
                for i, (x_off, y_off) in enumerate(positions):
                    cropped_img = image[y_off:y_off+crop_h, x_off:x_off+crop_w]
                    
                    if cropped_img.size == 0:
                        continue
                    
                    cropped_bboxes = crop_and_filter_bboxes(
                        bboxes, x_off, y_off, crop_w, crop_h, img_w, img_h
                    )
                    
                    if len(cropped_bboxes) > 0:
                        save_sample(cropped_img, cropped_bboxes, output_dirs['images'], output_dirs['labels'], 
                                   f"{base_name}_scale{scale_idx}_crop{i}")
                        saved_count += 1
        
        return saved_count
    
    def generate_flipped(self, image, bboxes, output_dirs, base_name):
        """Генерирует flipped версию."""
        # Извлекаем только bbox из словарей
        bbox_list = [item['bbox'] for item in bboxes]
        flipped_img, flipped_bboxes = apply_horizontal_flip(image, bbox_list)
        
        # Возвращаем обратно в формат словарей
        flipped_items = [{'class_id': item['class_id'], 'bbox': bbox} 
                        for item, bbox in zip(bboxes, flipped_bboxes)]
        
        save_sample(flipped_img, flipped_items, output_dirs['images'], output_dirs['labels'], f"{base_name}_flip")
        return 1
    
    def generate_color_jitter(self, image, bboxes, output_dirs, base_name):
        """Генерирует color jitter версию."""
        jittered_img = apply_color_jitter(image)
        # bboxes не изменяются, передаём как есть
        save_sample(jittered_img, bboxes, output_dirs['images'], output_dirs['labels'], f"{base_name}_color")
        return 1
    
    def generate_augmented_train(self):
        """Генерирует аугментированный train set."""
        print("\nГенерация аугментированного train set...")

        train_images_dir = self.dest_dir / 'images' / 'train'
        train_labels_dir = self.dest_dir / 'labels' / 'train'
        
        output_dirs = {
            'images': train_images_dir,
            'labels': train_labels_dir
        }
        
        # Собираем все изображения
        source_images = [
            p for p in self.source_images.iterdir()
            if p.suffix.lower() in IMG_EXTENSIONS
        ]
        
        total_original = len(source_images)
        target_total = int(total_original * self.target_multiplier)
        
        print(f"  Оригинально: {total_original} изображений")
        print(f"  Цель: {target_total} изображений (x{self.target_multiplier})")
        
        # Считаем employee для статистики
        total_employee = 0
        images_with_employee = []
        
        for img_path in source_images:
            label_path = self.source_labels / (img_path.stem + '.txt')
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            
            img_h, img_w = image.shape[:2]
            bboxes = self.load_bboxes(label_path, img_w, img_h)
            
            emp_count = sum(1 for b in bboxes if b['class_id'] == 1)
            total_employee += emp_count
            if emp_count > 0:
                images_with_employee.append((img_path, label_path, bboxes))
        
        print(f"  Employee объектов: {total_employee}")
        print(f"  Изображений с employee: {len(images_with_employee)}")
        
        saved_total = 0
        
        # Этап 1: Multi-scale кропы
        print("\n  [1/4] Multi-scale кропы...")
        for img_path in tqdm(source_images):
            label_path = self.source_labels / (img_path.stem + '.txt')
            
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            
            img_h, img_w = image.shape[:2]
            bboxes = self.load_bboxes(label_path, img_w, img_h)
            
            if len(bboxes) == 0:
                continue
            
            count = self.generate_multi_scale_crops(image, bboxes, output_dirs, img_path.stem)
            saved_total += count
        
        print(f"    Сохранено: {saved_total}")
        
        # Этап 2: Horizontal Flip
        print("\n  [2/4] Horizontal Flip...")
        flip_count = 0
        flip_candidates = random.sample(source_images, min(int(len(source_images) * FLIP_PROB), len(source_images)))
        
        for img_path in tqdm(flip_candidates):
            label_path = self.source_labels / (img_path.stem + '.txt')
            
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            
            img_h, img_w = image.shape[:2]
            bboxes = self.load_bboxes(label_path, img_w, img_h)
            
            if len(bboxes) == 0:
                continue
            
            self.generate_flipped(image, bboxes, output_dirs, img_path.stem)
            flip_count += 1
        
        print(f"    Сохранено: {flip_count}")
        saved_total += flip_count
        
        # Этап 3: Color Jitter
        print("\n  [3/4] Color Jitter...")
        color_count = 0
        color_candidates = random.sample(source_images, min(int(len(source_images) * COLOR_JITTER_PROB), len(source_images)))
        
        for img_path in tqdm(color_candidates):
            label_path = self.source_labels / (img_path.stem + '.txt')
            
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            
            img_h, img_w = image.shape[:2]
            bboxes = self.load_bboxes(label_path, img_w, img_h)
            
            if len(bboxes) == 0:
                continue
            
            self.generate_color_jitter(image, bboxes, output_dirs, img_path.stem)
            color_count += 1
        
        print(f"    Сохранено: {color_count}")
        saved_total += color_count
        
        # Этап 4: Copy-Paste для employee
        print("\n  [4/4] Copy-Paste для employee...")
        if images_with_employee:
            self.apply_copy_paste(images_with_employee, output_dirs, total_employee)
        
        # Финальный подсчёт
        final_count = len(list(train_images_dir.iterdir()))
        print(f"\n  Итого: {final_count} изображений в train")
        
        return final_count
    
    def apply_copy_paste(self, employee_images, output_dirs, original_employee_count):
        """Применяет Copy-Paste аугментацию для миноритарного класса."""
        target_count = int(original_employee_count * COPY_PASTE_TARGET_MULTIPLIER)
        objects_to_insert = target_count - original_employee_count
        
        if objects_to_insert <= 0:
            print("    Copy-Paste не требуется (цель достигнута)")
            return
        
        # Собираем все employee объекты
        employee_objects = []
        
        for img_path, label_path, bboxes in employee_images:
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            
            img_h, img_w = image.shape[:2]
            
            for bbox_item in bboxes:
                if bbox_item['class_id'] == 1:  # employee
                    x1, y1, x2, y2 = bbox_item['bbox']
                    
                    # Отступ для контекста
                    margin_x = int((x2 - x1) * 0.15)
                    margin_y = int((y2 - y1) * 0.15)
                    
                    obj_x1 = max(0, int(x1) - margin_x)
                    obj_y1 = max(0, int(y1) - margin_y)
                    obj_x2 = min(img_w, int(x2) + margin_x)
                    obj_y2 = min(img_h, int(y2) + margin_y)
                    
                    employee_objects.append({
                        'image_path': img_path,
                        'bbox': [obj_x1, obj_y1, obj_x2, obj_y2],
                        'class_id': 1
                    })
        
        print(f"    Найдено employee объектов: {len(employee_objects)}")
        print(f"    Цель: вставить {objects_to_insert} объектов")
        
        if len(employee_objects) == 0:
            return
        
        # Получаем все сгенерированные изображения
        target_images = list((output_dirs['images']).iterdir())
        
        if len(target_images) == 0:
            return
        
        inserted_count = 0
        
        for _ in tqdm(range(objects_to_insert)):
            # Выбираем случайное целевое изображение
            target_img_path = random.choice(target_images)
            target_label_path = output_dirs['labels'] / (target_img_path.stem + '.txt')
            
            # Загружаем целевое изображение
            target_image = cv2.imread(str(target_img_path))
            if target_image is None:
                continue
            
            tgt_h, tgt_w = target_image.shape[:2]
            
            # Загружаем существующие лейблы
            existing_bboxes = []
            if target_label_path.exists():
                with open(target_label_path, 'r') as f:
                    for line in f.readlines():
                        parsed = parse_yolo_label(line, tgt_w, tgt_h)
                        if parsed:
                            existing_bboxes.append(parsed)
            
            # Выбираем случайный employee объект
            source_obj = random.choice(employee_objects)
            
            # Загружаем источник
            source_image = cv2.imread(str(source_obj['image_path']))
            if source_image is None:
                continue
            
            src_h, src_w = source_image.shape[:2]
            
            # Вырезаем объект
            x1, y1, x2, y2 = source_obj['bbox']
            obj_patch = source_image[y1:y2, x1:x2].copy()
            
            if obj_patch.size == 0:
                continue
            
            obj_h, obj_w = obj_patch.shape[:2]
            
            if obj_h == 0 or obj_w == 0:
                continue
            
            # Выбираем случайную позицию
            max_x = max(0, tgt_w - obj_w)
            max_y = max(0, tgt_h - obj_h)
            
            paste_x = random.randint(0, max_x)
            paste_y = random.randint(0, max_y)
            
            # Alpha blending
            alpha = random.uniform(0.7, 0.9)
            
            roi_y1, roi_y2 = paste_y, min(paste_y + obj_h, tgt_h)
            roi_x1, roi_x2 = paste_x, min(paste_x + obj_w, tgt_w)
            
            patch_h = roi_y2 - roi_y1
            patch_w = roi_x2 - roi_x1
            
            if patch_h <= 0 or patch_w <= 0:
                continue
            
            roi = target_image[roi_y1:roi_y2, roi_x1:roi_x2].astype(np.float32)
            patch = obj_patch[:patch_h, :patch_w].astype(np.float32)
            
            blended = cv2.addWeighted(patch, alpha, roi, 1 - alpha, 0)
            target_image[roi_y1:roi_y2, roi_x1:roi_x2] = blended.astype(np.uint8)
            
            # Вычисляем новый bbox
            new_bbox = [paste_x, paste_y, paste_x + obj_w, paste_y + obj_h]
            existing_bboxes.append({'class_id': 1, 'bbox': new_bbox})
            
            # Сохраняем результат
            cv2.imwrite(str(target_img_path), target_image)
            
            # Сохраняем обновлённые лейблы
            with open(target_label_path, 'w') as f:
                for bbox_item in existing_bboxes:
                    yolo_bbox = bbox_to_yolo(bbox_item['bbox'], tgt_w, tgt_h)
                    if yolo_bbox:
                        f.write(f"{bbox_item['class_id']} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}\n")
            
            inserted_count += 1
        
        print(f"    Вставлено объектов: {inserted_count}")
    
    def create_data_yaml(self):
        """Создаёт data.yaml для нового датасета."""
        yaml_content = {
            'path': str(self.dest_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'nc': 2,
            'names': ['customer', 'employee']
        }
        
        yaml_path = self.dest_dir / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, sort_keys=False)
        
        print(f"\n  data.yaml создан: {yaml_path}")
    
    def generate(self, fold_idx):
        """Запускает полную генерацию аугментированного датасета."""
        print(f"\n{'='*60}")
        print(f"Генерация аугментированного датасета для Fold {fold_idx}")
        print(f"{'='*60}\n")
        
        # Разные seed для разных фолдов
        random.seed(self.seed + fold_idx * 1000)
        np.random.seed(self.seed + fold_idx * 1000)
        
        # Очищаем папку если существует
        if self.dest_dir.exists():
            print(f"Очистка существующей папки: {self.dest_dir}")
            import shutil
            shutil.rmtree(self.dest_dir)
        
        # Создаём структуру
        self.create_output_structure()
        
        # Копируем validation
        self.copy_validation_set()
        
        # Генерируем train
        self.generate_augmented_train()
        
        # Создаём data.yaml
        self.create_data_yaml()
        
        print(f"\n{'='*60}")
        print(f"✅ Готово! Датасет создан: {self.dest_dir}")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Аугментация датасета для object detection')
    parser.add_argument('--fold', type=int, required=True, help='Номер фолда (1, 2, 3)')
    parser.add_argument('--target-size', type=float, default=2.5, help='Целевой множитель размера (по умолчанию 2.5)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output', type=str, default=None, help='Путь для выходного датасета')
    
    args = parser.parse_args()
    
    # Определяем выходную папку
    if args.output:
        dest_dir = Path(args.output)
    else:
        dest_dir = DEST_BASE / f"fold_{args.fold}_augmented"
    
    # Создаём генератор
    generator = AugmentedDatasetGenerator(
        source_dir=SOURCE_DIR,
        dest_dir=dest_dir,
        target_multiplier=args.target_size,
        seed=args.seed
    )
    
    # Запускаем генерацию
    generator.generate(fold_idx=args.fold)


if __name__ == "__main__":
    main()
