"""
Полноценный предикт с Test-Time Augmentation (TTA) и SAHI для всех моделей.
"""

import os
import glob
import re
from pathlib import Path
import torch
import cv2
import numpy as np
from PIL import Image
import shutil


# ================= НАСТРОЙКИ =================

# Пути
TEST_IMAGES_DIR = "test_images/test_images" 
OUTPUT_BASE_DIR = "predictions_finetune_tta_sahi"
RUNS_DIR = "runs"

MODEL_WEIGHTS_PATHS = [
    "runs/detect/runs/finetune_add_data/yolo11m_fold_1_finetuned/weights/best.pt",
    "runs/detect/runs/finetune_add_data/yolo11m_fold_2_finetuned/weights/best.pt",
    "runs/detect/runs/finetune_add_data/yolo26m_fold_1_finetuned/weights/best.pt",
    "runs/detect/runs/finetune_add_data/yolo26m_fold_2_finetuned/weights/best.pt",
    # "runs/detect/runs/ensemble_training/rtdetr_fold_1/weights/best.pt",
    # "runs/detect/runs/ensemble_training/rtdetr_fold_2/weights/best.pt",
]

# TTA параметры
TTA_SCALES = [640, 768]  # Размеры для multi-scale TTA
TTA_FLIP = True  # Горизонтальное отражение

# SAHI параметры
USE_SAHI = True  # Включить SAHI
SAHI_SLICE_SIZE = 512  # Размер слайса для SAHI
SAHI_OVERLAP_RATIO = 0.2  # Коэффициент перекрытия между слайсами
SAHI_MERGE_IOU_THRESHOLD = 0.5  # IoU порог для слияния боксов SAHI

# NMS порог для усреднения боксов между аугментациями
TTA_NMS_IOU_THRESHOLD = 0.5

# Порог уверенности
CONF_THRESHOLD = 0.2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================= ПОИСК МОДЕЛЕЙ =================

def find_all_models(runs_dir, manual_weights_paths=None):
    """Найти модели для инференса."""
    models = {}

    # Если указаны пути вручную, используем их
    if manual_weights_paths and len(manual_weights_paths) > 0:
        print(f"\n[>] Используем вручную указанные модели: {len(manual_weights_paths)}")
        
        for weight_path in manual_weights_paths:
            if not os.path.exists(weight_path):
                print(f"    [!] Веса не найдены: {weight_path}")
                continue
            
            # Извлекаем имя модели из пути
            model_dir = Path(weight_path).parent.parent
            model_name = model_dir.name
            
            # Определяем тип модели по имени
            model_type = None
            if 'yolo' in model_name.lower():
                model_type = 'yolo'
            elif 'rtdetr' in model_name.lower() or 'detr' in model_name.lower():
                model_type = 'rtdetr'
        
            
            if model_type is None:
                # Пытаемся определить по названию файла
                if 'yolo' in weight_path.lower():
                    model_type = 'yolo'
                elif 'rtdetr' in weight_path.lower() or 'detr' in weight_path.lower():
                    model_type = 'rtdetr'
                else:
                    print(f"    [!] Не удалось определить тип модели: {model_name}")
                    continue
            
            # Создаем уникальное имя папки для предсказаний
            safe_name = re.sub(r'[^\w\-_]', '_', model_name)
            
            models[safe_name] = {
                "type": model_type,
                "weights": weight_path,
                "original_name": model_name
            }
            
            print(f"    [✓] {safe_name} ({model_type}): {weight_path}")
        
        return models

    # Автоматический поиск всех best.pt
    pattern = os.path.join(runs_dir, "**", "best.pt")
    weight_files = glob.glob(pattern, recursive=True)

    print(f"\n[>] Найдено моделей с best.pt: {len(weight_files)}")

    for weight_path in weight_files:
        # Извлекаем имя модели из пути
        # Пример: runs/detect/runs/ensemble_training/yolo26m_fold_1/weights/best.pt
        parts = Path(weight_path).parts

        # Находим имя модели (предпоследняя директория перед weights)
        model_dir = Path(weight_path).parent.parent
        model_name = model_dir.name

        # Определяем тип модели по имени
        model_type = None
        if 'yolo' in model_name.lower():
            model_type = 'yolo'
        elif 'rtdetr' in model_name.lower() or 'detr' in model_name.lower():
            model_type = 'rtdetr'
        elif 'faster' in model_name.lower() or 'frcnn' in model_name.lower():
            model_type = 'faster_rcnn'

        if model_type is None:
            # Пытаемся определить по названию файла
            if 'yolo' in weight_path.lower():
                model_type = 'yolo'
            elif 'rtdetr' in weight_path.lower() or 'detr' in weight_path.lower():
                model_type = 'rtdetr'
            else:
                print(f"    [!] Не удалось определить тип модели: {model_name}")
                continue

        # Создаем уникальное имя папки для предсказаний
        # Заменяем специальные символы для имени папки
        safe_name = re.sub(r'[^\w\-_]', '_', model_name)

        models[safe_name] = {
            "type": model_type,
            "weights": weight_path,
            "original_name": model_name
        }

        print(f"    [✓] {safe_name} ({model_type}): {weight_path}")

    return models


# ================= TTA ФУНКЦИИ =================

def boxes_nms_merge(boxes_list, iou_threshold):
    """Усреднить боксы из разных аугментаций с помощью NMS-подобного подхода."""
    if not boxes_list or all(len(b) == 0 for b in boxes_list):
        return []
    
    # Собираем все боксы в один список с меткой источника
    all_boxes = []
    for aug_idx, boxes in enumerate(boxes_list):
        for box in boxes:
            all_boxes.append({
                'cls': box[0],
                'xc': box[1],
                'yc': box[2],
                'w': box[3],
                'h': box[4],
                'conf': box[5],
                'aug_idx': aug_idx
            })
    
    if len(all_boxes) == 0:
        return []
    
    # Группируем боксы по классам
    by_class = {}
    for box in all_boxes:
        cls = box['cls']
        if cls not in by_class:
            by_class[cls] = []
        by_class[cls].append(box)
    
    merged_boxes = []
    
    for cls, cls_boxes in by_class.items():
        # Сортируем по confidence
        cls_boxes_sorted = sorted(cls_boxes, key=lambda x: x['conf'], reverse=True)
        
        used = [False] * len(cls_boxes_sorted)
        
        for i, box_i in enumerate(cls_boxes_sorted):
            if used[i]:
                continue
            
            # Находим все боксы с высоким IoU
            group = [box_i]
            used[i] = True
            
            for j, box_j in enumerate(cls_boxes_sorted[i+1:], start=i+1):
                if used[j]:
                    continue
                
                iou = calculate_iou_dict(box_i, box_j)
                
                if iou >= iou_threshold:
                    group.append(box_j)
                    used[j] = True
            
            # Усредняем боксы в группе
            if len(group) > 0:
                avg_xc = sum(b['xc'] for b in group) / len(group)
                avg_yc = sum(b['yc'] for b in group) / len(group)
                avg_w = sum(b['w'] for b in group) / len(group)
                avg_h = sum(b['h'] for b in group) / len(group)
                avg_conf = sum(b['conf'] for b in group) / len(group)
                
                merged_boxes.append([cls, avg_xc, avg_yc, avg_w, avg_h, avg_conf])
    
    return merged_boxes


def calculate_iou_dict(box1, box2):
    """Вычислить IoU между двумя боксами в формате dict."""
    x1_1 = box1['xc'] - box1['w'] / 2
    y1_1 = box1['yc'] - box1['h'] / 2
    x2_1 = box1['xc'] + box1['w'] / 2
    y2_1 = box1['yc'] + box1['h'] / 2

    x1_2 = box2['xc'] - box2['w'] / 2
    y1_2 = box2['yc'] - box2['h'] / 2
    x2_2 = box2['xc'] + box2['w'] / 2
    y2_2 = box2['yc'] + box2['h'] / 2

    x1 = max(x1_1, x1_2)
    y1 = max(y1_1, y1_2)
    x2 = min(x2_1, x2_2)
    y2 = min(y2_1, y2_2)

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = box1['w'] * box1['h']
    area2 = box2['w'] * box2['h']

    union = area1 + area2 - intersection

    if union == 0:
        return 0

    return intersection / union


# ================= SAHI ФУНКЦИИ =================

def generate_slices(img_width, img_height, slice_size, overlap_ratio):
    """Генерировать координаты слайсов для SAHI."""
    stride = int(slice_size * (1 - overlap_ratio))
    slices = []
    
    for y in range(0, img_height, stride):
        for x in range(0, img_width, stride):
            x_min = x
            y_min = y
            x_max = min(x + slice_size, img_width)
            y_max = min(y + slice_size, img_height)
            
            slices.append((x_min, y_min, x_max, y_max))
    
    return slices


def merge_sahi_predictions(slice_predictions, slices, img_width, img_height, iou_threshold):
    """Слить предсказания из всех слайсов SAHI в глобальные координаты."""
    if not slice_predictions or all(len(p) == 0 for p in slice_predictions):
        return []
    
    # Конвертируем локальные координаты слайсов в глобальные
    global_boxes = []
    
    for slice_idx, (slice_coords, boxes) in enumerate(zip(slices, slice_predictions)):
        x_min, y_min, x_max, y_max = slice_coords
        slice_width = x_max - x_min
        slice_height = y_max - y_min
        
        for box in boxes:
            cls_id = box[0]
            xc_local = box[1]  # нормализованные координаты внутри слайса
            yc_local = box[2]
            w_local = box[3]
            h_local = box[4]
            conf = box[5]
            
            # Конвертируем в абсолютные координаты слайса
            xc_abs = xc_local * slice_width + x_min
            yc_abs = yc_local * slice_height + y_min
            w_abs = w_local * slice_width
            h_abs = h_local * slice_height
            
            # Нормализуем к размеру изображения
            xc_global = xc_abs / img_width
            yc_global = yc_abs / img_height
            w_global = w_abs / img_width
            h_global = h_abs / img_height
            
            global_boxes.append({
                'cls': cls_id,
                'xc': xc_global,
                'yc': yc_global,
                'w': w_global,
                'h': h_global,
                'conf': conf
            })
    
    # Применяем NMS для удаления дублирующихся боксов
    if len(global_boxes) == 0:
        return []
    
    # Группируем по классам
    by_class = {}
    for box in global_boxes:
        cls = box['cls']
        if cls not in by_class:
            by_class[cls] = []
        by_class[cls].append(box)
    
    merged_boxes = []
    
    for cls, cls_boxes in by_class.items():
        # Сортируем по confidence
        cls_boxes_sorted = sorted(cls_boxes, key=lambda x: x['conf'], reverse=True)
        
        used = [False] * len(cls_boxes_sorted)
        
        for i, box_i in enumerate(cls_boxes_sorted):
            if used[i]:
                continue
            
            # Находим все боксы с высоким IoU
            group = [box_i]
            used[i] = True
            
            for j, box_j in enumerate(cls_boxes_sorted[i+1:], start=i+1):
                if used[j]:
                    continue
                
                iou = calculate_iou_dict(box_i, box_j)
                
                if iou >= iou_threshold:
                    group.append(box_j)
                    used[j] = True
            
            # Усредняем боксы в группе
            if len(group) > 0:
                avg_xc = sum(b['xc'] for b in group) / len(group)
                avg_yc = sum(b['yc'] for b in group) / len(group)
                avg_w = sum(b['w'] for b in group) / len(group)
                avg_h = sum(b['h'] for b in group) / len(group)
                avg_conf = sum(b['conf'] for b in group) / len(group)
                
                merged_boxes.append([cls, avg_xc, avg_yc, avg_w, avg_h, avg_conf])
    
    return merged_boxes


def predict_with_sahi(model, image, slice_size, overlap_ratio, conf_threshold, model_type='yolo'):
    """Выполнить инференс с использованием SAHI на одном изображении."""
    img_width, img_height = image.size
    
    # Генерируем слайсы
    slices = generate_slices(img_width, img_height, slice_size, overlap_ratio)
    
    slice_predictions = []
    
    for slice_idx, (x_min, y_min, x_max, y_max) in enumerate(slices):
        # Вырезаем слайс
        slice_img = image.crop((x_min, y_min, x_max, y_max))
        slice_width = x_max - x_min
        slice_height = y_max - y_min
        
        # Предсказание на слайсе
        if model_type in ['yolo', 'rtdetr']:
            results = model.predict(
                source=slice_img,
                conf=conf_threshold,
                device=0 if torch.cuda.is_available() else "cpu",
                verbose=False,
                augment=False,
                save=False,
            )
            
            # Собираем боксы
            aug_boxes = []
            for r in results:
                if len(r.boxes) > 0:
                    for box in r.boxes:
                        x_c, y_c, w, h = box.xywhn[0].tolist()
                        conf = box.conf[0].item()
                        cls_id = int(box.cls[0].item())
                        aug_boxes.append([cls_id, x_c, y_c, w, h, conf])
            
            slice_predictions.append(aug_boxes)
            
        elif model_type == 'faster_rcnn':
            import torchvision.transforms.functional as F
            
            image_tensor = F.to_tensor(slice_img).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                outputs = model(image_tensor)[0]
            
            aug_boxes = []
            boxes = outputs['boxes'].cpu().numpy()
            scores = outputs['scores'].cpu().numpy()
            labels = outputs['labels'].cpu().numpy()
            
            for box, label, score in zip(boxes, labels, scores):
                if score < conf_threshold:
                    continue
                
                x1, y1, x2, y2 = box
                w = x2 - x1
                h = y2 - y1
                xc = (x1 + w / 2) / slice_width
                yc = (y1 + h / 2) / slice_height
                w_norm = w / slice_width
                h_norm = h / slice_height
                cls_id = label - 1
                
                aug_boxes.append([cls_id, xc, yc, w_norm, h_norm, score])
            
            slice_predictions.append(aug_boxes)
    
    # Сливаем предсказания из всех слайсов
    merged_boxes = merge_sahi_predictions(
        slice_predictions, slices, img_width, img_height, SAHI_MERGE_IOU_THRESHOLD
    )
    
    return merged_boxes


# ================= ИНФЕРЕНС =================

def predict_yolo_with_tta(weights_path, output_folder, conf_threshold, tta_scales, tta_flip, use_sahi=False):
    """Полный инференс для YOLO с TTA и опционально SAHI."""
    from ultralytics import YOLO

    print(f"\n[>] Загрузка YOLO модели: {weights_path}")

    if not os.path.exists(weights_path):
        print(f"    [!] Веса не найдены: {weights_path}")
        return False

    model = YOLO(weights_path)

    image_paths = glob.glob(os.path.join(TEST_IMAGES_DIR, "*.jpg"))
    image_paths += glob.glob(os.path.join(TEST_IMAGES_DIR, "*.png"))

    if len(image_paths) == 0:
        print(f"    [!] Изображения не найдены в: {TEST_IMAGES_DIR}")
        return False

    print(f"    [✓] Модель загружена. Изображений: {len(image_paths)}")
    print(f"    TTA параметры: scales={tta_scales}, flip={tta_flip}, sahi={use_sahi}")

    os.makedirs(output_folder, exist_ok=True)

    # Временная директория
    temp_dir = os.path.join(output_folder, "temp")
    os.makedirs(temp_dir, exist_ok=True)

    total_images = 0
    total_boxes = 0

    for img_path in image_paths:
        img_name = Path(img_path).stem
        txt_path = os.path.join(output_folder, f"{img_name}.txt")

        # Загружаем изображение
        image = Image.open(img_path).convert("RGB")
        img_width, img_height = image.size

        all_aug_predictions = []

        # Если включен SAHI, выполняем инференс на слайсах
        if use_sahi:
            sahi_boxes = predict_with_sahi(
                model, image, SAHI_SLICE_SIZE, SAHI_OVERLAP_RATIO, conf_threshold, 'yolo'
            )
            all_aug_predictions.append(sahi_boxes)

        # 1. Предсказание на оригинальном изображении для каждого масштаба
        for scale in tta_scales:
            temp_path = os.path.join(temp_dir, f"{img_name}_{scale}.jpg")

            # Ресайз
            image_resized = image.resize((scale, scale), Image.Resampling.LANCZOS)
            image_resized.save(temp_path)

            # Предсказание
            results = model.predict(
                source=temp_path,
                conf=conf_threshold,
                device=0 if torch.cuda.is_available() else "cpu",
                verbose=False,
                augment=False,
                save=False,
            )

            # Собираем боксы
            aug_boxes = []
            for r in results:
                if len(r.boxes) > 0:
                    for box in r.boxes:
                        x_c, y_c, w, h = box.xywhn[0].tolist()
                        conf = box.conf[0].item()
                        cls_id = int(box.cls[0].item())
                        aug_boxes.append([cls_id, x_c, y_c, w, h, conf])

            all_aug_predictions.append(aug_boxes)

        # 2. Flip TTA
        if tta_flip:
            for scale in tta_scales:
                # Переворачиваем изображение
                image_flipped = image.transpose(Image.FLIP_LEFT_RIGHT)
                image_flipped_resized = image_flipped.resize((scale, scale), Image.Resampling.LANCZOS)

                temp_path = os.path.join(temp_dir, f"{img_name}_{scale}_flip.jpg")
                image_flipped_resized.save(temp_path)

                # Предсказание
                results = model.predict(
                    source=temp_path,
                    conf=conf_threshold,
                    device=0 if torch.cuda.is_available() else "cpu",
                    verbose=False,
                    augment=False,
                    save=False,
                )

                # Собираем боксы и отражаем обратно
                aug_boxes = []
                for r in results:
                    if len(r.boxes) > 0:
                        for box in r.boxes:
                            x_c, y_c, w, h = box.xywhn[0].tolist()
                            conf = box.conf[0].item()
                            cls_id = int(box.cls[0].item())

                            # Отражаем x обратно
                            x_c_flipped = 1.0 - x_c

                            aug_boxes.append([cls_id, x_c_flipped, y_c, w, h, conf])

                all_aug_predictions.append(aug_boxes)

        # 3. Усреднение боксов между аугментациями
        final_boxes = boxes_nms_merge(all_aug_predictions, TTA_NMS_IOU_THRESHOLD)

        # 4. Сохранение результатов
        with open(txt_path, 'w', encoding='utf-8') as f:
            for box in final_boxes:
                cls, xc, yc, w, h, conf = box
                f.write(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f} {conf:.6f}\n")

        total_images += 1
        total_boxes += len(final_boxes)

        if total_images % 50 == 0:
            print(f"    Обработано {total_images}/{len(image_paths)} изображений...")

    # Очистка temp директории
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

    print(f"    [✓] Готово! {total_images} изображений, {total_boxes} боксов")
    return True


def predict_rtdetr_with_tta(weights_path, output_folder, conf_threshold, tta_scales, tta_flip, use_sahi=False):
    """Полный инференс для RT-DETR с TTA и опционально SAHI."""
    from ultralytics import RTDETR

    print(f"\n[>] Загрузка RT-DETR модели: {weights_path}")

    if not os.path.exists(weights_path):
        print(f"    [!] Веса не найдены: {weights_path}")
        return False

    model = RTDETR(weights_path)

    image_paths = glob.glob(os.path.join(TEST_IMAGES_DIR, "*.jpg"))
    image_paths += glob.glob(os.path.join(TEST_IMAGES_DIR, "*.png"))

    if len(image_paths) == 0:
        print(f"    [!] Изображения не найдены в: {TEST_IMAGES_DIR}")
        return False

    print(f"    [✓] Модель загружена. Изображений: {len(image_paths)}")
    print(f"    TTA параметры: scales={tta_scales}, flip={tta_flip}, sahi={use_sahi}")

    os.makedirs(output_folder, exist_ok=True)

    # Временная директория
    temp_dir = os.path.join(output_folder, "temp")
    os.makedirs(temp_dir, exist_ok=True)

    total_images = 0
    total_boxes = 0

    for img_path in image_paths:
        img_name = Path(img_path).stem
        txt_path = os.path.join(output_folder, f"{img_name}.txt")

        # Загружаем изображение
        image = Image.open(img_path).convert("RGB")

        all_aug_predictions = []

        # Если включен SAHI, выполняем инференс на слайсах
        if use_sahi:
            sahi_boxes = predict_with_sahi(
                model, image, SAHI_SLICE_SIZE, SAHI_OVERLAP_RATIO, conf_threshold, 'rtdetr'
            )
            all_aug_predictions.append(sahi_boxes)

        # 1. Предсказание на оригинальном изображении для каждого масштаба
        for scale in tta_scales:
            temp_path = os.path.join(temp_dir, f"{img_name}_{scale}.jpg")

            # Ресайз
            image_resized = image.resize((scale, scale), Image.Resampling.LANCZOS)
            image_resized.save(temp_path)

            # Предсказание
            results = model.predict(
                source=temp_path,
                conf=conf_threshold,
                device=0 if torch.cuda.is_available() else "cpu",
                verbose=False,
                augment=False,
                save=False,
            )

            # Собираем боксы
            aug_boxes = []
            for r in results:
                if len(r.boxes) > 0:
                    for box in r.boxes:
                        x_c, y_c, w, h = box.xywhn[0].tolist()
                        conf = box.conf[0].item()
                        cls_id = int(box.cls[0].item())
                        aug_boxes.append([cls_id, x_c, y_c, w, h, conf])

            all_aug_predictions.append(aug_boxes)

        # 2. Flip TTA
        if tta_flip:
            for scale in tta_scales:
                # Переворачиваем изображение
                image_flipped = image.transpose(Image.FLIP_LEFT_RIGHT)
                image_flipped_resized = image_flipped.resize((scale, scale), Image.Resampling.LANCZOS)

                temp_path = os.path.join(temp_dir, f"{img_name}_{scale}_flip.jpg")
                image_flipped_resized.save(temp_path)

                # Предсказание
                results = model.predict(
                    source=temp_path,
                    conf=conf_threshold,
                    device=0 if torch.cuda.is_available() else "cpu",
                    verbose=False,
                    augment=False,
                    save=False,
                )

                # Собираем боксы и отражаем обратно
                aug_boxes = []
                for r in results:
                    if len(r.boxes) > 0:
                        for box in r.boxes:
                            x_c, y_c, w, h = box.xywhn[0].tolist()
                            conf = box.conf[0].item()
                            cls_id = int(box.cls[0].item())

                            # Отражаем x обратно
                            x_c_flipped = 1.0 - x_c

                            aug_boxes.append([cls_id, x_c_flipped, y_c, w, h, conf])

                all_aug_predictions.append(aug_boxes)

        # 3. Усреднение боксов между аугментациями
        final_boxes = boxes_nms_merge(all_aug_predictions, TTA_NMS_IOU_THRESHOLD)

        # 4. Сохранение результатов
        with open(txt_path, 'w', encoding='utf-8') as f:
            for box in final_boxes:
                cls, xc, yc, w, h, conf = box
                f.write(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f} {conf:.6f}\n")

        total_images += 1
        total_boxes += len(final_boxes)

        if total_images % 50 == 0:
            print(f"    Обработано {total_images}/{len(image_paths)} изображений...")

    # Очистка temp директории
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

    print(f"    [✓] Готово! {total_images} изображений, {total_boxes} боксов")
    return True

# ================= MAIN =================

def main():
    print("=" * 60)
    print("ПОЛНЫЙ TTA + SAHI ИНФЕРЕНС ДЛЯ ВСЕХ МОДЕЛЕЙ")
    print("=" * 60)

    # Проверка данных
    print("\n[1] Проверка данных...")
    test_images = glob.glob(os.path.join(TEST_IMAGES_DIR, "*.jpg"))
    test_images += glob.glob(os.path.join(TEST_IMAGES_DIR, "*.png"))

    if len(test_images) == 0:
        print(f"  [!] Изображения не найдены в: {TEST_IMAGES_DIR}")
        return

    print(f"  [✓] Найдено {len(test_images)} изображений")

    # Поиск моделей
    print("\n[2] Поиск моделей...")
    if MODEL_WEIGHTS_PATHS and len(MODEL_WEIGHTS_PATHS) > 0:
        print(f"  Используем вручную указанные веса из MODEL_WEIGHTS_PATHS")
    else:
        print(f"  Автоматический поиск в {RUNS_DIR}/...")

    models = find_all_models(RUNS_DIR, MODEL_WEIGHTS_PATHS)

    if len(models) == 0:
        print("\n  [!] Модели не найдены!")
        print("  Укажите пути к весам в переменной MODEL_WEIGHTS_PATHS в начале скрипта")
        return

    print(f"\n  Всего моделей: {len(models)}")

    # TTA + SAHI конфигурация
    print("\n[3] TTA + SAHI конфигурация:")
    print(f"  Scales: {TTA_SCALES}")
    print(f"  Flip: {TTA_FLIP}")
    print(f"  NMS IoU threshold: {TTA_NMS_IOU_THRESHOLD}")
    print(f"  Confidence threshold: {CONF_THRESHOLD}")
    print(f"  SAHI enabled: {USE_SAHI}")
    if USE_SAHI:
        print(f"  SAHI slice size: {SAHI_SLICE_SIZE}")
        print(f"  SAHI overlap ratio: {SAHI_OVERLAP_RATIO}")
        print(f"  SAHI merge IoU threshold: {SAHI_MERGE_IOU_THRESHOLD}")

    num_augmentations = len(TTA_SCALES)
    if TTA_FLIP:
        num_augmentations *= 2
    if USE_SAHI:
        num_augmentations += 1  # SAHI добавляет ещё одну аугментацию
    print(f"\n  Всего аугментаций на изображение: {num_augmentations}")
    print(f"  Ожидаемое время: ~{num_augmentations * 0.1 * len(test_images) / 60:.1f} мин на модель")

    # Инференс
    print("\n[4] Запуск инференса с TTA + SAHI...")

    results = {}

    for model_name, model_config in models.items():
        output_folder = os.path.join(OUTPUT_BASE_DIR, model_name)
        weights_path = model_config["weights"]
        model_type = model_config["type"]

        print(f"\n{'='*60}")
        print(f"Модель: {model_name} ({model_type})")
        print(f"Оригинальное имя: {model_config.get('original_name', model_name)}")
        print(f"{'='*60}")

        success = False

        if model_type == "yolo":
            success = predict_yolo_with_tta(
                weights_path, output_folder, CONF_THRESHOLD, TTA_SCALES, TTA_FLIP, USE_SAHI
            )
        elif model_type == "rtdetr":
            success = predict_rtdetr_with_tta(
                weights_path, output_folder, CONF_THRESHOLD, TTA_SCALES, TTA_FLIP, USE_SAHI
            )

        results[model_name] = {
            "success": success,
            "type": model_type,
            "output_folder": output_folder
        }

    # Итоги
    print("\n" + "=" * 60)
    print("ИТОГИ")
    print("=" * 60)

    print(f"\nПредсказания сохранены в: {os.path.abspath(OUTPUT_BASE_DIR)}")
    print("\nСтатистика по моделям:")

    for model_name, result in results.items():
        if not result["success"]:
            print(f"  {model_name}: [!] ОШИБКА")
            continue

        model_folder = result["output_folder"]
        txt_files = glob.glob(os.path.join(model_folder, "*.txt"))

        total_boxes = 0
        for txt_file in txt_files:
            with open(txt_file, 'r') as f:
                total_boxes += len(f.readlines())

        print(f"  {model_name}: {len(txt_files)} файлов, {total_boxes} боксов")

    # Генерация рекомендаций
    print(f"\n[✓] Готово к созданию submission!")
    print(f"\nДля использования в sub_create_wbf.py установите:")

    model_dirs = [os.path.join(OUTPUT_BASE_DIR, name) for name, r in results.items() if r["success"]]
    # Веса моделей для усреднения (равные веса по умолчанию)
    model_weights = [1.0] * len(model_dirs)

    print(f'  MODEL_DIRS = {model_dirs}')
    print(f'  MODEL_WEIGHTS = {model_weights}')
    print(f'  IOU_THR = {TTA_NMS_IOU_THRESHOLD}')
    print(f'  SKIP_BOX_THR = 0.001')
    print(f'  CONF_THR_FINAL = {CONF_THRESHOLD}')


if __name__ == "__main__":
    main()
