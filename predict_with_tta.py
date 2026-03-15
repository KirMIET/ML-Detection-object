"""
Полноценный предикт с Test-Time Augmentation (TTA) для всех моделей.

Поддерживаемые аугментации:
- Multi-scale (разные размеры изображений)
- Flip (горизонтальное отражение)

Автоматически находит все модели в runs/ и загружает лучшие параметры из best_params.txt
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


# ================= АВТОМАТИЧЕСКАЯ ЗАГРУЗКА ПАРАМЕТРОВ =================

def load_best_params(params_file="best_params.txt"):
    """Загрузить лучшие параметры из файла."""
    params = {
        'conf_threshold': 0.25,
        'iou_threshold': 0.5,
        'skip_box_threshold': 0.001,
        'model_weights': None
    }
    
    if not os.path.exists(params_file):
        print(f"[!] Файл {params_file} не найден, используются параметры по умолчанию")
        return params
    
    with open(params_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Парсим CONF_THRESHOLD
    match = re.search(r'CONF_THRESHOLD\s*=\s*([\d.]+)', content)
    if match:
        params['conf_threshold'] = float(match.group(1))
    
    # Парсим IOU_THRESHOLD
    match = re.search(r'IOU_THRESHOLD\s*=\s*([\d.]+)', content)
    if match:
        params['iou_threshold'] = float(match.group(1))
    
    # Парсим SKIP_BOX_THRESHOLD
    match = re.search(r'SKIP_BOX_THRESHOLD\s*=\s*([\d.]+)', content)
    if match:
        params['skip_box_threshold'] = float(match.group(1))
    
    # Парсим MODEL_WEIGHTS
    match = re.search(r'MODEL_WEIGHTS\s*=\s*\[(.*?)\]', content)
    if match:
        weights_str = match.group(1)
        params['model_weights'] = [float(w.strip()) for w in weights_str.split(',')]
    
    print(f"[✓] Загружены параметры из {params_file}:")
    print(f"    CONF_THRESHOLD: {params['conf_threshold']}")
    print(f"    IOU_THRESHOLD: {params['iou_threshold']}")
    print(f"    SKIP_BOX_THRESHOLD: {params['skip_box_threshold']}")
    print(f"    MODEL_WEIGHTS: {params['model_weights']}")
    
    return params


# ================= НАСТРОЙКИ =================

# Пути
TEST_IMAGES_DIR = "test_images/test_images"  # или "dataset/images/val" для валидации
OUTPUT_BASE_DIR = "predictions_tta"
RUNS_DIR = "runs"

# TTA параметры
TTA_SCALES = [640]#, 768, 896, 1024]  # Размеры для multi-scale TTA
TTA_FLIP = True  # Горизонтальное отражение

# NMS порог для усреднения боксов между аугментациями
TTA_NMS_IOU_THRESHOLD = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Загружаем лучшие параметры
BEST_PARAMS = load_best_params()
CONF_THRESHOLD = BEST_PARAMS['conf_threshold']


# ================= ПОИСК МОДЕЛЕЙ =================

def find_all_models(runs_dir):
    """
    Автоматически найти все модели с весами best.pt в папке runs/.
    
    Returns:
        dict: {model_name: {"type": "yolo"|"rtdetr", "weights": path}}
    """
    models = {}
    
    # Ищем все best.pt
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
    """
    Усреднить боксы из разных аугментаций с помощью NMS-подобного подхода.
    
    boxes_list: список списков боксов от разных аугментаций
    iou_threshold: порог IoU для группировки боксов
    
    Returns:
        Усреднённые боксы [cls, xc, yc, w, h, avg_conf]
    """
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
    """
    Вычислить IoU между двумя боксами в формате dict.
    """
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


# ================= ИНФЕРЕНС =================

def predict_yolo_with_tta(weights_path, output_folder, conf_threshold, tta_scales, tta_flip):
    """
    Полный инференс для YOLO с TTA (multi-scale + flip).
    """
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
    print(f"    TTA параметры: scales={tta_scales}, flip={tta_flip}")
    
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


def predict_rtdetr_with_tta(weights_path, output_folder, conf_threshold, tta_scales, tta_flip):
    """
    Полный инференс для RT-DETR с TTA (multi-scale + flip).
    """
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
    print(f"    TTA параметры: scales={tta_scales}, flip={tta_flip}")
    
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


def predict_faster_rcnn_with_tta(weights_path, output_folder, conf_threshold, tta_scales, tta_flip):
    """
    TTA инференс для Faster R-CNN.
    """
    import torchvision
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, fasterrcnn_mobilenet_v3_large_fpn
    from PIL import Image
    import torchvision.transforms.functional as F
    
    print(f"\n[>] Загрузка Faster R-CNN модели: {weights_path}")
    
    if not os.path.exists(weights_path):
        print(f"    [!] Веса не найдены: {weights_path}")
        return False
    
    # Определяем количество классов из пути или используем по умолчанию
    num_classes = 2  # customer + employee + background = 2 класса (background = 0)
    
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    image_paths = glob.glob(os.path.join(TEST_IMAGES_DIR, "*.jpg"))
    image_paths += glob.glob(os.path.join(TEST_IMAGES_DIR, "*.png"))
    
    if len(image_paths) == 0:
        print(f"    [!] Изображения не найдены в: {TEST_IMAGES_DIR}")
        return False
    
    print(f"    [✓] Модель загружена. Изображений: {len(image_paths)}")
    print(f"    TTA параметры: scales={tta_scales}, flip={tta_flip}")
    
    os.makedirs(output_folder, exist_ok=True)
    
    total_images = 0
    total_boxes = 0
    
    with torch.no_grad():
        for img_path in image_paths:
            img_name = Path(img_path).stem
            txt_path = os.path.join(output_folder, f"{img_name}.txt")
            
            # Загружаем изображение
            image = Image.open(img_path).convert("RGB")
            img_width, img_height = image.size
            
            all_aug_predictions = []
            
            # 1. Предсказание для каждого масштаба
            for scale in tta_scales:
                # Ресайз
                image_resized = image.resize((scale, scale), Image.Resampling.LANCZOS)
                
                # Конвертация в тензор
                image_tensor = F.to_tensor(image_resized).unsqueeze(0).to(DEVICE)
                
                # Предсказание
                outputs = model(image_tensor)[0]
                
                # Собираем боксы
                aug_boxes = []
                boxes = outputs['boxes'].cpu().numpy()
                scores = outputs['scores'].cpu().numpy()
                labels = outputs['labels'].cpu().numpy()
                
                for box, label, score in zip(boxes, labels, scores):
                    if score < conf_threshold:
                        continue
                    
                    x1, y1, x2, y2 = box
                    
                    # Конвертация в нормализованные xc, yc, w, h
                    w = x2 - x1
                    h = y2 - y1
                    xc = (x1 + w / 2) / scale
                    yc = (y1 + h / 2) / scale
                    w_norm = w / scale
                    h_norm = h / scale
                    
                    cls_id = label - 1  # Фон = 0, реальные классы с 1
                    
                    aug_boxes.append([cls_id, xc, yc, w_norm, h_norm, score])
                
                all_aug_predictions.append(aug_boxes)
            
            # 2. Flip TTA
            if tta_flip:
                for scale in tta_scales:
                    # Переворачиваем
                    image_flipped = image.transpose(Image.FLIP_LEFT_RIGHT)
                    image_flipped_resized = image_flipped.resize((scale, scale), Image.Resampling.LANCZOS)
                    
                    # Конвертация в тензор
                    image_tensor = F.to_tensor(image_flipped_resized).unsqueeze(0).to(DEVICE)
                    
                    # Предсказание
                    outputs = model(image_tensor)[0]
                    
                    # Собираем боксы и отражаем обратно
                    aug_boxes = []
                    boxes = outputs['boxes'].cpu().numpy()
                    scores = outputs['scores'].cpu().numpy()
                    labels = outputs['labels'].cpu().numpy()
                    
                    for box, label, score in zip(boxes, labels, scores):
                        if score < conf_threshold:
                            continue
                        
                        x1, y1, x2, y2 = box
                        
                        # Отражение x обратно
                        x1_flipped = scale - x2
                        x2_flipped = scale - x1
                        
                        w = x2_flipped - x1_flipped
                        h = y2 - y1
                        xc = (x1_flipped + w / 2) / scale
                        yc = (y1 + h / 2) / scale
                        w_norm = w / scale
                        h_norm = h / scale
                        
                        cls_id = label - 1
                        
                        aug_boxes.append([cls_id, xc, yc, w_norm, h_norm, score])
                    
                    all_aug_predictions.append(aug_boxes)
            
            # 3. Усреднение боксов
            final_boxes = boxes_nms_merge(all_aug_predictions, TTA_NMS_IOU_THRESHOLD)
            
            # 4. Сохранение
            with open(txt_path, 'w', encoding='utf-8') as f:
                for box in final_boxes:
                    cls, xc, yc, w, h, conf = box
                    f.write(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f} {conf:.6f}\n")
            
            total_images += 1
            total_boxes += len(final_boxes)
            
            if total_images % 50 == 0:
                print(f"    Обработано {total_images}/{len(image_paths)} изображений...")
    
    print(f"    [✓] Готово! {total_images} изображений, {total_boxes} боксов")
    return True


# ================= MAIN =================

def main():
    print("=" * 60)
    print("ПОЛНЫЙ TTA ИНФЕРЕНС ДЛЯ ВСЕХ МОДЕЛЕЙ")
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
    print("\n[2] Поиск моделей в runs/...")
    models = find_all_models(RUNS_DIR)
    
    if len(models) == 0:
        print("\n  [!] Модели не найдены!")
        return
    
    print(f"\n  Всего моделей: {len(models)}")
    
    # TTA конфигурация
    print("\n[3] TTA конфигурация:")
    print(f"  Scales: {TTA_SCALES}")
    print(f"  Flip: {TTA_FLIP}")
    print(f"  NMS IoU threshold: {TTA_NMS_IOU_THRESHOLD}")
    print(f"  Confidence threshold: {CONF_THRESHOLD}")
    
    num_augmentations = len(TTA_SCALES)
    if TTA_FLIP:
        num_augmentations *= 2
    print(f"\n  Всего аугментаций на изображение: {num_augmentations}")
    print(f"  Ожидаемое время: ~{num_augmentations * 0.1 * len(test_images) / 60:.1f} мин на модель")
    
    # Инференс
    print("\n[4] Запуск инференса с TTA...")
    
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
                weights_path, output_folder, CONF_THRESHOLD, TTA_SCALES, TTA_FLIP
            )
        elif model_type == "rtdetr":
            success = predict_rtdetr_with_tta(
                weights_path, output_folder, CONF_THRESHOLD, TTA_SCALES, TTA_FLIP
            )
        elif model_type == "faster_rcnn":
            success = predict_faster_rcnn_with_tta(
                weights_path, output_folder, CONF_THRESHOLD, TTA_SCALES, TTA_FLIP
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
    print(f'  MODEL_DIRS = {model_dirs}')
    print(f'  MODEL_WEIGHTS = {BEST_PARAMS["model_weights"]}')
    print(f'  IOU_THR = {BEST_PARAMS["iou_threshold"]}')
    print(f'  SKIP_BOX_THR = {BEST_PARAMS["skip_box_threshold"]}')
    print(f'  CONF_THR_FINAL = {BEST_PARAMS["conf_threshold"]}')


if __name__ == "__main__":
    main()
