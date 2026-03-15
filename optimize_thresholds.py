"""
Скрипт для оптимизации порогов и весов моделей на валидационном наборе.

Использует Grid Search для поиска оптимальных:
- Confidence threshold для каждой модели
- IoU threshold для WBF
- Весов для каждой модели

Сохраняет лучшие параметры в файл best_params.txt
"""

import os
import json
import glob
from pathlib import Path
import numpy as np
from itertools import product
from ensemble_boxes import weighted_boxes_fusion
import cv2


# ================= НАСТРОЙКИ =================

# Пути к валидационным данным
VAL_IMAGES_DIR = "dataset/images/val"  # Изображения для валидации
VAL_LABELS_DIR = "dataset/labels/val"  # Ground truth метки

# Папки с предсказаниями моделей (должны быть получены заранее)
MODEL_PREDICTIONS = [
    "predictions_val/yolo11m_fold_1",
    "predictions_val/yolo11m_fold_2",
    "predictions_val/rtdetrv1_fold_1",
    "predictions_val/rtdetrv1_fold_2",
]

# Веса моделей по умолчанию (можно переопределить)
DEFAULT_MODEL_WEIGHTS = [1.0, 1.0, 1.0, 1.0]

# Файл для сохранения лучших параметров
OUTPUT_PARAMS_FILE = "best_params.txt"


# ================= ФУНКЦИИ =================

def parse_yolo_txt(filepath):
    """
    Читает txt файл с предсказаниями в формате YOLO.
    Возвращает список: [class_id, x_c, y_c, w, h, confidence]
    """
    boxes = []
    if not os.path.exists(filepath):
        return boxes
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        if not content:
            return boxes
        
        for line in content.splitlines():
            parts = line.strip().split()
            if len(parts) >= 6:
                try:
                    cls = int(float(parts[0]))
                    xc = float(parts[1])
                    yc = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])
                    conf = float(parts[5])
                    boxes.append([cls, xc, yc, w, h, conf])
                except ValueError:
                    continue
    return boxes


def parse_ground_truth(filepath):
    """
    Читает txt файл с ground truth метками.
    Возвращает список: [class_id, x_c, y_c, w, h]
    """
    boxes = []
    if not os.path.exists(filepath):
        return boxes
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        if not content:
            return boxes
        
        for line in content.splitlines():
            parts = line.strip().split()
            if len(parts) >= 5:
                try:
                    cls = int(float(parts[0]))
                    xc = float(parts[1])
                    yc = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])
                    boxes.append([cls, xc, yc, w, h])
                except ValueError:
                    continue
    return boxes


def yolo_to_xyxy(xc, yc, w, h):
    """Конвертация YOLO формата в XYXY для WBF."""
    x1 = max(0.0, xc - w / 2)
    y1 = max(0.0, yc - h / 2)
    x2 = min(1.0, xc + w / 2)
    y2 = min(1.0, yc + h / 2)
    return [x1, y1, x2, y2]


def xyxy_to_yolo(x1, y1, x2, y2):
    """Конвертация XYXY обратно в YOLO формат."""
    w = x2 - x1
    h = y2 - y1
    xc = x1 + w / 2
    yc = y1 + h / 2
    return [xc, yc, w, h]


def calculate_iou(box1, box2):
    """
    Вычисляет IoU между двумя боксами в формате [x1, y1, x2, y2].
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0
    
    return intersection / union


def calculate_map_95(pred_boxes, gt_boxes, iou_thresholds=np.arange(0.5, 1.0, 0.05)):
    """
    Вычисляет mAP@0.5:0.95 между предсказаниями и ground truth.
    
    Args:
        pred_boxes: список предсказанных боксов [cls, xc, yc, w, h, conf]
        gt_boxes: список GT боксов [cls, xc, yc, w, h]
        iou_thresholds: пороги IoU для усреднения
    
    Returns:
        mAP значение
    """
    if len(gt_boxes) == 0:
        return 0.0
    
    if len(pred_boxes) == 0:
        return 0.0
    
    # Группируем по классам
    pred_by_class = {}
    gt_by_class = {}
    
    for box in pred_boxes:
        cls = box[0]
        if cls not in pred_by_class:
            pred_by_class[cls] = []
        pred_by_class[cls].append(box)
    
    for box in gt_boxes:
        cls = box[0]
        if cls not in gt_by_class:
            gt_by_class[cls] = []
        gt_by_class[cls].append(box)
    
    all_classes = set(pred_by_class.keys()) | set(gt_by_class.keys())
    
    ap_per_class = {}
    
    for cls in all_classes:
        cls_preds = pred_by_class.get(cls, [])
        cls_gts = gt_by_class.get(cls, [])
        
        if len(cls_gts) == 0:
            continue
        
        if len(cls_preds) == 0:
            ap_per_class[cls] = 0.0
            continue
        
        # Сортируем предсказания по confidence
        cls_preds_sorted = sorted(cls_preds, key=lambda x: x[5], reverse=True)
        
        # Вычисляем precision-recall для каждого порога IoU
        ap_sum = 0.0
        
        for iou_thr in iou_thresholds:
            tp = np.zeros(len(cls_preds_sorted))
            fp = np.zeros(len(cls_preds_sorted))
            gt_matched = [False] * len(cls_gts)
            
            for pred_idx, pred in enumerate(cls_preds_sorted):
                pred_box_xyxy = yolo_to_xyxy(pred[1], pred[2], pred[3], pred[4])
                
                best_iou = 0.0
                best_gt_idx = -1
                
                for gt_idx, gt in enumerate(cls_gts):
                    gt_box_xyxy = yolo_to_xyxy(gt[1], gt[2], gt[3], gt[4])
                    iou = calculate_iou(pred_box_xyxy, gt_box_xyxy)
                    
                    if iou > best_iou and iou >= iou_thr and not gt_matched[gt_idx]:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                if best_gt_idx >= 0:
                    tp[pred_idx] = 1
                    gt_matched[best_gt_idx] = True
                else:
                    fp[pred_idx] = 1
            
            # Вычисляем precision и recall
            cumsum_fp = np.cumsum(fp)
            cumsum_tp = np.cumsum(tp)
            
            recall = cumsum_tp / len(cls_gts)
            precision = cumsum_tp / (cumsum_tp + cumsum_fp + 1e-10)
            
            # Вычисляем AP (area under PR curve)
            # Interpolated average precision (COCO style)
            mrec = np.concatenate(([0.0], recall, [1.0]))
            mpre = np.concatenate(([0.0], precision, [0.0]))
            
            for i in range(len(mpre) - 1, 0, -1):
                mpre[i - 1] = max(mpre[i - 1], mpre[i])
            
            ap = 0.0
            for i in range(1, len(mrec)):
                ap += (mrec[i] - mrec[i - 1]) * mpre[i]
            
            ap_sum += ap
        
        ap_per_class[cls] = ap_sum / len(iou_thresholds)
    
    # Усредняем AP по всем классам
    if len(ap_per_class) == 0:
        return 0.0
    
    mAP = sum(ap_per_class.values()) / len(ap_per_class)
    return mAP


def apply_wbf_ensemble(predictions_list, conf_thresholds, model_weights, iou_thr, skip_box_thr):
    """
    Применяет WBF к предсказаниям от нескольких моделей.
    
    Args:
        predictions_list: список списков предсказаний от каждой модели
        conf_thresholds: список порогов confidence для каждой модели
        model_weights: веса моделей
        iou_thr: IoU threshold для WBF
        skip_box_thr: порог пропуска боксов
    
    Returns:
        Список финальных боксов после WBF
    """
    boxes_list = []
    scores_list = []
    labels_list = []
    
    for preds, conf_thr, weight in zip(predictions_list, conf_thresholds, model_weights):
        model_boxes = []
        model_scores = []
        model_labels = []
        
        for box in preds:
            cls, xc, yc, w, h, conf = box
            
            if conf < conf_thr:
                continue
            
            # Конвертируем в XYXY для WBF
            x1, y1, x2, y2 = yolo_to_xyxy(xc, yc, w, h)
            
            model_boxes.append([x1, y1, x2, y2])
            model_scores.append(conf * weight)  # Применяем вес модели
            model_labels.append(cls)
        
        boxes_list.append(model_boxes)
        scores_list.append(model_scores)
        labels_list.append(model_labels)
    
    # Проверяем, есть ли хоть один бокс
    if not any(len(b) > 0 for b in boxes_list):
        return []
    
    # Применяем WBF
    try:
        fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
            boxes_list,
            scores_list,
            labels_list,
            weights=model_weights,
            iou_thr=iou_thr,
            skip_box_thr=skip_box_thr
        )
    except Exception as e:
        print(f"Ошибка WBF: {e}")
        return []
    
    # Конвертируем обратно в YOLO формат
    final_boxes = []
    for box, score, label in zip(fused_boxes, fused_scores, fused_labels):
        xc, yc, w, h = xyxy_to_yolo(*box)
        final_boxes.append([int(label), xc, yc, w, h, float(score)])
    
    return final_boxes


def evaluate_on_validation(params, verbose=False):
    """
    Оценивает качество на валидационном наборе с заданными параметрами.
    
    Args:
        params: словарь с параметрами:
            - conf_thresholds: список порогов для каждой модели
            - iou_thr: IoU threshold для WBF
            - skip_box_thr: порог пропуска боксов
            - model_weights: веса моделей
        verbose: если True, выводить отладочную информацию
    
    Returns:
        mAP@0.5:0.95
    """
    conf_thresholds = params['conf_thresholds']
    iou_thr = params['iou_thr']
    skip_box_thr = params.get('skip_box_thr', 0.001)
    model_weights = params.get('model_weights', DEFAULT_MODEL_WEIGHTS)
    
    # Получаем список изображений
    image_files = glob.glob(os.path.join(VAL_IMAGES_DIR, "*.jpg"))
    image_files += glob.glob(os.path.join(VAL_IMAGES_DIR, "*.png"))
    
    if len(image_files) == 0:
        if verbose:
            print("Валидационные изображения не найдены!")
        return 0.0
    
    total_map = 0.0
    num_images = 0
    total_gt_boxes = 0
    total_pred_boxes = 0
    
    for img_path in image_files:
        img_name = Path(img_path).stem
        
        # Загружаем ground truth
        gt_path = os.path.join(VAL_LABELS_DIR, f"{img_name}.txt")
        gt_boxes = parse_ground_truth(gt_path)
        
        if len(gt_boxes) == 0:
            continue  # Пропускаем изображения без GT
        
        total_gt_boxes += len(gt_boxes)
        
        # Загружаем предсказания от каждой модели
        predictions_list = []
        total_model_preds = 0
        for model_dir in MODEL_PREDICTIONS:
            pred_path = os.path.join(model_dir, f"{img_name}.txt")
            preds = parse_yolo_txt(pred_path)
            total_model_preds += len(preds)
            predictions_list.append(preds)
        
        total_pred_boxes += total_model_preds
        
        # Применяем ансамбль с текущими параметрами
        final_boxes = apply_wbf_ensemble(
            predictions_list,
            conf_thresholds,
            model_weights,
            iou_thr,
            skip_box_thr
        )
        
        # Вычисляем mAP
        map_score = calculate_map_95(final_boxes, gt_boxes)
        total_map += map_score
        num_images += 1
        
        if verbose and num_images <= 3:
            print(f"    {img_name}: GT={len(gt_boxes)}, Pred={len(final_boxes)}, mAP={map_score:.4f}")
    
    if num_images == 0:
        if verbose:
            print(f"    Нет изображений с GT! Проверьте пути:")
            print(f"    VAL_IMAGES_DIR: {VAL_IMAGES_DIR}")
            print(f"    VAL_LABELS_DIR: {VAL_LABELS_DIR}")
        return 0.0
    
    avg_map = total_map / num_images
    
    if verbose:
        print(f"    Всего: {num_images} изображений, {total_gt_boxes} GT боксов, {total_pred_boxes} предсказаний от моделей")
    
    return avg_map


def grid_search():
    """
    Выполняет Grid Search по параметрам.
    """
    print("=" * 60)
    print("ОПТИМИЗАЦИЯ ПОРОГОВ И ВЕСОВ МОДЕЛЕЙ")
    print("=" * 60)
    
    # Проверка наличия предсказаний
    print("\n[1] Проверка папок с предсказаниями...")
    for model_dir in MODEL_PREDICTIONS:
        if not os.path.exists(model_dir):
            print(f"  [!] Папка не найдена: {model_dir}")
            print("      Сначала запустите predict_with_model.py")
            return None
        txt_files = glob.glob(os.path.join(model_dir, "*.txt"))
        if len(txt_files) == 0:
            print(f"  [!] В папке нет предсказаний: {model_dir}")
            print("      Сначала запустите predict_with_model.py")
            return None
        print(f"  [✓] {model_dir}: {len(txt_files)} предсказаний")
    
    # Проверка валидационных данных
    print("\n[2] Проверка валидационных данных...")
    val_images = glob.glob(os.path.join(VAL_IMAGES_DIR, "*.jpg"))
    val_images += glob.glob(os.path.join(VAL_IMAGES_DIR, "*.png"))
    print(f"  Найдено изображений: {len(val_images)}")
    
    if len(val_images) == 0:
        print(f"  [!] Изображения не найдены в: {VAL_IMAGES_DIR}")
        print("      Проверьте путь к валидационным данным в начале скрипта")
        return None
    
    # Проверяем первые 3 изображения
    print("\n[3] Проверка соответствия имён файлов...")
    for img_path in val_images[:3]:
        img_name = Path(img_path).stem
        gt_path = os.path.join(VAL_LABELS_DIR, f"{img_name}.txt")
        gt_exists = os.path.exists(gt_path)
        gt_boxes = parse_ground_truth(gt_path) if gt_exists else []
        
        pred_counts = []
        for model_dir in MODEL_PREDICTIONS:
            pred_path = os.path.join(model_dir, f"{img_name}.txt")
            preds = parse_yolo_txt(pred_path)
            pred_counts.append(len(preds))
        
        print(f"  {img_name}: GT={len(gt_boxes)} боксов, Предсказания={pred_counts}")
    
    print(f"\n[4] Параметры для поиска...")
    
    # Confidence thresholds (можно настроить для каждой модели отдельно)
    conf_thresholds_range = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    
    # IoU thresholds для WBF
    iou_thresholds_range = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
    
    # Skip box thresholds
    skip_box_thresholds_range = [0.001, 0.01, 0.05, 0.1]
    
    print(f"  Confidence thresholds: {conf_thresholds_range}")
    print(f"  IoU thresholds: {iou_thresholds_range}")
    print(f"  Skip box thresholds: {skip_box_thresholds_range}")
    
    total_combinations = len(conf_thresholds_range) * len(iou_thresholds_range) * len(skip_box_thresholds_range)
    print(f"\n  Всего комбинаций: {total_combinations}")
    print(f"  Ожидаемое время: ~{total_combinations * 0.5 / 60:.1f} мин (оценка)")
    
    print("\n" + "=" * 60)
    print("ЗАПУСК GRID SEARCH")
    print("=" * 60)
    
    best_params = None
    best_map = 0.0
    current_combination = 0
    
    # Упрощённый поиск: одинаковый conf_threshold для всех моделей
    for conf_thr in conf_thresholds_range:
        for iou_thr in iou_thresholds_range:
            for skip_thr in skip_box_thresholds_range:
                current_combination += 1
                
                params = {
                    'conf_thresholds': [conf_thr] * len(MODEL_PREDICTIONS),
                    'iou_thr': iou_thr,
                    'skip_box_thr': skip_thr,
                    'model_weights': DEFAULT_MODEL_WEIGHTS
                }
                
                map_score = evaluate_on_validation(params, verbose=False)
                print(f"[{current_combination}/{total_combinations}] conf={conf_thr}, iou={iou_thr}, skip={skip_thr} → mAP={map_score:.4f}")
                
                if map_score > best_map:
                    best_map = map_score
                    best_params = params.copy()
                    print(f"    ↑ НОВЫЙ ЛИДЕР!")
    
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ")
    print("=" * 60)
    
    if best_params is None:
        print("Не удалось найти оптимальные параметры!")
        return None
    
    print(f"\nВсего проверено комбинаций: {total_combinations}")
    print(f"Лучший mAP@0.5:0.95: {best_map:.4f}")
    print(f"\nЛучшие параметры:")
    print(f"  Confidence threshold (все модели): {best_params['conf_thresholds'][0]}")
    print(f"  IoU threshold (WBF): {best_params['iou_thr']}")
    print(f"  Skip box threshold: {best_params['skip_box_thr']}")
    print(f"  Model weights: {best_params['model_weights']}")
    
    # Сохраняем результаты
    save_best_params(best_params, best_map, total_combinations)
    
    return best_params


def save_best_params(params, map_score, total_combinations):
    """Сохраняет лучшие параметры в файл."""
    
    timestamp = np.datetime64('now', 's')
    
    content = f"""# Лучшие параметры для ансамбля моделей
# Сгенерировано: {timestamp}
# Всего проверено комбинаций: {total_combinations}

# ================= РЕЗУЛЬТАТ =================
# mAP@0.5:0.95: {map_score:.4f}

# ================= ПАРАМЕТРЫ =================

# Confidence threshold для всех моделей
CONF_THRESHOLD = {params['conf_thresholds'][0]}

# Индивидуальные пороги для каждой модели (если нужны)
CONF_THRESHOLDS_PER_MODEL = {params['conf_thresholds']}

# IoU threshold для Weighted Box Fusion
IOU_THRESHOLD = {params['iou_thr']}

# Skip box threshold для WBF
SKIP_BOX_THRESHOLD = {params['skip_box_thr']}

# Веса моделей
MODEL_WEIGHTS = {params['model_weights']}

# ================= ИСПОЛЬЗОВАНИЕ =================
"""
    
    # Добавляем рекомендации по использованию
    content += f"""
# В predict_with_model.py установите:
CONF_THRESHOLD = {params['conf_thresholds'][0]}

# В sub_create_wbf.py установите:
IOU_THR = {params['iou_thr']}
SKIP_BOX_THR = {params['skip_box_thr']}
MODEL_WEIGHTS = {params['model_weights']}
CONF_THR_FINAL = {params['conf_thresholds'][0]}
"""
    
    with open(OUTPUT_PARAMS_FILE, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"\nПараметры сохранены в файл: {OUTPUT_PARAMS_FILE}")


def quick_search():
    """
    Быстрый поиск с уменьшенным количеством комбинаций.
    Для первоначальной оценки.
    """
    print("=" * 60)
    print("БЫСТРАЯ ОПТИМИЗАЦИЯ (упрощённая)")
    print("=" * 60)
    
    # Проверка наличия предсказаний
    print("\n[1] Проверка папок с предсказаниями...")
    for model_dir in MODEL_PREDICTIONS:
        if not os.path.exists(model_dir):
            print(f"  [!] Папка не найдена: {model_dir}")
            print("      Сначала запустите predict_with_model.py")
            return None
        txt_files = glob.glob(os.path.join(model_dir, "*.txt"))
        if len(txt_files) == 0:
            print(f"  [!] В папке нет предсказаний: {model_dir}")
            print("      Сначала запустите predict_with_model.py")
            return None
        print(f"  [✓] {model_dir}: {len(txt_files)} файлов")
    
    # Проверка валидационных данных
    print("\n[2] Проверка валидационных данных...")
    val_images = glob.glob(os.path.join(VAL_IMAGES_DIR, "*.jpg"))
    val_images += glob.glob(os.path.join(VAL_IMAGES_DIR, "*.png"))
    print(f"  Найдено изображений: {len(val_images)}")
    
    if len(val_images) == 0:
        print(f"  [!] Изображения не найдены в: {VAL_IMAGES_DIR}")
        print("      Проверьте путь к валидационным данным в начале скрипта")
        return None
    
    # Проверяем первые 3 изображения на наличие GT и предсказаний
    print("\n[3] Проверка соответствия имён файлов...")
    for img_path in val_images[:3]:
        img_name = Path(img_path).stem
        gt_path = os.path.join(VAL_LABELS_DIR, f"{img_name}.txt")
        gt_exists = os.path.exists(gt_path)
        gt_boxes = parse_ground_truth(gt_path) if gt_exists else []
        
        # Проверяем предсказания
        pred_counts = []
        for model_dir in MODEL_PREDICTIONS:
            pred_path = os.path.join(model_dir, f"{img_name}.txt")
            preds = parse_yolo_txt(pred_path)
            pred_counts.append(len(preds))
        
        print(f"  {img_name}: GT={len(gt_boxes)} боксов, Предсказания={pred_counts}")
    
    print("\n[4] Запуск поиска...")
    conf_thresholds_range = [0.20, 0.30, 0.40]
    iou_thresholds_range = [0.40, 0.50, 0.60]
    skip_box_thresholds_range = [0.01, 0.1]
    
    best_params = None
    best_map = 0.0
    
    total_combinations = len(conf_thresholds_range) * len(iou_thresholds_range) * len(skip_box_thresholds_range)
    current_combination = 0
    
    for conf_thr in conf_thresholds_range:
        for iou_thr in iou_thresholds_range:
            for skip_thr in skip_box_thresholds_range:
                current_combination += 1
                params = {
                    'conf_thresholds': [conf_thr] * len(MODEL_PREDICTIONS),
                    'iou_thr': iou_thr,
                    'skip_box_thr': skip_thr,
                    'model_weights': DEFAULT_MODEL_WEIGHTS
                }
                
                map_score = evaluate_on_validation(params, verbose=False)
                progress = current_combination / total_combinations * 100
                print(f"[{current_combination}/{total_combinations}] conf={conf_thr}, iou={iou_thr}, skip={skip_thr} → mAP={map_score:.4f}")
                
                if map_score > best_map:
                    best_map = map_score
                    best_params = params.copy()
                    print(f"    ↑ НОВЫЙ ЛИДЕР!")
    
    print(f"\n{'='*60}")
    print(f"ЛУЧШИЙ mAP: {best_map:.4f}")
    
    if best_params:
        save_best_params(best_params, best_map, len(conf_thresholds_range) * len(iou_thresholds_range) * len(skip_box_thresholds_range))
    
    return best_params


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Оптимизация порогов и весов для ансамбля")
    parser.add_argument("--quick", action="store_true", help="Быстрый поиск (меньше комбинаций)")
    parser.add_argument("--full", action="store_true", help="Полный Grid Search")
    
    args = parser.parse_args()
    
    if args.quick:
        quick_search()
    elif args.full:
        grid_search()
    else:
        # По умолчанию быстрый поиск
        print("Используйте --quick для быстрого поиска или --full для полного")
        print("Запуск быстрого поиска по умолчанию...\n")
        quick_search()
