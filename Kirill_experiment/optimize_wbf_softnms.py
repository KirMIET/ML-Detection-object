"""
Скрипт для оптимизации параметров для sub_create_wbf.py (метод wbf_softnms).

Сохраняет лучшие параметры в файл best_params_wbf_softnms.txt
"""

import os
import json
import glob
from pathlib import Path
import numpy as np
from itertools import product
from ensemble_boxes import weighted_boxes_fusion
import torch
import torchvision
import time
import pandas as pd


# ================= НАСТРОЙКИ =================

# Пути к валидационным данным
VAL_IMAGES_DIR = "dataset/images/val"
VAL_LABELS_DIR = "dataset/labels/val"

# Папки с предсказаниями моделей
MODEL_PREDICTIONS = [
    "predictions_finetune_val/yolo11m_fold_1",
    "predictions_finetune_val/yolo11m_fold_2",
    "predictions_finetune_val/yolo26m_fold_1",
    "predictions_finetune_val/yolo26m_fold_2",
]

# Веса моделей по умолчанию
DEFAULT_MODEL_WEIGHTS = [1.0, 1.0, 1.0, 1.0]

# Файл для сохранения лучших параметров
OUTPUT_PARAMS_FILE = "best_params_wbf_softnms.txt"

# Настройки поиска
OPTIMIZE_MODEL_WEIGHTS = True  # Оптимизировать веса моделей
FAST_MODE = False  # Быстрый режим с меньшим количеством комбинаций


# ================= ФУНКЦИИ =================

def parse_yolo_txt(filepath):
    """Читает txt файл с предсказаниями в формате YOLO."""
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
    """Читает txt файл с ground truth метками."""
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
    """Вычисляет IoU между двумя боксами в формате [x1, y1, x2, y2]."""
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


def soft_nms_single_class(boxes, scores, iou_threshold=0.5, method='gaussian', sigma=0.5, conf_threshold=0.1):
    """Soft-NMS для одного класса."""
    if len(boxes) == 0:
        return torch.tensor([], dtype=torch.long), torch.tensor([])

    boxes = boxes.clone()
    scores = scores.clone()

    N = len(boxes)
    keep = []

    indices = torch.argsort(scores, descending=True)

    while len(indices) > 0:
        if len(keep) == 0:
            current_idx = indices[0].item()
            keep.append(current_idx)
            indices = indices[1:]
            continue

        current_idx = indices[0].item()
        current_box = boxes[current_idx:current_idx+1]

        remaining_indices = indices[1:]
        if len(remaining_indices) == 0:
            break

        remaining_boxes = boxes[remaining_indices]

        x1 = torch.max(current_box[:, 0], remaining_boxes[:, 0])
        y1 = torch.max(current_box[:, 1], remaining_boxes[:, 1])
        x2 = torch.min(current_box[:, 2], remaining_boxes[:, 2])
        y2 = torch.min(current_box[:, 3], remaining_boxes[:, 3])

        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        area_current = (current_box[:, 2] - current_box[:, 0]) * (current_box[:, 3] - current_box[:, 1])
        area_remaining = (remaining_boxes[:, 2] - remaining_boxes[:, 0]) * (remaining_boxes[:, 3] - remaining_boxes[:, 1])
        union = area_current + area_remaining - intersection

        ious = intersection / (union + 1e-10)

        if method == 'linear':
            weight = torch.ones_like(ious)
            mask = ious > iou_threshold
            weight[mask] = 1 - ious[mask]
            scores[remaining_indices] *= weight
        elif method == 'gaussian':
            weight = torch.exp(-(ious ** 2) / sigma)
            scores[remaining_indices] *= weight

        keep.append(current_idx)

        mask = scores[remaining_indices] > conf_threshold
        indices = remaining_indices[mask]

        if len(indices) > 0:
            indices = indices[torch.argsort(scores[indices], descending=True)]

    keep_tensor = torch.tensor(keep, dtype=torch.long)
    final_mask = scores[keep_tensor] > conf_threshold

    return keep_tensor[final_mask], scores[keep_tensor[final_mask]]


def apply_soft_nms(boxes, scores, labels, iou_threshold=0.5, method='gaussian', sigma=0.5, conf_threshold=0.1):
    """Soft-NMS для всех классов."""
    if len(boxes) == 0:
        return [], [], []

    boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
    scores_tensor = torch.tensor(scores, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.int64)

    unique_classes = torch.unique(labels_tensor)

    final_boxes = []
    final_scores = []
    final_labels = []

    for cls in unique_classes:
        cls_mask = labels_tensor == cls
        cls_boxes = boxes_tensor[cls_mask]
        cls_scores = scores_tensor[cls_mask]

        keep_indices, new_scores = soft_nms_single_class(
            cls_boxes, cls_scores,
            iou_threshold=iou_threshold,
            method=method,
            sigma=sigma,
            conf_threshold=conf_threshold
        )

        if len(keep_indices) > 0:
            final_boxes.extend(cls_boxes[keep_indices].tolist())
            final_scores.extend(new_scores.tolist())
            final_labels.extend([cls.item()] * len(keep_indices))

    return final_boxes, final_scores, final_labels


def apply_wbf_softnms(predictions_list, model_weights, wbf_iou_thr, wbf_skip_box_thr,
                      soft_nms_iou_thr, soft_nms_sigma, conf_thr_final):
    """Применяет WBF + Soft-NMS к предсказаниям от нескольких моделей."""
    boxes_list = []
    scores_list = []
    labels_list = []

    for preds, weight in zip(predictions_list, model_weights):
        model_boxes = []
        model_scores = []
        model_labels = []

        for box in preds:
            cls, xc, yc, w, h, conf = box

            # Фильтр по confidence (применяется до WBF через skip_box_thr)
            if conf < wbf_skip_box_thr:
                continue

            # Конвертируем в XYXY для WBF
            x1, y1, x2, y2 = yolo_to_xyxy(xc, yc, w, h)

            model_boxes.append([x1, y1, x2, y2])
            model_scores.append(conf)  # НЕ умножаем на вес - WBF сам применет
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
            iou_thr=wbf_iou_thr,
            skip_box_thr=wbf_skip_box_thr
        )
    except Exception as e:
        return []

    # Подготавливаем боксы для Soft-NMS
    wbf_boxes = []
    wbf_scores = []
    wbf_labels = []

    for box, score, label in zip(fused_boxes, fused_scores, fused_labels):
        wbf_boxes.append(list(box))  # уже в XYXY
        wbf_scores.append(score)
        wbf_labels.append(int(label))

    # Применяем Soft-NMS
    snms_boxes, snms_scores, snms_labels = apply_soft_nms(
        wbf_boxes, wbf_scores, wbf_labels,
        iou_threshold=soft_nms_iou_thr,
        method='gaussian',
        sigma=soft_nms_sigma,
        conf_threshold=conf_thr_final  # Soft-NMS conf threshold
    )

    # Конвертируем обратно в YOLO и фильтруем по финальному порогу
    final_boxes = []
    for box, score, label in zip(snms_boxes, snms_scores, snms_labels):
        if score < conf_thr_final:
            continue
        
        xc, yc, w, h = xyxy_to_yolo(*box)
        final_boxes.append([int(label), xc, yc, w, h, float(score)])

    return final_boxes


def calculate_map_95(pred_boxes, gt_boxes, iou_thresholds=np.arange(0.5, 1.0, 0.05)):
    """Вычисляет mAP@0.5:0.95 между предсказаниями и ground truth."""
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

            cumsum_fp = np.cumsum(fp)
            cumsum_tp = np.cumsum(tp)

            recall = cumsum_tp / len(cls_gts)
            precision = cumsum_tp / (cumsum_tp + cumsum_fp + 1e-10)

            mrec = np.concatenate(([0.0], recall, [1.0]))
            mpre = np.concatenate(([0.0], precision, [0.0]))

            for i in range(len(mpre) - 1, 0, -1):
                mpre[i - 1] = max(mpre[i - 1], mpre[i])

            ap = 0.0
            for i in range(1, len(mrec)):
                ap += (mrec[i] - mrec[i - 1]) * mpre[i]

            ap_sum += ap

        ap_per_class[cls] = ap_sum / len(iou_thresholds)

    if len(ap_per_class) == 0:
        return 0.0

    mAP = sum(ap_per_class.values()) / len(ap_per_class)
    return mAP


def load_all_data():
    """Загрузить все предсказания и GT метки в память для кэширования."""
    print("\n[>] Загрузка данных в память...")
    
    image_files = glob.glob(os.path.join(VAL_IMAGES_DIR, "*.jpg"))
    image_files += glob.glob(os.path.join(VAL_IMAGES_DIR, "*.png"))
    
    all_predictions = {}
    all_gt = {}
    image_names = []
    
    for img_path in image_files:
        img_name = Path(img_path).stem
        image_names.append(img_name)
        
        gt_path = os.path.join(VAL_LABELS_DIR, f"{img_name}.txt")
        all_gt[img_name] = parse_ground_truth(gt_path)
        
        model_preds = []
        for model_dir in MODEL_PREDICTIONS:
            pred_path = os.path.join(model_dir, f"{img_name}.txt")
            preds = parse_yolo_txt(pred_path)
            model_preds.append(preds)
        
        all_predictions[img_name] = model_preds
    
    print(f"  [✓] Загружено {len(image_names)} изображений")
    print(f"  [✓] Загружено {sum(len(v) for v in all_gt.values())} GT боксов")
    print(f"  [✓] Загружено {sum(len(preds) for img_preds in all_predictions.values() for preds in img_preds)} предсказаний от моделей")
    
    return all_predictions, all_gt, image_names


def evaluate_on_validation(params, all_predictions, all_gt, image_names, verbose=False):
    """Оценивает качество на валидационном наборе с заданными параметрами."""
    model_weights = params['model_weights']
    wbf_iou_thr = params['wbf_iou_thr']
    wbf_skip_box_thr = params['wbf_skip_box_thr']
    soft_nms_iou_thr = params['soft_nms_iou_thr']
    soft_nms_sigma = params['soft_nms_sigma']
    conf_thr_final = params['conf_thr_final']

    if len(image_names) == 0:
        return 0.0

    total_map = 0.0
    num_images = 0
    total_gt_boxes = 0
    total_pred_boxes = 0

    for img_name in image_names:
        gt_boxes = all_gt.get(img_name, [])

        if len(gt_boxes) == 0:
            continue

        total_gt_boxes += len(gt_boxes)

        predictions_list = all_predictions.get(img_name, [[]] * len(MODEL_PREDICTIONS))
        total_model_preds = sum(len(preds) for preds in predictions_list)
        total_pred_boxes += total_model_preds

        final_boxes = apply_wbf_softnms(
            predictions_list,
            model_weights,
            wbf_iou_thr,
            wbf_skip_box_thr,
            soft_nms_iou_thr,
            soft_nms_sigma,
            conf_thr_final
        )

        map_score = calculate_map_95(final_boxes, gt_boxes)
        total_map += map_score
        num_images += 1

        if verbose and num_images <= 3:
            print(f"    {img_name}: GT={len(gt_boxes)}, Pred={len(final_boxes)}, mAP={map_score:.4f}")

    if num_images == 0:
        return 0.0

    avg_map = total_map / num_images

    if verbose:
        print(f"    Всего: {num_images} изображений")

    return avg_map


def grid_search():
    """Выполняет Grid Search по параметрам для WBF + Soft-NMS."""
    print("=" * 60)
    print("ОПТИМИЗАЦИЯ ПАРАМЕТРОВ ДЛЯ WBF + SOFT-NMS")
    print("=" * 60)
    
    print("\n[0] Конфигурация:")
    print(f"  OPTIMIZE_MODEL_WEIGHTS: {OPTIMIZE_MODEL_WEIGHTS}")
    print(f"  FAST_MODE: {FAST_MODE}")
    print(f"  MODEL_PREDICTIONS: {len(MODEL_PREDICTIONS)} моделей")

    # Проверка наличия предсказаний
    print("\n[1] Проверка папок с предсказаниями...")
    for model_dir in MODEL_PREDICTIONS:
        if not os.path.exists(model_dir):
            print(f"  [!] Папка не найдена: {model_dir}")
            return None
        txt_files = glob.glob(os.path.join(model_dir, "*.txt"))
        if len(txt_files) == 0:
            print(f"  [!] В папке нет предсказаний: {model_dir}")
            return None
        print(f"  [✓] {model_dir}: {len(txt_files)} предсказаний")

    # Проверка валидационных данных
    print("\n[2] Проверка валидационных данных...")
    val_images = glob.glob(os.path.join(VAL_IMAGES_DIR, "*.jpg"))
    val_images += glob.glob(os.path.join(VAL_IMAGES_DIR, "*.png"))
    print(f"  Найдено изображений: {len(val_images)}")

    if len(val_images) == 0:
        print(f"  [!] Изображения не найдены в: {VAL_IMAGES_DIR}")
        return None

    # Загружаем все данные в память
    all_predictions, all_gt, image_names = load_all_data()
    
    # Фильтруем изображения без GT
    image_names_with_gt = [name for name in image_names if len(all_gt.get(name, [])) > 0]
    print(f"\n[>] Изображений с GT для оценки: {len(image_names_with_gt)}")
    
    if len(image_names_with_gt) == 0:
        print("  [!] Нет изображений с GT метками!")
        return None

    print(f"\n[3] Параметры для поиска...")

    # Параметры для поиска
    if FAST_MODE:
        wbf_iou_range = [0.45, 0.55, 0.65]
        wbf_skip_range = [0.001, 0.01]
        soft_nms_iou_range = [0.4, 0.5, 0.6]
        soft_nms_sigma_range = [0.3, 0.5]
        conf_thr_range = [0.15, 0.20, 0.25]
        
        if OPTIMIZE_MODEL_WEIGHTS:
            weights_range = [
                [1.0, 1.0, 1.0, 1.0],
                [1.2, 1.2, 1.0, 1.0],
                [1.0, 1.0, 1.2, 1.2],
            ]
        else:
            weights_range = [DEFAULT_MODEL_WEIGHTS]
    else:
        wbf_iou_range = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
        wbf_skip_range = [0.001, 0.005, 0.01, 0.05, 0.1]
        soft_nms_iou_range = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
        soft_nms_sigma_range = [0.3, 0.4, 0.5, 0.6, 0.7]
        conf_thr_range = [0.10, 0.15, 0.20, 0.25, 0.30]
        
        if OPTIMIZE_MODEL_WEIGHTS:
            weights_range = [
                [1.0, 1.0, 1.0, 1.0],
                [1.2, 1.2, 1.0, 1.0],
                [1.5, 1.5, 1.0, 1.0],
                [1.0, 1.0, 1.2, 1.2],
                [1.0, 1.0, 1.5, 1.5],
                [1.3, 1.3, 1.3, 1.3],
            ]
        else:
            weights_range = [DEFAULT_MODEL_WEIGHTS]

    print(f"  WBF IoU thresholds: {wbf_iou_range}")
    print(f"  WBF Skip box thresholds: {wbf_skip_range}")
    print(f"  Soft-NMS IoU thresholds: {soft_nms_iou_range}")
    print(f"  Soft-NMS Sigma: {soft_nms_sigma_range}")
    print(f"  Final confidence thresholds: {conf_thr_range}")
    print(f"  Model weights: {len(weights_range)} вариантов")

    total_combinations = (
        len(wbf_iou_range) *
        len(wbf_skip_range) *
        len(soft_nms_iou_range) *
        len(soft_nms_sigma_range) *
        len(conf_thr_range) *
        len(weights_range)
    )

    print(f"\n  Всего комбинаций: {total_combinations:,}")
    print(f"  Ожидаемое время: ~{total_combinations * 0.15 / 60:.1f} мин (оценка)")

    print("\n" + "=" * 60)
    print("ЗАПУСК GRID SEARCH")
    print("=" * 60)

    best_params = None
    best_map = 0.0
    current_combination = 0
    
    start_time = time.time()

    for wbf_iou in wbf_iou_range:
        for wbf_skip in wbf_skip_range:
            for snms_iou in soft_nms_iou_range:
                for snms_sigma in soft_nms_sigma_range:
                    for conf_thr in conf_thr_range:
                        for weights in weights_range:
                            current_combination += 1

                            params = {
                                'model_weights': weights,
                                'wbf_iou_thr': wbf_iou,
                                'wbf_skip_box_thr': wbf_skip,
                                'soft_nms_iou_thr': snms_iou,
                                'soft_nms_sigma': snms_sigma,
                                'conf_thr_final': conf_thr
                            }

                            map_score = evaluate_on_validation(
                                params, all_predictions, all_gt, image_names_with_gt, verbose=False
                            )
                            
                            elapsed = time.time() - start_time
                            eta = (total_combinations - current_combination) * (elapsed / max(current_combination, 1)) / 60
                            
                            if current_combination % 10 == 0 or map_score > best_map:
                                print(f"[{current_combination}/{total_combinations}] "
                                      f"wbf_iou={wbf_iou}, skip={wbf_skip}, "
                                      f"snms_iou={snms_iou}, sigma={snms_sigma}, "
                                      f"conf={conf_thr}, w={weights[0]} → mAP={map_score:.4f} "
                                      f"(ост. {eta:.1f} мин)")

                            if map_score > best_map:
                                best_map = map_score
                                best_params = params.copy()
                                print(f"    ↑ НОВЫЙ ЛИДЕР! mAP={best_map:.4f}")

    elapsed_total = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ")
    print("=" * 60)

    if best_params is None:
        print("Не удалось найти оптимальные параметры!")
        return None

    print(f"\nВсего проверено комбинаций: {current_combination}")
    print(f"Затраченное время: {elapsed_total / 60:.1f} мин")
    print(f"Лучший mAP@0.5:0.95: {best_map:.4f}")
    print(f"\nЛучшие параметры:")
    print(f"  MODEL_WEIGHTS: {best_params['model_weights']}")
    print(f"  WBF IoU threshold (IOU_THR): {best_params['wbf_iou_thr']}")
    print(f"  WBF Skip box threshold (SKIP_BOX_THR): {best_params['wbf_skip_box_thr']}")
    print(f"  Soft-NMS IoU threshold (SOFT_NMS_IOU_THR): {best_params['soft_nms_iou_thr']}")
    print(f"  Soft-NMS Sigma (SOFT_NMS_SIGMA): {best_params['soft_nms_sigma']}")
    print(f"  Final confidence threshold (CONF_THR_FINAL): {best_params['conf_thr_final']}")

    save_best_params(best_params, best_map, current_combination, elapsed_total)

    return best_params


def quick_search():
    """Быстрый поиск с минимальным количеством комбинаций."""
    print("=" * 60)
    print("БЫСТРАЯ ОПТИМИЗАЦИЯ ДЛЯ WBF + SOFT-NMS")
    print("=" * 60)
    
    print("\n[0] Конфигурация:")
    print(f"  OPTIMIZE_MODEL_WEIGHTS: {OPTIMIZE_MODEL_WEIGHTS}")
    print(f"  MODEL_PREDICTIONS: {len(MODEL_PREDICTIONS)} моделей")

    # Проверка наличия предсказаний
    print("\n[1] Проверка папок с предсказаниями...")
    for model_dir in MODEL_PREDICTIONS:
        if not os.path.exists(model_dir):
            print(f"  [!] Папка не найдена: {model_dir}")
            return None
        txt_files = glob.glob(os.path.join(model_dir, "*.txt"))
        if len(txt_files) == 0:
            print(f"  [!] В папке нет предсказаний: {model_dir}")
            return None
        print(f"  [✓] {model_dir}: {len(txt_files)} файлов")

    # Загружаем все данные в память
    all_predictions, all_gt, image_names = load_all_data()
    
    # Фильтруем изображения без GT
    image_names_with_gt = [name for name in image_names if len(all_gt.get(name, [])) > 0]
    print(f"\n[>] Изображений с GT для оценки: {len(image_names_with_gt)}")
    
    if len(image_names_with_gt) == 0:
        print("  [!] Нет изображений с GT метками!")
        return None

    print("\n[2] Запуск быстрого поиска...")
    
    # Минимум комбинаций для быстрой оценки
    wbf_iou_range = [0.50, 0.55, 0.60]
    wbf_skip_range = [0.001, 0.01]
    soft_nms_iou_range = [0.45, 0.55]
    soft_nms_sigma_range = [0.4, 0.5]
    conf_thr_range = [0.15, 0.20, 0.25]
    
    if OPTIMIZE_MODEL_WEIGHTS:
        weights_range = [
            [1.0, 1.0, 1.0, 1.0],
            [1.2, 1.2, 1.0, 1.0],
        ]
    else:
        weights_range = [DEFAULT_MODEL_WEIGHTS]

    best_params = None
    best_map = 0.0

    total_combinations = (
        len(wbf_iou_range) *
        len(wbf_skip_range) *
        len(soft_nms_iou_range) *
        len(soft_nms_sigma_range) *
        len(conf_thr_range) *
        len(weights_range)
    )
    current_combination = 0
    
    start_time = time.time()

    for wbf_iou in wbf_iou_range:
        for wbf_skip in wbf_skip_range:
            for snms_iou in soft_nms_iou_range:
                for snms_sigma in soft_nms_sigma_range:
                    for conf_thr in conf_thr_range:
                        for weights in weights_range:
                            current_combination += 1
                            
                            params = {
                                'model_weights': weights,
                                'wbf_iou_thr': wbf_iou,
                                'wbf_skip_box_thr': wbf_skip,
                                'soft_nms_iou_thr': snms_iou,
                                'soft_nms_sigma': snms_sigma,
                                'conf_thr_final': conf_thr
                            }

                            map_score = evaluate_on_validation(
                                params, all_predictions, all_gt, image_names_with_gt, verbose=False
                            )
                            
                            elapsed = time.time() - start_time
                            eta = (total_combinations - current_combination) * (elapsed / max(current_combination, 1)) / 60
                            
                            print(f"[{current_combination}/{total_combinations}] "
                                  f"wbf_iou={wbf_iou}, skip={wbf_skip}, "
                                  f"snms_iou={snms_iou}, sigma={snms_sigma}, "
                                  f"conf={conf_thr} → mAP={map_score:.4f} (ост. {eta:.1f} мин)")

                            if map_score > best_map:
                                best_map = map_score
                                best_params = params.copy()
                                print(f"    ↑ НОВЫЙ ЛИДЕР! mAP={best_map:.4f}")

    elapsed_total = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"ЛУЧШИЙ mAP: {best_map:.4f}")
    print(f"Затраченное время: {elapsed_total / 60:.1f} мин")

    if best_params:
        save_best_params(best_params, best_map, current_combination, elapsed_total)

    return best_params


def save_best_params(params, map_score, total_combinations, elapsed_time=0):
    """Сохраняет лучшие параметры в файл."""

    timestamp = np.datetime64('now', 's')

    content = f"""# Лучшие параметры для WBF + Soft-NMS (sub_create_wbf.py)
# Сгенерировано: {timestamp}
# Всего проверено комбинаций: {total_combinations}
# Затраченное время: {elapsed_time / 60:.1f} мин

# ================= РЕЗУЛЬТАТ =================
# mAP@0.5:0.95: {map_score:.4f}

# ================= ПАРАМЕТРЫ =================

# Веса моделей
MODEL_WEIGHTS = {params['model_weights']}

# WBF параметры
IOU_THR = {params['wbf_iou_thr']}
SKIP_BOX_THR = {params['wbf_skip_box_thr']}

# Soft-NMS параметры
SOFT_NMS_IOU_THR = {params['soft_nms_iou_thr']}
SOFT_NMS_SIGMA = {params['soft_nms_sigma']}

# Финальный порог уверенности
CONF_THR_FINAL = {params['conf_thr_final']}

# ================= ИСПОЛЬЗОВАНИЕ =================
# В sub_create_wbf.py установите:
"""

    content += f"""
SUBMISSION_FILES = [
    "submission1.csv",
    "submission2.csv",
]

MODEL_WEIGHTS = {params['model_weights']}

IOU_THR = {params['wbf_iou_thr']}
SKIP_BOX_THR = {params['wbf_skip_box_thr']}
CONF_THR_FINAL = {params['conf_thr_final']}

USE_SOFT_NMS = True
SOFT_NMS_METHOD = 'gaussian'
SOFT_NMS_SIGMA = {params['soft_nms_sigma']}
SOFT_NMS_IOU_THR = {params['soft_nms_iou_thr']}
SOFT_NMS_CONF_THR = {params['conf_thr_final']}

ENSEMBLE_METHOD = 'wbf_softnms'
"""

    with open(OUTPUT_PARAMS_FILE, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"\nПараметры сохранены в файл: {OUTPUT_PARAMS_FILE}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Оптимизация параметров для WBF + Soft-NMS")
    parser.add_argument("--quick", action="store_true", help="Быстрый поиск (меньше комбинаций)")
    parser.add_argument("--full", action="store_true", help="Полный Grid Search")
    parser.add_argument("--no-weight-opt", action="store_true", help="Не оптимизировать веса моделей")
    parser.add_argument("--fast", action="store_true", help="Режим быстрой оптимизации (промежуточный)")

    args = parser.parse_args()

    if args.no_weight_opt:
        OPTIMIZE_MODEL_WEIGHTS = False
        print("[i] OPTIMIZE_MODEL_WEIGHTS отключено")
    
    if args.fast:
        FAST_MODE = True
        print("[i] FAST_MODE включено")

    if args.quick:
        quick_search()
    elif args.full:
        grid_search()
    else:
        print("Используйте --quick для быстрого поиска или --full для полного")
        print("Запуск быстрого поиска по умолчанию...\n")
        quick_search()
