"""
Скрипт для создания submission с ансамблированием моделей и Soft-NMS.

Поддерживаемые методы:
- WBF (Weighted Box Fusion)
- WBF + Soft-NMS (двухэтапное слияние)
- NMS (обычный)
- Soft-NMS (вместо жёсткого удаления)

Автоматически загружает параметры из best_params.txt
"""

import os
import re
import json
import pandas as pd
import numpy as np
from pathlib import Path
from ensemble_boxes import weighted_boxes_fusion
import torch
import torchvision


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

SOLUTION_CSV = "sample_sub.csv"          # Путь к файлу, задающему порядок картинок
OUTPUT_CSV = "submission_wbf_softnms.csv"  # Итоговый файл

# Пути к моделям - ИСПОЛЬЗУЙТЕ predictions/ для оригинальных предсказаний
# Или predictions_tta/ для TTA предсказаний
PREDS_BASE_DIR = "predictions_tta"  # ← Измените на "predictions_tta" если используете TTA

MODEL_DIRS = [
    "predictions_tta/yolo11m_fold_1",
    "predictions_tta/yolo11m_fold_2",
    "predictions_tta/rtdetrv1_fold_1",
    "predictions_tta/rtdetrv1_fold_2",
]

# Веса моделей (должно совпадать с длиной MODEL_DIRS)
MODEL_WEIGHTS = [1.0, 1.0, 1.0, 1.0]

# Настройки WBF
IOU_THR = 0.55         # Порог пересечения (IoU) для слияния боксов
SKIP_BOX_THR = 0.001   # Игнорировать боксы с уверенностью ниже этого ДО WBF
CONF_THR_FINAL = 0.20  # Фильтр после WBF/Soft-NMS 

# Настройки Soft-NMS
USE_SOFT_NMS = True            # Использовать Soft-NMS вместо обычного NMS
SOFT_NMS_METHOD = 'gaussian'   # 'linear' или 'gaussian'
SOFT_NMS_SIGMA = 0.5           # Sigma для gaussian Soft-NMS
SOFT_NMS_IOU_THR = 0.5         # Порог IoU для Soft-NMS
SOFT_NMS_CONF_THR = 0.20       # ← Минимальная уверенность после Soft-NMS (до CONF_THR_FINAL)

# Метод ансамблирования
ENSEMBLE_METHOD = 'wbf_softnms'  # 'wbf' | 'wbf_softnms' | 'nms' | 'softnms'


# ================= ФУНКЦИИ =================

def yolo_to_xyxy(xc, yc, w, h):
    """Конвертация [x_center, y_center, width, height] в [x1, y1, x2, y2]"""
    x1 = max(0.0, xc - (w / 2))
    y1 = max(0.0, yc - (h / 2))
    x2 = min(1.0, xc + (w / 2))
    y2 = min(1.0, yc + (h / 2))
    return [x1, y1, x2, y2]


def xyxy_to_yolo(x1, y1, x2, y2):
    """Конвертация [x1, y1, x2, y2] обратно в [x_center, y_center, width, height]"""
    w = x2 - x1
    h = y2 - y1
    xc = x1 + (w / 2)
    yc = y1 + (h / 2)
    return [xc, yc, w, h]


def soft_nms_single_class(boxes, scores, iou_threshold=0.5, method='gaussian', sigma=0.5, conf_threshold=0.1):
    """
    Soft-NMS для одного класса.
    """
    if len(boxes) == 0:
        return torch.tensor([], dtype=torch.long), torch.tensor([])
    
    boxes = boxes.clone()
    scores = scores.clone()
    
    N = len(boxes)
    keep = []
    
    # Индексы, отсортированные по убыванию уверенности
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
        
        # Вычисление IoU
        x1 = torch.max(current_box[:, 0], remaining_boxes[:, 0])
        y1 = torch.max(current_box[:, 1], remaining_boxes[:, 1])
        x2 = torch.min(current_box[:, 2], remaining_boxes[:, 2])
        y2 = torch.min(current_box[:, 3], remaining_boxes[:, 3])
        
        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        area_current = (current_box[:, 2] - current_box[:, 0]) * (current_box[:, 3] - current_box[:, 1])
        area_remaining = (remaining_boxes[:, 2] - remaining_boxes[:, 0]) * (remaining_boxes[:, 3] - remaining_boxes[:, 1])
        union = area_current + area_remaining - intersection
        
        ious = intersection / (union + 1e-10)
        
        # Применяем Soft-NMS
        if method == 'linear':
            weight = torch.ones_like(ious)
            mask = ious > iou_threshold
            weight[mask] = 1 - ious[mask]
            scores[remaining_indices] *= weight
        elif method == 'gaussian':
            weight = torch.exp(-(ious ** 2) / sigma)
            scores[remaining_indices] *= weight
        
        keep.append(current_idx)
        
        # Удаляем боксы с уверенностью ниже порога
        mask = scores[remaining_indices] > conf_threshold
        indices = remaining_indices[mask]
        
        if len(indices) > 0:
            indices = indices[torch.argsort(scores[indices], descending=True)]
    
    keep_tensor = torch.tensor(keep, dtype=torch.long)
    final_mask = scores[keep_tensor] > conf_threshold
    
    return keep_tensor[final_mask], scores[keep_tensor[final_mask]]


def apply_soft_nms(boxes, scores, labels, iou_threshold=0.5, method='gaussian', sigma=0.5, conf_threshold=0.1):
    """
    Soft-NMS для всех классов.
    """
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


def ensemble_and_build_submission(
    solution_csv: str,
    model_dirs: list,
    output_csv: str,
    model_weights: list = None,
    iou_thr: float = 0.5,
    skip_box_thr: float = 0.001,
    conf_thr_final: float = 0.25,
    use_soft_nms: bool = True,
    soft_nms_method: str = 'gaussian',
    soft_nms_sigma: float = 0.5,
    soft_nms_iou_thr: float = 0.5,
    soft_nms_conf_thr: float = 0.1,
    ensemble_method: str = 'wbf_softnms',
    image_col: str = "image_name",
    boxes_col: str = "boxes",
    row_id_col: str = "id",
    keep_only_class: int | None = None,
):
    """
    Создание submission с ансамблированием и Soft-NMS.
    """
    sol_path = Path(solution_csv)
    if not sol_path.exists():
        raise FileNotFoundError(f"Файл {solution_csv} не найден.")

    sol = pd.read_csv(sol_path)
    if image_col not in sol.columns:
        raise ValueError(f"В {solution_csv} нет колонки '{image_col}'")

    image_names = sol[image_col].astype(str).tolist()
    rows = []

    print(f"\n{'='*60}")
    print(f"АНСАМБЛИРОВАНИЕ + {'SOFT-NMS' if use_soft_nms else 'NMS'}")
    print(f"{'='*60}")
    print(f"Метод: {ensemble_method}")
    print(f"Моделей: {len(model_dirs)}")
    print(f"Изображений: {len(image_names)}")
    print(f"Параметры:")
    print(f"  IoU threshold: {iou_thr}")
    print(f"  Skip box threshold: {skip_box_thr}")
    print(f"  Final confidence threshold: {conf_thr_final}")
    if use_soft_nms:
        print(f"  Soft-NMS method: {soft_nms_method}")
        print(f"  Soft-NMS sigma: {soft_nms_sigma}")
        print(f"  Soft-NMS IoU threshold: {soft_nms_iou_thr}")
    
    # Проверка наличия предсказаний
    print(f"\n[1] Проверка папок с предсказаниями...")
    for pdir in model_dirs:
        if not os.path.exists(pdir):
            print(f"  [!] Папка не найдена: {pdir}")
        else:
            n_files = len(os.listdir(pdir))
            print(f"  [✓] {pdir}: {n_files} файлов")
    
    # Счётчики для статистики
    total_boxes_before = 0
    total_boxes_after = 0
    images_with_detections = 0
    
    print(f"\n[2] Обработка изображений...")
    
    for idx, image_name in enumerate(image_names):
        stem = Path(image_name).stem

        boxes_list = []
        scores_list = []
        labels_list = []

        # 1. Собираем предсказания всех моделей
        for pdir in model_dirs:
            pred_file = Path(pdir) / f"{stem}.txt"

            model_boxes = []
            model_scores = []
            model_labels = []

            if pred_file.exists():
                content = pred_file.read_text(encoding="utf-8").strip()
                if content:
                    for ln in content.splitlines():
                        parts = ln.split()
                        if len(parts) < 6:
                            continue

                        try:
                            cls = int(float(parts[0]))
                            xc = float(parts[1])
                            yc = float(parts[2])
                            w = float(parts[3])
                            h = float(parts[4])
                            sc = float(parts[5])
                        except ValueError:
                            continue

                        # Фильтр по классу (если указан)
                        if keep_only_class is not None and cls != keep_only_class:
                            continue

                        # Переводим в формат x1, y1, x2, y2 для WBF/NMS
                        model_boxes.append(yolo_to_xyxy(xc, yc, w, h))
                        model_scores.append(sc)
                        model_labels.append(cls)

            boxes_list.append(model_boxes)
            scores_list.append(model_scores)
            labels_list.append(model_labels)
            total_boxes_before += len(model_boxes)

        # 2. Применяем ансамблирование
        final_boxes_json = []

        # Проверяем, есть ли вообще боксы
        if any(len(b) > 0 for b in boxes_list):
            
            if ensemble_method in ['wbf', 'wbf_softnms']:
                # WBF слияние
                try:
                    b_wbf, s_wbf, l_wbf = weighted_boxes_fusion(
                        boxes_list, scores_list, labels_list,
                        weights=model_weights,
                        iou_thr=iou_thr,
                        skip_box_thr=skip_box_thr
                    )
                    
                    if ensemble_method == 'wbf':
                        # Только WBF, без Soft-NMS
                        for box, score, label in zip(b_wbf, s_wbf, l_wbf):
                            cls_id = int(label)
                            conf = float(score)
                            
                            if conf < conf_thr_final:
                                continue
                            
                            xc, yc, w, h = xyxy_to_yolo(*box)
                            final_boxes_json.append([xc, yc, w, h, conf])
                    
                    elif ensemble_method == 'wbf_softnms':
                        # WBF + Soft-NMS
                        wbf_boxes = []
                        wbf_scores = []
                        wbf_labels = []
                        
                        for box, score, label in zip(b_wbf, s_wbf, l_wbf):
                            wbf_boxes.append(list(box))  # уже в XYXY
                            wbf_scores.append(score)
                            wbf_labels.append(int(label))
                        
                        # Применяем Soft-NMS
                        snms_boxes, snms_scores, snms_labels = apply_soft_nms(
                            wbf_boxes, wbf_scores, wbf_labels,
                            iou_threshold=soft_nms_iou_thr,
                            method=soft_nms_method,
                            sigma=soft_nms_sigma,
                            conf_threshold=soft_nms_conf_thr
                        )
                        
                        # Конвертируем обратно в YOLO и фильтруем
                        # Формат для submission: [xc, yc, w, h, conf] - БЕЗ КЛАССА!
                        for box, score, label in zip(snms_boxes, snms_scores, snms_labels):
                            cls_id = int(label)
                            conf = float(score)

                            if conf < conf_thr_final:
                                continue

                            xc, yc, w, h = xyxy_to_yolo(*box)
                            final_boxes_json.append([xc, yc, w, h, conf])
                
                except Exception as e:
                    print(f"  [!] Ошибка WBF для {image_name}: {e}")
                    # Fallback на простой NMS
                    all_boxes = []
                    all_scores = []
                    all_labels = []
                    for m_boxes, m_scores, m_labels in zip(boxes_list, scores_list, labels_list):
                        all_boxes.extend(m_boxes)
                        all_scores.extend(m_scores)
                        all_labels.extend(m_labels)
                    
                    if len(all_boxes) > 0:
                        boxes_tensor = torch.tensor(all_boxes, dtype=torch.float32)
                        scores_tensor = torch.tensor(all_scores, dtype=torch.float32)
                        labels_tensor = torch.tensor(all_labels, dtype=torch.int64)
                        
                        keep_indices = torchvision.ops.batched_nms(
                            boxes_tensor, scores_tensor, labels_tensor, iou_threshold=iou_thr
                        )
                        
                        for i in keep_indices:
                            if scores_tensor[i].item() >= conf_thr_final:
                                box = boxes_tensor[i].tolist()
                                xc, yc, w, h = xyxy_to_yolo(*box)
                                # Формат: [xc, yc, w, h, conf] - БЕЗ КЛАССА!
                                final_boxes_json.append([xc, yc, w, h, scores_tensor[i].item()])
            
            elif ensemble_method in ['nms', 'softnms']:
                # Собираем все боксы в одну кучу
                all_boxes = []
                all_scores = []
                all_labels = []
                
                for m_boxes, m_scores, m_labels in zip(boxes_list, scores_list, labels_list):
                    all_boxes.extend(m_boxes)
                    all_scores.extend(m_scores)
                    all_labels.extend(m_labels)
                
                if len(all_boxes) > 0:
                    if ensemble_method == 'nms':
                        # Обычный NMS
                        boxes_tensor = torch.tensor(all_boxes, dtype=torch.float32)
                        scores_tensor = torch.tensor(all_scores, dtype=torch.float32)
                        labels_tensor = torch.tensor(all_labels, dtype=torch.int64)
                        
                        keep_indices = torchvision.ops.batched_nms(
                            boxes_tensor, scores_tensor, labels_tensor, iou_threshold=iou_thr
                        )
                        
                        for i in keep_indices:
                            if scores_tensor[i].item() >= conf_thr_final:
                                box = boxes_tensor[i].tolist()
                                xc, yc, w, h = xyxy_to_yolo(*box)
                                # Формат: [xc, yc, w, h, conf] - БЕЗ КЛАССА!
                                final_boxes_json.append([xc, yc, w, h, scores_tensor[i].item()])
                    
                    elif ensemble_method == 'softnms':
                        # Только Soft-NMS
                        snms_boxes, snms_scores, snms_labels = apply_soft_nms(
                            all_boxes, all_scores, all_labels,
                            iou_threshold=soft_nms_iou_thr,
                            method=soft_nms_method,
                            sigma=soft_nms_sigma,
                            conf_threshold=soft_nms_conf_thr
                        )

                        for box, score, label in zip(snms_boxes, snms_scores, snms_labels):
                            if score >= conf_thr_final:
                                xc, yc, w, h = xyxy_to_yolo(*box)
                                # Формат: [xc, yc, w, h, conf] - БЕЗ КЛАССА!
                                final_boxes_json.append([xc, yc, w, h, score])

        # Подсчёт статистики
        total_boxes_after += len(final_boxes_json)
        if len(final_boxes_json) > 0:
            images_with_detections += 1

        # 3. Добавляем строку
        rows.append({
            row_id_col: idx,
            image_col: image_name,
            boxes_col: json.dumps(final_boxes_json, separators=(",", ":"))
        })
        
        # Прогресс
        if (idx + 1) % 200 == 0:
            print(f"  Обработано {idx + 1}/{len(image_names)} изображений...")

    # 4. Сохраняем итоговый CSV
    sub = pd.DataFrame(rows, columns=[row_id_col, image_col, boxes_col])
    sub.to_csv(output_csv, index=False)
    
    print(f"\n{'='*60}")
    print(f"ГОТОВО!")
    print(f"{'='*60}")
    print(f"Сохранено в: {output_csv}")
    print(f"Всего строк: {len(sub)}")
    print(f"\nСтатистика:")
    print(f"  Боксов до ансамблирования: {total_boxes_before}")
    print(f"  Боксов после: {total_boxes_after}")
    print(f"  Изображений с детекциями: {images_with_detections}/{len(image_names)}")
    print(f"  Среднее боксов на изображение: {total_boxes_after / len(image_names):.2f}")


if __name__ == "__main__":
    # Загружаем лучшие параметры
    params = load_best_params()
    
    # Обновляем настройки из загруженных параметров
    if params['model_weights'] is not None and len(params['model_weights']) == len(MODEL_DIRS):
        MODEL_WEIGHTS = params['model_weights']
    
    # Создаём submission
    ensemble_and_build_submission(
        solution_csv=SOLUTION_CSV,
        model_dirs=MODEL_DIRS,
        output_csv=OUTPUT_CSV,
        model_weights=MODEL_WEIGHTS,
        iou_thr=IOU_THR,
        skip_box_thr=SKIP_BOX_THR,
        conf_thr_final=CONF_THR_FINAL,
        use_soft_nms=USE_SOFT_NMS,
        soft_nms_method=SOFT_NMS_METHOD,
        soft_nms_sigma=SOFT_NMS_SIGMA,
        soft_nms_iou_thr=SOFT_NMS_IOU_THR,
        soft_nms_conf_thr=SOFT_NMS_CONF_THR,
        ensemble_method=ENSEMBLE_METHOD,
        keep_only_class=1  # Поставьте 0 или 1, если нужен конкретный класс
    )
