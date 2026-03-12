import json
import torch
import torchvision
import pandas as pd
from pathlib import Path

def yolo_to_xyxy(xc, yc, w, h):
    """Переводит YOLO [xc, yc, w, h] в формат [x_min, y_min, x_max, y_max] для NMS"""
    x_min = max(0.0, xc - w / 2.0)
    y_min = max(0.0, yc - h / 2.0)
    x_max = min(1.0, xc + w / 2.0)
    y_max = min(1.0, yc + h / 2.0)
    return [x_min, y_min, x_max, y_max]

def xyxy_to_yolo(x_min, y_min, x_max, y_max):
    """Возвращает из [x_min, y_min, x_max, y_max] обратно в YOLO [xc, yc, w, h]"""
    xc = (x_min + x_max) / 2.0
    yc = (y_min + y_max) / 2.0
    w = x_max - x_min
    h = y_max - y_min
    return [xc, yc, w, h]

def parse_yolo_txt_for_nms(
    filepath: Path, 
    require_score: bool, 
    keep_only_class: int | None,
    conf_thr: float,
    weight: float
):
    """Читает 1 txt файл. Умножает score на вес модели (для приоритизации в NMS)."""
    boxes, scores, labels = [], [], []
    if not filepath.exists():
        return boxes, scores, labels
        
    content = filepath.read_text(encoding="utf-8", errors="replace").strip()
    if not content:
        return boxes, scores, labels
        
    for ln in content.splitlines():
        ln = ln.strip()
        if not ln: continue
        parts = ln.split()
        if require_score and len(parts) < 6: continue
        if len(parts) < 5: continue

        try:
            cls = int(float(parts[0]))
            if keep_only_class is not None and cls != keep_only_class: continue

            xc, yc, w, h = map(float, parts[1:5])
            sc = float(parts[5]) if len(parts) >= 6 else 1.0
        except ValueError:
            continue
            
        # ЖЕСТКИЙ ФИЛЬТР: Сразу отсекаем мусор (до применения весов)
        if sc < conf_thr:
            continue
            
        # Применяем вес модели. 
        # Если weight > 1.0, эта модель будет чаще "побеждать" в NMS
        weighted_score = sc * weight

        boxes.append(yolo_to_xyxy(xc, yc, w, h))
        scores.append(weighted_score)
        labels.append(cls)
        
    return boxes, scores, labels

def build_nms_submission_from_solution_order(
    solution_csv: str,
    preds_dirs: list[str],
    weights: list[float] = None,
    iou_thr: float = 0.45,         # Порог пересечения NMS (0.45 - стандарт для NMS)
    conf_thr: float = 0.35,        # Базовый порог отсечения (берем только уверенные предикты)
    output_csv: str = "submission_nms.csv",
    image_col: str = "image_name",
    boxes_col: str = "boxes",
    row_id_col: str = "id",
    require_score: bool = True,
    keep_only_class: int | None = None,
) -> None:
    
    sol_path = Path(solution_csv)
    if not sol_path.exists(): raise FileNotFoundError(f"Not found: {sol_path}")
        
    sol = pd.read_csv(sol_path)
    
    if weights is None:
        weights = [1.0] * len(preds_dirs)
        
    if len(weights) != len(preds_dirs):
        raise ValueError("Длина weights должна совпадать с количеством preds_dirs")

    image_names = sol[image_col].astype(str).tolist()
    rows = []

    for idx, image_name in enumerate(image_names):
        stem = Path(image_name).stem
        
        # Сюда скидываем в одну кучу рамки СО ВСЕХ моделей
        all_boxes, all_scores, all_labels = [], [], []
        
        for pdir, w in zip(preds_dirs, weights):
            pred_file = Path(pdir) / f"{stem}.txt"
            b, s, l = parse_yolo_txt_for_nms(pred_file, require_score, keep_only_class, conf_thr, w)
            all_boxes.extend(b)
            all_scores.extend(s)
            all_labels.extend(l)

        final_boxes_json = []
        
        if len(all_boxes) > 0:
            # 1. Переводим в тензоры PyTorch
            boxes_tensor = torch.tensor(all_boxes, dtype=torch.float32)
            scores_tensor = torch.tensor(all_scores, dtype=torch.float32)
            labels_tensor = torch.tensor(all_labels, dtype=torch.int64)

            # 2. Магия: batched_nms применяет NMS сразу ко всем классам раздельно!
            keep_indices = torchvision.ops.batched_nms(
                boxes=boxes_tensor,
                scores=scores_tensor,
                idxs=labels_tensor,
                iou_threshold=iou_thr
            )

            # 3. Достаем боксы, которые "выжили" после NMS
            final_b = boxes_tensor[keep_indices].numpy()
            final_s = scores_tensor[keep_indices].numpy()

            # 4. Формируем твой JSON формат
            for box, score in zip(final_b, final_s):
                xc, yc, w, h = xyxy_to_yolo(*box)
                
                # ЯВНО приводим все numpy.float32 к родному питоновскому float
                xc_f = float(xc)
                yc_f = float(yc)
                w_f  = float(w)
                h_f  = float(h)
                sc_f = float(score)
                
                # Ограничиваем score до 1.0 (если он стал больше из-за весов > 1.0)
                sc_f = min(1.0, sc_f)

                final_boxes_json.append([
                    round(xc_f, 5), 
                    round(yc_f, 5), 
                    round(w_f, 5), 
                    round(h_f, 5), 
                    round(sc_f, 5)
                ])

        rows.append({
            row_id_col: idx,
            image_col: image_name,
            boxes_col: json.dumps(final_boxes_json, separators=(",", ":"))
        })

    sub = pd.DataFrame(rows, columns=[row_id_col, image_col, boxes_col])
    sub.to_csv(output_csv, index=False)
    print(f"Saved: {output_csv}. NMS Ensembling finished.")

if __name__ == "__main__":
    model_directories = [
        r"D:\LabsMIET\ML-Detection-object\predictions\yolo_fold_1\labels",
        r"D:\LabsMIET\ML-Detection-object\predictions\rtdetr_fold_1\labels",
        r"D:\LabsMIET\ML-Detection-object\predictions\faster_rcnn_fold_1\labels"
    ]
    
    # Настройки "Только лучшее из лучших" для NMS:
    build_nms_submission_from_solution_order(
        solution_csv="sample_sub.csv",
        preds_dirs=model_directories,
        
        weights=[1.0, 1.2, 0.8], 
        
        iou_thr=0.6,  # 0.45-0.5 - классика. Если объекты сильно налеплены друг на друга, ставь 0.6
        conf_thr=0.45, # ГЛАВНЫЙ ФИЛЬТР ОТ МУСОРА. Все что ниже 40% даже не дойдет до NMS
        
        output_csv="ensemble_submission_nms.csv"
    )