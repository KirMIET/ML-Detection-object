import os
import json
import pandas as pd
from pathlib import Path
from ensemble_boxes import weighted_boxes_fusion

SOLUTION_CSV = "sample_sub.csv"          
PREDS_BASE_DIR = "predictions"        
OUTPUT_CSV = "submission_wbf.csv"      

MODEL_DIRS = [
    "predictions/yolo11s_fold_1",
    "predictions/yolo11s_fold_2",
    "predictions/rtdetr_fold_1",
    "predictions/rtdetr_fold_2"
]

MODEL_WEIGHTS = [1.1, 1.1, 1.0, 1.0]  

# Настройки WBF
IOU_THR = 0.5          # Порог пересечения (IoU) для слияния боксов
SKIP_BOX_THR = 0.001   # Игнорировать боксы с уверенностью ниже этого ДО WBF
CONF_THR_FINAL = 0.25  # Игнорировать боксы с уверенностью ниже этого ПОСЛЕ WBF

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

def ensemble_and_build_submission(
    solution_csv: str,
    model_dirs: list,
    output_csv: str,
    image_col: str = "image_name",
    boxes_col: str = "boxes",
    row_id_col: str = "id",
    keep_only_class: int | None = None,
):
    sol_path = Path(solution_csv)
    if not sol_path.exists():
        raise FileNotFoundError(f"Файл {solution_csv} не найден.")

    sol = pd.read_csv(sol_path)
    if image_col not in sol.columns:
        raise ValueError(f"В {solution_csv} нет колонки '{image_col}'")

    image_names = sol[image_col].astype(str).tolist()
    rows = []

    print(f"Запуск WBF для {len(image_names)} изображений из {len(model_dirs)} моделей...")

    for idx, image_name in enumerate(image_names):
        stem = Path(image_name).stem
        
        boxes_list = []
        scores_list = []
        labels_list = []

        # 1. Собираем предсказания всех моделей для конкретной картинки
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

                        # Переводим в формат x1, y1, x2, y2 для WBF
                        model_boxes.append(yolo_to_xyxy(xc, yc, w, h))
                        model_scores.append(sc)
                        model_labels.append(cls)

            boxes_list.append(model_boxes)
            scores_list.append(model_scores)
            labels_list.append(model_labels)

        # 2. Применяем WBF (если есть хоть один найденный бокс во всех моделях)
        final_boxes_json = []
        
        # Проверяем, есть ли вообще боксы у какой-либо модели
        if any(len(b) > 0 for b in boxes_list):
            
            # WBF делает магию слияния
            b_wbf, s_wbf, l_wbf = weighted_boxes_fusion(
                boxes_list, scores_list, labels_list, 
                weights=MODEL_WEIGHTS, 
                iou_thr=IOU_THR, 
                skip_box_thr=SKIP_BOX_THR
            )
            
            # 3. Переводим обратно в YOLO формат и фильтруем по финальному порогу и классу
            for box, score, label in zip(b_wbf, s_wbf, l_wbf):
                cls_id = int(label)
                conf = float(score)
                
                if conf < CONF_THR_FINAL:
                    continue
                    
                if keep_only_class is not None and cls_id != keep_only_class:
                    continue

                xc, yc, w, h = xyxy_to_yolo(*box)
                final_boxes_json.append([xc, yc, w, h, conf])

        # 4. Добавляем строку в список (если ничего не найдено, будет [])
        rows.append({
            row_id_col: idx,
            image_col: image_name,
            boxes_col: json.dumps(final_boxes_json, separators=(",", ":"))
        })

    # 5. Сохраняем итоговый CSV
    sub = pd.DataFrame(rows, columns=[row_id_col, image_col, boxes_col])
    sub.to_csv(output_csv, index=False)
    print(f"Готово! Сохранено в {output_csv} ({len(sub)} строк).")


if __name__ == "__main__":
    ensemble_and_build_submission(
        solution_csv=SOLUTION_CSV,
        model_dirs=MODEL_DIRS,
        output_csv=OUTPUT_CSV,
        keep_only_class=1  # Поставьте 1 (или другой номер), если нужен конкретный класс
    )