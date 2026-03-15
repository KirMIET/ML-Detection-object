"""
Скрипт для получения предсказаний моделей на валидационном наборе.

Сохраняет предсказания в папку predictions_val/{model_name}/
"""

import os
import glob
from pathlib import Path
import torch


# ================= НАСТРОЙКИ =================

# Пути к валидационным данным
VAL_IMAGES_DIR = "dataset/images/val"

# Базовая папка для сохранения предсказаний валидации
OUTPUT_BASE_DIR = "predictions_val"

# Модели для инференса
# Укажи пути к весам моделей, которые хочешь использовать

MODELS = {
    # Формат: "имя_папки": {"type": "yolo"|"rtdetr", "weights": "путь/к/весам.pt"}
    
    # YOLO11s (если есть веса)
    "yolo11m_fold_1": {
        "type": "yolo",
        "weights": "runs/detect/runs/ensemble_training/yolo11m_fold_1/weights/best.pt"
    },
    "yolo11m_fold_2": {
        "type": "yolo",
        "weights": "runs/detect/runs/ensemble_training/yolo11m_fold_2/weights/best.pt"
    },
    
    # YOLO26m
    # "yolo26m_fold_1": {
    #     "type": "yolo",
    #     "weights": "runs/detect/runs/ensemble_training/yolo26m_fold_1/weights/best.pt"
    # },
    # "yolo26m_fold_2": {
    #     "type": "yolo",
    #     "weights": "runs/detect/runs/ensemble_training/yolo26m_fold_2/weights/best.pt"
    # },
    
    # RT-DETR (если есть веса)
    "rtdetrv1_fold_1": {
        "type": "rtdetr",
        "weights": "runs/detect/runs/ensemble_training/rtdetrv1_fold_1/weights/best.pt"
    },
    "rtdetrv1_fold_2": {
        "type": "rtdetr",
        "weights": "runs/detect/runs/ensemble_training/rtdetrv1_fold_2/weights/best.pt"
    },
}

# Порог уверенности
CONF_THRESHOLD = 0.01  # Очень низкий, чтобы сохранить все боксы для последующей оптимизации

# Устройство
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================= ФУНКЦИИ =================

def predict_yolo(weights_path, output_folder, conf_threshold):
    """
    Инференс для YOLO моделей с сохранением предсказаний.
    """
    from ultralytics import YOLO
    
    print(f"\n[>] Загрузка YOLO модели: {weights_path}")
    
    if not os.path.exists(weights_path):
        print(f"    [!] Веса не найдены: {weights_path}")
        return False
    
    model = YOLO(weights_path)
    
    print(f"    [✓] Модель загружена, выполняем инференс...")
    
    # Делаем предикт
    results = model.predict(
        source=VAL_IMAGES_DIR,
        conf=conf_threshold,
        device=0 if torch.cuda.is_available() else "cpu",
        verbose=False,
        stream=True,
        augment=False,  # TTA не нужна для валидации
        save=False,     # Не сохраняем автоматически
    )
    
    # Сохраняем результаты
    os.makedirs(output_folder, exist_ok=True)
    
    num_images = 0
    total_boxes = 0
    
    for r in results:
        img_name = Path(r.path).stem
        txt_path = os.path.join(output_folder, f"{img_name}.txt")
        
        with open(txt_path, 'w', encoding='utf-8') as f:
            if len(r.boxes) > 0:
                for box in r.boxes:
                    # Ultralytics выдаёт нормализованные xywhn
                    x_c, y_c, w, h = box.xywhn[0].tolist()
                    conf = box.conf[0].item()
                    cls_id = int(box.cls[0].item())
                    
                    f.write(f"{cls_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f} {conf:.6f}\n")
                    total_boxes += 1
        
        num_images += 1
        
        if num_images % 100 == 0:
            print(f"    Обработано {num_images} изображений...")
    
    print(f"    [✓] Готово! {num_images} изображений, {total_boxes} боксов")
    return True


def predict_rtdetr(weights_path, output_folder, conf_threshold):
    """
    Инференс для RT-DETR моделей с сохранением предсказаний.
    """
    from ultralytics import RTDETR
    
    print(f"\n[>] Загрузка RT-DETR модели: {weights_path}")
    
    if not os.path.exists(weights_path):
        print(f"    [!] Веса не найдены: {weights_path}")
        return False
    
    model = RTDETR(weights_path)
    
    print(f"    [✓] Модель загружена, выполняем инференс...")
    
    results = model.predict(
        source=VAL_IMAGES_DIR,
        conf=conf_threshold,
        device=0 if torch.cuda.is_available() else "cpu",
        verbose=False,
        stream=True,
        augment=False,
        save=False,
    )
    
    os.makedirs(output_folder, exist_ok=True)
    
    num_images = 0
    total_boxes = 0
    
    for r in results:
        img_name = Path(r.path).stem
        txt_path = os.path.join(output_folder, f"{img_name}.txt")
        
        with open(txt_path, 'w', encoding='utf-8') as f:
            if len(r.boxes) > 0:
                for box in r.boxes:
                    x_c, y_c, w, h = box.xywhn[0].tolist()
                    conf = box.conf[0].item()
                    cls_id = int(box.cls[0].item())
                    
                    f.write(f"{cls_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f} {conf:.6f}\n")
                    total_boxes += 1
        
        num_images += 1
        
        if num_images % 100 == 0:
            print(f"    Обработано {num_images} изображений...")
    
    print(f"    [✓] Готово! {num_images} изображений, {total_boxes} боксов")
    return True


def main():
    print("=" * 60)
    print("ПРЕДСКАЗАНИЯ НА ВАЛИДАЦИОННОМ НАБОРЕ")
    print("=" * 60)
    
    # Проверка валидационных данных
    print("\n[1] Проверка валидационных данных...")
    val_images = glob.glob(os.path.join(VAL_IMAGES_DIR, "*.jpg"))
    val_images += glob.glob(os.path.join(VAL_IMAGES_DIR, "*.png"))
    
    if len(val_images) == 0:
        print(f"  [!] Изображения не найдены в: {VAL_IMAGES_DIR}")
        return
    
    print(f"  [✓] Найдено {len(val_images)} изображений")
    
    # Проверка весов моделей
    print("\n[2] Проверка весов моделей...")
    available_models = {}
    
    for model_name, model_config in MODELS.items():
        weights_path = model_config["weights"]
        if os.path.exists(weights_path):
            print(f"  [✓] {model_name}: {weights_path}")
            available_models[model_name] = model_config
        else:
            print(f"  [!] {model_name}: веса не найдены ({weights_path})")
    
    if len(available_models) == 0:
        print("\n  [!] Нет доступных моделей для инференса!")
        print("      Проверьте пути к весам в настройках MODELS")
        return
    
    # Инференс
    print("\n[3] Запуск инференса...")
    
    for model_name, model_config in available_models.items():
        output_folder = os.path.join(OUTPUT_BASE_DIR, model_name)
        weights_path = model_config["weights"]
        model_type = model_config["type"]
        
        print(f"\n{'='*60}")
        print(f"Модель: {model_name} ({model_type})")
        print(f"{'='*60}")
        
        if model_type == "yolo":
            predict_yolo(weights_path, output_folder, CONF_THRESHOLD)
        elif model_type == "rtdetr":
            predict_rtdetr(weights_path, output_folder, CONF_THRESHOLD)
    
    # Итоговая статистика
    print("\n" + "=" * 60)
    print("ИТОГИ")
    print("=" * 60)
    
    print(f"\nПредсказания сохранены в: {os.path.abspath(OUTPUT_BASE_DIR)}")
    print("\nСтатистика по моделям:")
    
    for model_name in available_models.keys():
        model_folder = os.path.join(OUTPUT_BASE_DIR, model_name)
        txt_files = glob.glob(os.path.join(model_folder, "*.txt"))
        
        total_boxes = 0
        for txt_file in txt_files:
            with open(txt_file, 'r') as f:
                total_boxes += len(f.readlines())
        
        print(f"  {model_name}: {len(txt_files)} файлов, {total_boxes} боксов")
    
    print(f"\n[✓] Готово к запуску optimize_thresholds.py!")
    print(f"\nДля использования в optimize_thresholds.py установите:")
    print(f'  MODEL_PREDICTIONS = {[os.path.join(OUTPUT_BASE_DIR, name) for name in available_models.keys()]}')


if __name__ == "__main__":
    main()
