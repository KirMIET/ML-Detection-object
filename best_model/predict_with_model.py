import os
import glob
from pathlib import Path
import torch
import torchvision
from PIL import Image
import torchvision.transforms.functional as F

# ================= НАСТРОЙКИ =================
NUM_FOLDS = 5  
TEST_IMAGES_DIR = "test_images/test_images" 
OUTPUT_BASE_DIR = "predictions"         

NUM_CLASSES = 2
CONF_THRESHOLD = 0.25 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================= ФУНКЦИИ ИНФЕРЕНСА ================

def predict_ultralytics(weights_path, output_folder, model_type="yolo"):
    """Инференс для YOLO / RT-DETR с ручным сохранением (как в Faster R-CNN)"""
    print(f"  [>] Запуск {model_type.upper()} с весами: {weights_path}")
    try:
        if model_type == "yolo":
            from ultralytics import YOLO
            model = YOLO(weights_path)
        else:
            from ultralytics import RTDETR
            model = RTDETR(weights_path)
    except ImportError:
        print("      [!] Библиотека ultralytics не установлена. Пропускаем.")
        return

    os.makedirs(output_folder, exist_ok=True)

    # Делаем предикт, но НЕ просим ultralytics сохранять txt файлы
    results = model.predict(
        source=TEST_IMAGES_DIR,
        conf=CONF_THRESHOLD,
        device=0 if torch.cuda.is_available() else "cpu",
        verbose=False,
        stream=True,     # Оставляем stream=True для экономии памяти
        augment=True
    )

    # Вручную итерируемся по результатам и сохраняем
    for r in results:
        img_name = Path(r.path).stem
        txt_path = os.path.join(output_folder, f"{img_name}.txt")
        
        # Открываем файл. Если детекций нет, создастся пустой файл.
        with open(txt_path, 'w', encoding='utf-8') as f:
            # Проверяем, есть ли найденные боксы
            if len(r.boxes) > 0:
                for box in r.boxes:
                    # Ultralytics уже может выдать нормализованные xywh (xywhn)
                    x_c, y_c, w, h = box.xywhn[0].tolist() 
                    conf = box.conf[0].item()
                    cls_id = int(box.cls[0].item())
                    
                    f.write(f"{cls_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f} {conf:.6f}\n")


def main():
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    
    for fold in range(1, NUM_FOLDS + 1):
        print(f"\n{'='*20} ОБРАБОТКА ФОЛДА {fold} {'='*20}")

        # YOLO11m
        yolo_weights = f"runs/detect/runs/ensemble_training/yolo11m_fold_{fold}/weights/best.pt" 
        if os.path.exists(yolo_weights):
            yolo_out = os.path.join(OUTPUT_BASE_DIR, f"yolo11m_fold_{fold}")
            predict_ultralytics(yolo_weights, yolo_out, model_type="yolo")
        else:
            print(f"  [!] Веса YOLO (фолд {fold}) не найдены: {yolo_weights}")

        # RT-DETR
        rtdetr_weights = f"runs/detect/runs/ensemble_training/rtdetr_fold_{fold}/weights/best.pt"
        if os.path.exists(rtdetr_weights):
            rtdetr_out = os.path.join(OUTPUT_BASE_DIR, f"rtdetr_fold_{fold}")
            predict_ultralytics(rtdetr_weights, rtdetr_out, model_type="rtdetr")

if __name__ == "__main__":
    main()