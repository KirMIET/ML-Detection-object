import os
import glob
from pathlib import Path
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, fasterrcnn_mobilenet_v3_large_fpn
from PIL import Image
import torchvision.transforms.functional as F

# ================= НАСТРОЙКИ =================
NUM_FOLDS = 2  # Количество фолдов (увеличьте, когда их станет больше)
TEST_IMAGES_DIR = "test_images/test_images"  # ПУТЬ К ТЕСТОВЫМ КАРТИНКАМ (замените!)
OUTPUT_BASE_DIR = "predictions"          # Главная папка для результатов

# ВАЖНО: Для Faster R-CNN число классов = (ваши реальные классы + 1 для фона)
# Если у вас 5 классов объектов, ставьте NUM_CLASSES = 6
NUM_CLASSES = 2
CONF_THRESHOLD = 0.25 # Порог уверенности (сохраняем боксы с conf > 0.25)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================= ФУНКЦИИ ИНФЕРЕНСА =================

def get_faster_rcnn_model(num_classes):
    """Инициализация архитектуры Faster R-CNN MobileNet FPN v2"""
    # Загружаем архитектуру (без предобученных весов, так как загрузим свои)
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=None)
    
    # Меняем "голову" (predictor) под наше количество классов
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

def predict_faster_rcnn(weights_path, output_folder):
    """Инференс для Faster R-CNN (PyTorch) с сохранением в формате YOLO"""
    print(f"  [>] Запуск Faster R-CNN с весами: {weights_path}")
    
    model = get_faster_rcnn_model(NUM_CLASSES)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    image_paths = glob.glob(os.path.join(TEST_IMAGES_DIR, "*.*"))
    
    os.makedirs(output_folder, exist_ok=True)

    with torch.no_grad():
        for img_path in image_paths:
            img_name = Path(img_path).stem
            
            # Загружаем картинку через PIL
            image = Image.open(img_path).convert("RGB")
            img_width, img_height = image.size
            
            # Превращаем в тензор [C, H, W] со значениями 0-1
            image_tensor = F.to_tensor(image).unsqueeze(0).to(DEVICE)
            
            # Делаем предикт
            outputs = model(image_tensor)[0]
            
            txt_path = os.path.join(output_folder, f"{img_name}.txt")
            
            with open(txt_path, 'w', encoding='utf-8') as f:
                boxes = outputs['boxes'].cpu().numpy()
                scores = outputs['scores'].cpu().numpy()
                labels = outputs['labels'].cpu().numpy()
                
                for box, label, score in zip(boxes, labels, scores):
                    if score < CONF_THRESHOLD:
                        continue
                        
                    x1, y1, x2, y2 = box
                    
                    # 1. Переводим x1 y1 x2 y2 в x_center y_center width height
                    w = x2 - x1
                    h = y2 - y1
                    x_c = x1 + w / 2
                    y_c = y1 + h / 2
                    
                    # 2. Нормализуем координаты (от 0 до 1), как любит YOLO
                    x_c /= img_width
                    y_c /= img_height
                    w /= img_width
                    h /= img_height
                    
                    # 3. Сдвигаем класс. В Faster R-CNN класс 0 - это фон. 
                    # Реальные классы начинаются с 1. Делаем -1, чтобы совпало с YOLO.
                    yolo_label = label - 1 
                    
                    f.write(f"{yolo_label} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f} {score:.6f}\n")


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
        
        # 1. Faster R-CNN MobileNet V3
        # frcnn_weights = f"runs/faster_rcnn/fold_{fold}_best.pth"
        # if os.path.exists(frcnn_weights):
        #     frcnn_out = os.path.join(OUTPUT_BASE_DIR, f"faster_rcnn_fold_{fold}")
        #     predict_faster_rcnn(frcnn_weights, frcnn_out)
        # else:
        #     print(f"  [!] Веса Faster R-CNN (фолд {fold}) не найдены: {frcnn_weights}")

        # 2. YOLO11s 
        # Проверьте путь! У Ultralytics веса обычно в папке weights/
        yolo_weights = f"runs/detect/runs/ensemble_training/yolo11s_fold_{fold}/weights/best.pt" 
        if os.path.exists(yolo_weights):
            yolo_out = os.path.join(OUTPUT_BASE_DIR, f"yolo11s_fold_{fold}")
            predict_ultralytics(yolo_weights, yolo_out, model_type="yolo")
        else:
            print(f"  [!] Веса YOLO (фолд {fold}) не найдены: {yolo_weights}")

        # 3. RT-DETR
        rtdetr_weights = f"runs/detect/runs/ensemble_training/rtdetr_fold_{fold}/weights/best.pt"
        if os.path.exists(rtdetr_weights):
            rtdetr_out = os.path.join(OUTPUT_BASE_DIR, f"rtdetr_fold_{fold}")
            predict_ultralytics(rtdetr_weights, rtdetr_out, model_type="rtdetr")

if __name__ == "__main__":
    main()