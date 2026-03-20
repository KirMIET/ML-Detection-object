"""
Скрипт для визуализации предсказаний для конкретного изображения.
"""

import os
import json
import pandas as pd
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


# ================= НАСТРОЙКИ =================

# Пути
SUBMISSION_FILE = "submission_finetune_wbf_softnms.csv"  # Submission файл
IMAGES_DIR = "test_images/test_images"  # Папка с изображениями
OUTPUT_DIR = "visualization_output"  # Папка для сохранения результатов

# Визуализация
BOX_COLOR = (255, 0, 0)  # Красный цвет боксов
BOX_THICKNESS = 3
FONT_SIZE = 24
SHOW_CLASS = True  # Показывать класс на боксе
SHOW_CONF = True   # Показывать уверенность на боксе

# Классы (измените под ваш датасет)
CLASS_NAMES = {
    0: "customer",
    1: "employee"
}


# ================= ФУНКЦИИ =================

def load_submission(submission_path):
    """Загрузить submission и распарсить боксы."""
    if not os.path.exists(submission_path):
        raise FileNotFoundError(f"Submission не найден: {submission_path}")

    df = pd.read_csv(submission_path)
    
    submissions = {}
    
    for _, row in df.iterrows():
        image_name = row['image_name']
        boxes_str = row['boxes']
        
        if pd.isna(boxes_str) or boxes_str == '[]':
            submissions[image_name] = []
            continue
        
        try:
            boxes_list = json.loads(boxes_str)
        except json.JSONDecodeError:
            submissions[image_name] = []
            continue
        
        # Конвертируем в удобный формат
        boxes = []
        for box in boxes_list:
            if len(box) >= 5:
                boxes.append({
                    'xc': box[0],
                    'yc': box[1],
                    'w': box[2],
                    'h': box[3],
                    'conf': box[4],
                    # Если есть класс в submission (опционально)
                    'cls': box[5] if len(box) > 5 else 0
                })
        
        submissions[image_name] = boxes
    
    return submissions


def yolo_to_xyxy(xc, yc, w, h, img_width, img_height):
    """Конвертация нормализованных YOLO координат в пиксели."""
    x1 = int((xc - w / 2) * img_width)
    y1 = int((yc - h / 2) * img_height)
    x2 = int((xc + w / 2) * img_width)
    y2 = int((yc + h / 2) * img_height)
    
    # Ограничиваем размерами изображения
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_width, x2)
    y2 = min(img_height, y2)
    
    return [x1, y1, x2, y2]


def draw_boxes(image_path, boxes, output_path, class_names=None):
    """Нарисовать боксы на изображении."""
    # Открываем изображение
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    img_width, img_height = image.size
    
    # Попытка загрузить шрифт
    try:
        font = ImageFont.truetype("arial.ttf", FONT_SIZE)
    except:
        font = ImageFont.load_default()
    
    print(f"\nИзображение: {image_path}")
    print(f"Размер: {img_width}x{img_height}")
    print(f"Найдено боксов: {len(boxes)}")
    
    # Рисуем каждый бокс
    for i, box_data in enumerate(boxes):
        # Конвертируем в xyxy
        x1, y1, x2, y2 = yolo_to_xyxy(
            box_data['xc'], box_data['yc'],
            box_data['w'], box_data['h'],
            img_width, img_height
        )
        
        # Рисуем рамку
        draw.rectangle([x1, y1, x2, y2], outline=BOX_COLOR, width=BOX_THICKNESS)
        
        # Формируем подпись
        labels = []
        
        if SHOW_CLASS:
            cls_id = int(box_data.get('cls', 0))
            cls_name = class_names.get(cls_id, f"class {cls_id}") if class_names else f"class {cls_id}"
            labels.append(cls_name)
        
        if SHOW_CONF:
            conf = box_data['conf']
            labels.append(f"{conf:.2f}")
        
        label = " | ".join(labels)
        
        # Рисуем фон для текста
        text_bbox = draw.textbbox((x1, y1), label, font=font)
        draw.rectangle(
            [x1, y1 - text_bbox[3] + text_bbox[1] - 4, 
             x1 + text_bbox[2] - text_bbox[0] + 4, y1],
            fill=BOX_COLOR
        )
        
        # Рисуем текст
        draw.text((x1 + 2, y1 - 2), label, fill="white", font=font)
        
        print(f"  Бокс {i+1}: [{box_data['xc']:.3f}, {box_data['yc']:.3f}, {box_data['w']:.3f}, {box_data['h']:.3f}] conf={box_data['conf']:.3f}")
    
    # Сохраняем
    image.save(output_path)
    print(f"\n✓ Сохранено: {output_path}")


def find_image_in_dir(image_name, images_dir):
    """Найти изображение по имени (с разными расширениями)."""
    # Проверяем точное совпадение
    full_path = os.path.join(images_dir, image_name)
    if os.path.exists(full_path):
        return full_path
    
    # Пробуем добавить расширения
    for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
        # Если имя уже с расширением, пробуем заменить
        stem = Path(image_name).stem
        test_path = os.path.join(images_dir, stem + ext)
        if os.path.exists(test_path):
            return test_path
    
    return None


def list_available_images(submissions, images_dir):
    """Показать доступные изображения."""
    print("\n" + "=" * 60)
    print("ДОСТУПНЫЕ ИЗОБРАЖЕНИЯ В SUBMISSION:")
    print("=" * 60)
    
    image_names = sorted(submissions.keys())
    
    # Проверяем, какие изображения существуют
    existing = []
    missing = []
    
    for name in image_names:
        path = find_image_in_dir(name, images_dir)
        if path:
            existing.append(name)
        else:
            missing.append(name)
    
    print(f"\nВсего в submission: {len(image_names)}")
    print(f"Существуют на диске: {len(existing)}")
    print(f"Отсутствуют: {len(missing)}")
    
    if existing:
        print("\nПримеры доступных изображений:")
        for name in existing[:10]:
            print(f"  - {name}")
        if len(existing) > 10:
            print(f"  ... и ещё {len(existing) - 10}")
    
    return existing


# ================= MAIN =================

def main():
    print("=" * 60)
    print("ВИЗУАЛИЗАЦИЯ ПРЕДСКАЗАНИЙ ДЛЯ КОНКРЕТНОГО ИЗОБРАЖЕНИЯ")
    print("=" * 60)
    
    # Создаём выходную папку
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Загружаем submission
    print(f"\n[1] Загрузка submission из {SUBMISSION_FILE}...")
    try:
        submissions = load_submission(SUBMISSION_FILE)
        print(f"✓ Загружено {len(submissions)} изображений")
    except FileNotFoundError as e:
        print(f"✗ Ошибка: {e}")
        return
    
    # Показываем доступные изображения
    available = list_available_images(submissions, IMAGES_DIR)
    
    if not available:
        print("\n✗ Нет доступных изображений!")
        return
    
    # Запрос имени изображения у пользователя
    print("\n" + "=" * 60)
    print("ВВЕДИТЕ ИМЯ ИЗОБРАЖЕНИЯ ДЛЯ ВИЗУАЛИЗАЦИИ")
    print("=" * 60)
    print("\nПодсказка: можно ввести полное имя (с расширением) или только название")
    print("Пример: 'image_001.jpg' или 'image_001'")
    print("\nДля выхода введите 'q' или 'exit'")
    
    while True:
        print("\n" + "-" * 40)
        image_name = input("\nИмя изображения: ").strip()
        
        if not image_name:
            print("⚠ Введите имя изображения")
            continue
        
        if image_name.lower() in ['q', 'quit', 'exit', 'выход']:
            print("\nВыход из программы...")
            break
        
        # Ищем изображение
        image_path = find_image_in_dir(image_name, IMAGES_DIR)
        
        if not image_path:
            print(f"✗ Изображение не найдено: {image_name}")
            print("  Проверьте имя или посмотрите список доступных выше")
            continue
        
        # Проверяем, есть ли предсказания
        if image_name not in submissions:
            # Пробуем найти по stem
            stem = Path(image_name).stem
            found = False
            for key in submissions.keys():
                if Path(key).stem == stem:
                    image_name = key
                    found = True
                    break
            
            if not found:
                print(f"✗ Нет предсказаний для: {image_name}")
                continue
        
        boxes = submissions[image_name]
        
        if len(boxes) == 0:
            print(f"⚠ Для изображения нет предсказаний (пустой submission)")
            proceed = input("Продолжить без боксов? (y/n): ").strip().lower()
            if proceed != 'y':
                continue
        
        # Визуализируем
        output_filename = f"viz_{Path(image_name).stem}.jpg"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        print("\n" + "=" * 60)
        draw_boxes(image_path, boxes, output_path, CLASS_NAMES)
        print("=" * 60)
        
        # Предлагаем открыть изображение
        if os.name == 'nt':  # Windows
            open_img = input("\nОткрыть изображение? (y/n): ").strip().lower()
            if open_img == 'y':
                os.startfile(output_path)
                print(f"✓ Открыто: {output_path}")


if __name__ == "__main__":
    main()
