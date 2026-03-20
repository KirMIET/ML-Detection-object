"""
Скрипт для визуального сравнения двух submission файлов.
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import shutil


# ================= НАСТРОЙКИ =================

# Пути к submission файлам
SUBMISSION_1 = "submission_finetune_wbf_softnms.csv"
SUBMISSION_2 = "submission_full_wbf_softnms.csv"

# Папка для сохранения сравнений
OUTPUT_DIR = "submission_comparison"

# Изображения для сравнения
IMAGES_DIR = "test_images/test_images" 

# Параметры сравнения
MAX_COMPARISONS = 50  # Максимальное количество изображений для сохранения
TOP_N_DIFFERENCES = 30  # Сколько изображений с наибольшими различиями показать

# Порог для "большого различия" (разница в количестве боксов)
MIN_BOX_COUNT_DIFF = 1 # Минимальная разница в количестве боксов для попадания в выборку

# Визуализация
BOX_COLOR_1 = (255, 0, 0)      
BOX_COLOR_2 = (0, 255, 0)      
BOX_COLOR_BOTH = (255, 255, 0) 
BOX_THICKNESS = 3
FONT_SIZE = 20


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
                    'conf': box[4]
                })
        
        submissions[image_name] = boxes
    
    print(f"[✓] Загружен {submission_path}: {len(submissions)} изображений")
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


def calculate_iou(box1, box2):
    """Вычислить IoU между двумя боксами в формате [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y3 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y3 - y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0
    
    return intersection / union


def find_matching_boxes(boxes1, boxes2, img_width, img_height, iou_threshold=0.5):
    """Найти пересекающиеся боксы между двумя наборами."""
    matched1 = set()
    matched2 = set()
    
    # Конвертируем все боксы в xyxy
    boxes1_xyxy = []
    for box in boxes1:
        boxes1_xyxy.append(yolo_to_xyxy(box['xc'], box['yc'], box['w'], box['h'], img_width, img_height))
    
    boxes2_xyxy = []
    for box in boxes2:
        boxes2_xyxy.append(yolo_to_xyxy(box['xc'], box['yc'], box['w'], box['h'], img_width, img_height))
    
    # Ищем совпадения
    for i, box1 in enumerate(boxes1_xyxy):
        for j, box2 in enumerate(boxes2_xyxy):
            if j in matched2:
                continue
            
            iou = calculate_iou(box1, box2)
            if iou >= iou_threshold:
                matched1.add(i)
                matched2.add(j)
                break
    
    unmatched1 = [i for i in range(len(boxes1)) if i not in matched1]
    unmatched2 = [j for j in range(len(boxes2)) if j not in matched2]
    
    return list(matched1), list(matched2), unmatched1, unmatched2


def calculate_difference_score(boxes1, boxes2, img_width, img_height):
    """Вычислить степень различия между двумя наборами боксов."""
    count_diff = abs(len(boxes1) - len(boxes2))
    
    _, _, unmatched1, unmatched2 = find_matching_boxes(boxes1, boxes2, img_width, img_height)
    
    total_unmatched = len(unmatched1) + len(unmatched2)
    
    # Комбинированный скор различия
    diff_score = count_diff + total_unmatched
    
    return diff_score, {
        'count_diff': count_diff,
        'unmatched_count': total_unmatched,
        'boxes1_count': len(boxes1),
        'boxes2_count': len(boxes2),
        'unmatched1': unmatched1,
        'unmatched2': unmatched2
    }


def draw_boxes_on_image(image_path, boxes1, boxes2, img_width, img_height,
                        matched1, matched2, unmatched1, unmatched2, output_path):
    """Нарисовать боксы от обоих submission на одном изображении."""
    # Открываем изображение
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    
    # Попытка загрузить шрифт
    try:
        font = ImageFont.truetype("arial.ttf", FONT_SIZE)
    except:
        font = ImageFont.load_default()
    
    # Конвертируем все боксы в xyxy
    boxes1_xyxy = []
    for box in boxes1:
        boxes1_xyxy.append(yolo_to_xyxy(box['xc'], box['yc'], box['w'], box['h'], img_width, img_height))
    
    boxes2_xyxy = []
    for box in boxes2:
        boxes2_xyxy.append(yolo_to_xyxy(box['xc'], box['yc'], box['w'], box['h'], img_width, img_height))
    
    # Рисуем боксы из submission 1
    for i, box in enumerate(boxes1_xyxy):
        color = BOX_COLOR_BOTH if i in matched1 else BOX_COLOR_1
        draw.rectangle(box, outline=color, width=BOX_THICKNESS)
        
        # Подпись
        conf = boxes1[i]['conf']
        label = f"S1: {conf:.2f}"
        
        # Фон для текста
        text_bbox = draw.textbbox((box[0], box[1]), label, font=font)
        draw.rectangle(
            [text_bbox[0] - 2, text_bbox[1] - 2, text_bbox[2] + 2, text_bbox[3] + 2],
            fill=color
        )
        draw.text((box[0], box[1]), label, fill="white", font=font)
    
    # Рисуем боксы из submission 2 (со смещением, чтобы не перекрывать)
    offset = 15  # Смещение для боксов, которые есть только во втором submission
    
    for i, box in enumerate(boxes2_xyxy):
        if i in matched2:
            # Уже нарисован жёлтым из submission 1
            continue
        
        # Смещаем бокс для лучшей видимости
        shifted_box = [
            box[0] + offset,
            box[1] + offset,
            box[2] + offset,
            box[3] + offset
        ]
        
        draw.rectangle(shifted_box, outline=BOX_COLOR_2, width=BOX_THICKNESS)
        
        # Подпись
        conf = boxes2[i]['conf']
        label = f"S2: {conf:.2f}"
        
        # Фон для текста
        text_bbox = draw.textbbox((shifted_box[0], shifted_box[1]), label, font=font)
        draw.rectangle(
            [text_bbox[0] - 2, text_bbox[1] - 2, text_bbox[2] + 2, text_bbox[3] + 2],
            fill=BOX_COLOR_2
        )
        draw.text((shifted_box[0], shifted_box[1]), label, fill="white", font=font)
    
    # Добавляем легенду
    legend_y = 20
    legend_items = [
        (BOX_COLOR_1, "Только Submission 1"),
        (BOX_COLOR_2, "Только Submission 2"),
        (BOX_COLOR_BOTH, "Оба submission"),
    ]
    
    # Фон для легенды
    legend_height = len(legend_items) * 30 + 20
    draw.rectangle([0, 0, 350, legend_height], fill=(0, 0, 0, 180))
    
    for i, (color, text) in enumerate(legend_items):
        y = legend_y + i * 30
        # Образец цвета
        draw.rectangle([15, y, 35, y + 15], fill=color)
        draw.text((45, y), text, fill="white", font=font)
    
    # Добавляем статистику
    stats_y = image.height - 150
    stats_text = [
        f"Submission 1: {len(boxes1)} боксов",
        f"Submission 2: {len(boxes2)} боксов",
        f"Совпадает: {len(matched1)}",
        f"Только S1: {len(unmatched1)}",
        f"Только S2: {len(unmatched2)}",
    ]
    
    # Фон для статистики
    stats_height = len(stats_text) * 25 + 20
    draw.rectangle([0, image.height - stats_height - 10, 300, image.height - 10], fill=(0, 0, 0, 180))
    
    for i, text in enumerate(stats_text):
        y = stats_y + i * 25
        draw.text((15, y), text, fill="white", font=font)
    
    # Сохраняем
    image.save(output_path)
    print(f"  Сохранено: {output_path}")


def compare_submissions(sub1, sub2, images_dir, output_dir, max_comparisons=50, top_n=30, min_box_diff=2):
    """Сравнить два submission и сохранить визуализации различий."""
    # Создаём выходную папку
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    
    # Находим общие изображения
    common_images = set(sub1.keys()) & set(sub2.keys())
    print(f"\n[>] Общих изображений: {len(common_images)}")
    
    # Вычисляем различия для всех изображений
    print("\n[>] Вычисление различий...")
    
    differences = []
    
    for image_name in common_images:
        boxes1 = sub1[image_name]
        boxes2 = sub2[image_name]
        
        # Пытаемся найти изображение для получения размеров
        img_path = os.path.join(images_dir, image_name)
        if not os.path.exists(img_path):
            # Пробуем без расширения
            for ext in ['.jpg', '.png', '.jpeg']:
                img_path = os.path.join(images_dir, Path(image_name).stem + ext)
                if os.path.exists(img_path):
                    break
        
        img_width, img_height = 1920, 1080  # Размеры по умолчанию
        
        if os.path.exists(img_path):
            try:
                with Image.open(img_path) as img:
                    img_width, img_height = img.size
            except:
                pass
        
        diff_score, diff_info = calculate_difference_score(boxes1, boxes2, img_width, img_height)
        
        differences.append({
            'image_name': image_name,
            'diff_score': diff_score,
            'boxes1': boxes1,
            'boxes2': boxes2,
            'info': diff_info,
            'img_path': img_path if os.path.exists(img_path) else None,
            'img_width': img_width,
            'img_height': img_height
        })
    
    # Сортируем по убыванию различий
    differences.sort(key=lambda x: x['diff_score'], reverse=True)
    
    # Фильтруем только изображения с существенными различиями
    significant_diffs = [d for d in differences if d['info']['count_diff'] >= min_box_diff]
    
    print(f"\n[>] Изображений с различиями >= {min_box_diff} боксов: {len(significant_diffs)}")
    
    # Берём топ-N
    top_differences = significant_diffs[:max_comparisons]
    
    if len(top_differences) == 0:
        print("\n[!] Не найдено изображений с существенными различиями!")
        print(f"Попробуйте уменьшить MIN_BOX_COUNT_DIFF (текущий: {min_box_diff})")
        return
    
    # Сохраняем отчёт о различиях
    report_path = os.path.join(output_dir, "difference_report.csv")
    report_data = []
    
    for d in differences[:top_n]:
        report_data.append({
            'image_name': d['image_name'],
            'diff_score': d['diff_score'],
            'boxes_1_count': d['info']['boxes1_count'],
            'boxes_2_count': d['info']['boxes2_count'],
            'count_diff': d['info']['count_diff'],
            'unmatched_count': d['info']['unmatched_count'],
            'only_s1': len(d['info']['unmatched1']),
            'only_s2': len(d['info']['unmatched2']),
            'has_image': d['img_path'] is not None
        })
    
    report_df = pd.DataFrame(report_data)
    report_df.to_csv(report_path, index=False)
    print(f"\n[✓] Отчёт сохранён: {report_path}")
    
    # Визуализируем топ различий
    print(f"\n[>] Визуализация топ-{len(top_differences)} различий...")
    
    for i, diff in enumerate(top_differences):
        if diff['img_path'] is None:
            print(f"  [!] Изображение не найдено: {diff['image_name']}")
            continue
        
        image_name = diff['image_name']
        boxes1 = diff['boxes1']
        boxes2 = diff['boxes2']
        
        # Находим совпадающие и несовпадающие боксы
        matched1, matched2, unmatched1, unmatched2 = find_matching_boxes(
            boxes1, boxes2, diff['img_width'], diff['img_height']
        )
        
        # Формируем имя выходного файла
        output_filename = f"{i+1:03d}_{Path(image_name).stem}.jpg"
        output_path = os.path.join(output_dir, "images", output_filename)
        
        print(f"\n[{i+1}/{len(top_differences)}] {image_name}")
        print(f"    S1: {len(boxes1)} боксов, S2: {len(boxes2)} боксов, разница: {diff['info']['count_diff']}")
        
        draw_boxes_on_image(
            diff['img_path'],
            boxes1, boxes2,
            diff['img_width'], diff['img_height'],
            matched1, matched2, unmatched1, unmatched2,
            output_path
        )
    
    # Создаём HTML для удобного просмотра
    html_path = os.path.join(output_dir, "comparison.html")
    create_html_report(output_dir, top_differences, html_path)
    
    print(f"\n[✓] Готово! Откройте {html_path} для просмотра")


def create_html_report(output_dir, differences, html_path):
    """Создать HTML отчёт для удобного просмотра сравнений."""

    html_content = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Сравнение Submission</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background: #f5f5f5;
        }
        h1 {
            color: #333;
        }
        .legend {
            background: white;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .legend-item {
            display: inline-block;
            margin-right: 20px;
            margin-bottom: 10px;
        }
        .color-box {
            display: inline-block;
            width: 20px;
            height: 20px;
            margin-right: 8px;
            border: 2px solid #333;
        }
        .gallery {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
            gap: 20px;
        }
        .card {
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        }
        .card img {
            width: 100%;
            height: auto;
            display: block;
        }
        .card-info {
            padding: 15px;
        }
        .card-title {
            font-weight: bold;
            margin-bottom: 10px;
            color: #333;
        }
        .stats {
            font-size: 14px;
            color: #666;
        }
        .stats span {
            display: inline-block;
            margin-right: 15px;
        }
        .diff-high {
            color: #d32f2f;
            font-weight: bold;
        }
        .diff-med {
            color: #f57c00;
        }
        .diff-low {
            color: #388e3c;
        }
    </style>
</head>
<body>
    <h1>🔍 Сравнение Submission</h1>
    
    <div class="legend">
        <h3>Условные обозначения:</h3>
        <div class="legend-item">
            <span class="color-box" style="background: #FF0000;"></span>
            Только Submission 1
        </div>
        <div class="legend-item">
            <span class="color-box" style="background: #00FF00;"></span>
            Только Submission 2
        </div>
        <div class="legend-item">
            <span class="color-box" style="background: #FFFF00;"></span>
            Оба submission (совпадают)
        </div>
    </div>
    
    <div class="gallery">
"""
    
    for i, diff in enumerate(differences):
        image_name = diff['image_name']
        img_filename = f"{i+1:03d}_{Path(image_name).stem}.jpg"
        
        diff_class = "diff-high" if diff['info']['count_diff'] >= 4 else \
                     "diff-med" if diff['info']['count_diff'] >= 2 else "diff-low"
        
        html_content += f"""
        <div class="card">
            <img src="images/{img_filename}" alt="{image_name}">
            <div class="card-info">
                <div class="card-title">{image_name}</div>
                <div class="stats">
                    <span>S1: <b>{diff['info']['boxes1_count']}</b></span>
                    <span>S2: <b>{diff['info']['boxes2_count']}</b></span>
                    <span class="{diff_class}">Разница: {diff['info']['count_diff']}</span>
                </div>
                <div class="stats" style="margin-top: 8px;">
                    <span style="color: #FF0000;">Только S1: {len(diff['info']['unmatched1'])}</span>
                    <span style="color: #00FF00;">Только S2: {len(diff['info']['unmatched2'])}</span>
                </div>
            </div>
        </div>
"""
    
    html_content += """
    </div>
</body>
</html>
"""
    
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"[✓] HTML отчёт сохранён: {html_path}")


# ================= MAIN =================

if __name__ == "__main__":
    print("=" * 60)
    print("СРАВНЕНИЕ SUBMISSION ФАЙЛОВ")
    print("=" * 60)
    
    print(f"\nSubmission 1: {SUBMISSION_1}")
    print(f"Submission 2: {SUBMISSION_2}")
    print(f"Выходная папка: {OUTPUT_DIR}")
    
    # Загружаем submission
    sub1 = load_submission(SUBMISSION_1)
    sub2 = load_submission(SUBMISSION_2)
    
    # Сравниваем
    compare_submissions(
        sub1, sub2,
        images_dir=IMAGES_DIR,
        output_dir=OUTPUT_DIR,
        max_comparisons=MAX_COMPARISONS,
        top_n=TOP_N_DIFFERENCES,
        min_box_diff=MIN_BOX_COUNT_DIFF
    )
    
    print("\n" + "=" * 60)
    print("ГОТОВО!")
    print("=" * 60)
