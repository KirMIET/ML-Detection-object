"""
Диагностика submission файла.

Проверяет:
1. Количество строк в submission
2. Количество детекций
3. Распределение confidence
4. Формат боксов
"""

import pandas as pd
import json
import os

SUBMISSION_FILE = "submission_wbf_softnms.csv"

def diagnose_submission():
    if not os.path.exists(SUBMISSION_FILE):
        print(f"[!] Файл {SUBMISSION_FILE} не найден!")
        print("    Сначала запустите sub_create_wbf.py")
        return
    
    print(f"Чтение {SUBMISSION_FILE}...")
    df = pd.read_csv(SUBMISSION_FILE)
    
    print(f"\n{'='*60}")
    print(f"ДИАГНОСТИКА SUBMISSION")
    print(f"{'='*60}")
    
    # Общая статистика
    print(f"\n[1] Общая статистика:")
    print(f"    Всего строк: {len(df)}")
    print(f"    Колонки: {list(df.columns)}")
    
    # Статистика детекций
    print(f"\n[2] Статистика детекций:")
    
    total_boxes = 0
    boxes_per_image = []
    all_confidences = []
    all_classes = []
    images_with_boxes = 0
    
    for idx, row in df.iterrows():
        boxes_str = row['boxes']
        
        if pd.isna(boxes_str) or boxes_str == '':
            boxes = []
        else:
            try:
                boxes = json.loads(boxes_str)
            except:
                boxes = []
        
        boxes_per_image.append(len(boxes))
        total_boxes += len(boxes)
        
        if len(boxes) > 0:
            images_with_boxes += 1
        
        for box in boxes:
            # Ожидаемый формат submission: [xc, yc, w, h, conf] (5 элементов, БЕЗ класса!)
            if len(box) == 5:
                # Правильный формат
                xc, yc, w, h, conf = box
                all_confidences.append(conf)
                all_classes.append(0)  # Класс не указан, предполагаем 0
            elif len(box) == 6:
                # Формат с классом [cls, xc, yc, w, h, conf]
                cls, xc, yc, w, h, conf = box
                all_confidences.append(conf)
                all_classes.append(int(cls))
            else:
                format_errors += 1
                continue
            
            # Проверка нормализации
            if xc < 0 or xc > 1 or yc < 0 or yc > 1:
                out_of_range += 1
            if w < 0 or w > 1 or h < 0 or h > 1:
                out_of_range += 1
    
    print(f"    Всего боксов: {total_boxes}")
    print(f"    Изображений с детекциями: {images_with_boxes}/{len(df)}")
    print(f"    Изображений без детекций: {len(df) - images_with_boxes}")
    print(f"    Среднее боксов на изображение: {total_boxes / len(df):.2f}")
    print(f"    Медиана боксов на изображение: {sorted(boxes_per_image)[len(boxes_per_image)//2]}")
    
    # Статистика confidence
    if len(all_confidences) > 0:
        print(f"\n[3] Статистика confidence:")
        print(f"    Мин: {min(all_confidences):.4f}")
        print(f"    Макс: {max(all_confidences):.4f}")
        print(f"    Среднее: {sum(all_confidences)/len(all_confidences):.4f}")
        print(f"    Медиана: {sorted(all_confidences)[len(all_confidences)//2]:.4f}")
    
    # Статистика классов
    if len(all_classes) > 0:
        print(f"\n[4] Распределение классов:")
        class_counts = {}
        for cls in all_classes:
            class_counts[cls] = class_counts.get(cls, 0) + 1
        
        for cls, count in sorted(class_counts.items()):
            pct = count / len(all_classes) * 100
            print(f"    Класс {cls}: {count} ({pct:.1f}%)")
    
    # Проверка формата боксов
    print(f"\n[5] Проверка формата боксов:")
    format_errors = 0
    out_of_range = 0
    
    for idx, row in df.iterrows():
        boxes_str = row['boxes']
        if pd.isna(boxes_str) or boxes_str == '':
            continue
        
        try:
            boxes = json.loads(boxes_str)
        except:
            format_errors += 1
            continue
        
        for box in boxes:
            # Ожидаемый формат submission: [xc, yc, w, h, conf] (5 элементов)
            if len(box) != 5:
                format_errors += 1
                continue
            
            xc, yc, w, h, conf = box
            
            # Проверка нормализации
            if xc < 0 or xc > 1 or yc < 0 or yc > 1:
                out_of_range += 1
            if w < 0 or w > 1 or h < 0 or h > 1:
                out_of_range += 1
    
    print(f"    Ошибок формата: {format_errors}")
    print(f"    Боксов вне диапазона [0,1]: {out_of_range}")
    
    if format_errors == 0:
        print(f"    [✓] Все боксы имеют правильный формат [xc, yc, w, h, conf]")
    
    # Примеры
    print(f"\n[6] Примеры детекций (первые 5 изображений с боксами):")
    count = 0
    for idx, row in df.iterrows():
        boxes_str = row['boxes']
        if pd.isna(boxes_str) or boxes_str == '':
            continue
        
        try:
            boxes = json.loads(boxes_str)
        except:
            continue
        
        if len(boxes) > 0:
            img_name = row['image_name']
            print(f"    {img_name}: {len(boxes)} боксов")
            for i, box in enumerate(boxes[:2]):  # Показываем первые 2 бокса
                print(f"      [{i}] cls={box[0]}, conf={box[4]:.3f}, box=[{box[1]:.3f}, {box[2]:.3f}, {box[3]:.3f}, {box[4]:.3f}]")
            
            count += 1
            if count >= 5:
                break
    
    print(f"\n{'='*60}")
    
    # Рекомендации
    print(f"\n[7] Рекомендации:")
    
    if total_boxes == 0:
        print(f"    [!] КРИТИЧНО: Нет детекций!")
        print(f"        Проверьте MODEL_DIRS в sub_create_wbf.py")
        print(f"        Убедитесь, что папки с предсказаниями существуют")
    
    if images_with_boxes < len(df) * 0.5:
        print(f"    [!] Меньше 50% изображений имеют детекции")
        print(f"        Попробуйте снизить CONF_THR_FINAL")
        print(f"        Проверьте, что предсказания существуют для всех изображений")
    
    if len(all_confidences) > 0 and min(all_confidences) > 0.5:
        print(f"    [!] Высокий минимальный confidence ({min(all_confidences):.3f})")
        print(f"        Возможно, CONF_THR_FINAL слишком высокий")
    
    if out_of_range > 0:
        print(f"    [!] {out_of_range} боксов вне нормализованного диапазона")
        print(f"        Проверьте конвертацию координат")


if __name__ == "__main__":
    diagnose_submission()
