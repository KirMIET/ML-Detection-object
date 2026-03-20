import cv2
import os
from pathlib import Path

# Конфигурация
SOURCE_IMAGES = 'yolo_dataset/yolo_dataset/train/images'
OUTPUT_FOLDER = 'classification_dataset/Other'
DISPLAY_SCALE = 0.7 

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Глобальные переменные 
ref_pt = []
drawing = False
img_display = None
img_original = None
current_img_name = ""
crop_count = 0

def click_and_crop(event, x, y, flags, param):
    global ref_pt, drawing, img_display, crop_count

    if event == cv2.EVENT_LBUTTONDOWN:
        ref_pt = [(x, y)]
        drawing = True

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            temp_img = img_display.copy()
            cv2.rectangle(temp_img, ref_pt[0], (x, y), (0, 255, 0), 2)
            cv2.imshow("Labeler", temp_img)

    elif event == cv2.EVENT_LBUTTONUP:
        ref_pt.append((x, y))
        drawing = False
        
        # Рисуем финальный прямоугольник
        cv2.rectangle(img_display, ref_pt[0], ref_pt[1], (255, 0, 0), 2)
        cv2.imshow("Labeler", img_display)

        # Вычисляем координаты для оригинального (полного) изображения
        x1, y1 = ref_pt[0]
        x2, y2 = ref_pt[1]
        
        # Сортируем координаты (чтобы можно было тянуть в любую сторону)
        ix1, ix2 = sorted([x1, x2])
        iy1, iy2 = sorted([y1, y2])

        # Масштабируем обратно к оригиналу
        ox1, oy1 = int(ix1 / DISPLAY_SCALE), int(iy1 / DISPLAY_SCALE)
        ox2, oy2 = int(ix2 / DISPLAY_SCALE), int(iy2 / DISPLAY_SCALE)

        # Вырезаем из оригинала
        crop = img_original[oy1:oy2, ox1:ox2]
        
        if crop.size > 0:
            crop_count += 1
            crop_name = f"{current_img_name}_other_{crop_count}.jpg"
            save_path = os.path.join(OUTPUT_FOLDER, crop_name)
            cv2.imwrite(save_path, crop)
            print(f"Сохранено: {crop_name}")

def main():
    global img_display, img_original, current_img_name, crop_count
    
    image_paths = sorted(list(Path(SOURCE_IMAGES).glob('*.jpg')))
    
    cv2.namedWindow("Labeler")
    cv2.setMouseCallback("Labeler", click_and_crop)

    print("-" * 30)
    print("ИНСТРУКЦИЯ:")
    print("1. Выделяйте прямоугольник МЫШКОЙ (ЛКМ)")
    print("2. 'N' - следующее изображение")
    print("3. 'Q' - выход")
    print("4. 'R' - сбросить выделение на текущем фото")
    print("-" * 30)

    for img_path in image_paths:
        img_original = cv2.imread(str(img_path))
        if img_original is None: continue
        
        current_img_name = img_path.stem
        crop_count = 0
        
        # Масштабируем для удобства экрана
        h, w = img_original.shape[:2]
        img_display = cv2.resize(img_original, (int(w * DISPLAY_SCALE), int(h * DISPLAY_SCALE)))
        clone = img_display.copy()

        while True:
            cv2.imshow("Labeler", img_display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("r"): # Сброс
                img_display = clone.copy()
            elif key == ord("n"): # Next
                break
            elif key == ord("q"): # Quit
                print("Выход...")
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()