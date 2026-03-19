import shutil
import random
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import yaml


SOURCE_IMAGES_DIR = Path(r"yolo_dataset_2/add_images")
SOURCE_LABELS_DIR = Path(r"yolo_dataset_2/add_labels")

DEST_DIR = Path(r"dataset_add")

TRAIN_RATIO = 0.8
IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}


def main():
    if not SOURCE_IMAGES_DIR.exists() or not SOURCE_LABELS_DIR.exists():
        print(f"Ошибка: Не найдены исходные папки:\n{SOURCE_IMAGES_DIR}\n{SOURCE_LABELS_DIR}")
        return

    for split in ['train', 'val']:
        (DEST_DIR / 'images' / split).mkdir(parents=True, exist_ok=True)
        (DEST_DIR / 'labels' / split).mkdir(parents=True, exist_ok=True)

    print("Сканирование файлов...")
    all_images = [
        p for p in SOURCE_IMAGES_DIR.iterdir() 
        if p.suffix.lower() in IMG_EXTENSIONS
    ]

    shop_groups = defaultdict(list)
    
    for img_path in all_images:
        stem = img_path.stem
        try:
            shop_id = stem.split('_')[0]
        except IndexError:
            shop_id = "unknown" 
            
        shop_groups[shop_id].append(img_path)

    print(f"Найдено изображений: {len(all_images)}")
    print(f"Найдено уникальных магазинов (групп): {len(shop_groups)}")

    # Разбиваем на train/val внутри каждой группы
    train_imgs = []
    val_imgs = []

    for shop_id, images in shop_groups.items():
        random.shuffle(images)
        
        count = len(images)
        split_idx = int(count * TRAIN_RATIO)
        
        if count == 1:
            split_idx = 1
        
        train_imgs.extend(images[:split_idx])
        val_imgs.extend(images[split_idx:])

    print(f"Итого в Train: {len(train_imgs)}")
    print(f"Итого в Val:   {len(val_imgs)}")

    def copy_files(image_list, split_name):
        print(f"Копирование {split_name}...")
        for img_path in tqdm(image_list):
            # Исходный путь лейбла
            label_name = img_path.stem + ".txt"
            src_label = SOURCE_LABELS_DIR / label_name
            
            # Целевые пути
            dst_img = DEST_DIR / 'images' / split_name / img_path.name
            dst_label = DEST_DIR / 'labels' / split_name / label_name
            
            # Копируем картинку
            shutil.copy2(img_path, dst_img)
            

            if src_label.exists():
                shutil.copy2(src_label, dst_label)
            else:
                pass 

    copy_files(train_imgs, 'train')
    copy_files(val_imgs, 'val')


    yaml_content = {
        'path': str(DEST_DIR.absolute()), 
        'train': 'images/train',
        'val': 'images/val',
        'nc': 2, 
        'names': ['customer', 'employee'] 
    }

    yaml_path = DEST_DIR / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False)

    print(f"\nГотово! Датасет создан в: {DEST_DIR.absolute()}")
    print(f"Не забудь проверить classes в созданном data.yaml!")

if __name__ == "__main__":
    main()