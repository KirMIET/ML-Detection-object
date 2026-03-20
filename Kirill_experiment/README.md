# ML-Detection-object

Проект по детекции людей (customer/employee) на изображениях с использованием ансамбля моделей.

## Архитектура

### Модели

Обучаются 3 типа моделей:

1. **YOLO11m** — 100 эпох, imgsz=768, усиленные аугментации
2. **YOLO26m** — 100 эпох, imgsz=768, усиленные аугментации  
3. **RT-DETR** — 130 эпох, imgsz=640, gradient clipping

**train_model_v2.ipynb** — основной ноутбук для обучения всех моделей:
- 2-х фолдовая кросс-валидация на аугментированном датасете
- Усиленные аугментации: mosaic=0.6, mixup=0.25, cutmix=0.15, copy_paste=0.3
- Геометрические трансформации: degrees=35°, translate=0.25, scale=0.3, shear=15°
- Cosine learning rate scheduler
- Early stopping с patience=10-15
- Автоматическая очистка памяти между эпохами

### Ансамбль

**sub_create_wbf.py** — создание submission через ансамбль моделей:
- Weighted Box Fusion (WBF) для слияния боксов от разных моделей
- Soft-NMS для удаления дубликатов
- Индивидуальные веса для каждой модели
- Оптимизированные пороги через grid search

**optimize_wbf_softnms.py** — оптимизация параметров ансамбля на валидации:
- Подбор весов моделей, IoU thresholds, confidence thresholds
- Grid search с оценкой по mAP@0.5:0.95

## Структура проекта

### Основные скрипты

| Скрипт | Описание |
|--------|----------|
| `train_model_v2.ipynb` | Обучение 3 моделей (YOLO11m, YOLO26m, RT-DETR) с 2-х фолдовой кросс-валидацией |
| `finetune_on_add_data.ipynb` | Дообучение моделей на дополнительных данных |
| `ensemble_model.ipynb` | Создание ансамбля моделей через WBF |
| `main.ipynb` | Базовое обучение и инференс моделей |

### Скрипты для работы с данными

| Скрипт | Описание |
|--------|----------|
| `augment_dataset.py` | Аугментация датасета: multi-scale кропы, flip, color jitter, copy-paste |
| `split_add_data.py` | Разбиение дополнительных данных на 2 фолда с кропами |
| `create_finetune_yaml.py` | Создание data.yaml для фолдов дополнительных данных |

### Скрипты для инференса

| Скрипт | Описание |
|--------|----------|
| `predict_with_tta.py` | Инференс с TTA (multi-scale, flip, SAHI) |
| `predict_val.py` | Получение предсказаний на валидации для оптимизации |

### Скрипты для оптимизации

| Скрипт | Описание |
|--------|----------|
| `optimize_thresholds.py` | Grid search порогов и весов моделей на валидации |
| `optimize_wbf_softnms.py` | Оптимизация параметров WBF + Soft-NMS |

### Скрипты для создания submission

| Скрипт | Описание |
|--------|----------|
| `sub_create_wbf.py` | Создание submission через WBF + Soft-NMS ансамбль |

### Скрипты для анализа

| Скрипт | Описание |
|--------|----------|
| `compare_submissions.py` | Визуальное сравнение двух submission с подсветкой различий |
| `visualize_predictions.py` | Визуализация предсказаний для конкретного изображения |


## Конфигурация

### Параметры аугментаций

```python
# Геометрические
degrees = 35.0
translate = 0.25
scale = 0.3
shear = 15.0

# Advanced
mosaic = 0.6
mixup = 0.25
cutmix = 0.15
copy_paste = 0.3
```

### Параметры ансамбля

```python
# Веса моделей (оптимизируются через grid search)
MODEL_WEIGHTS = [1.2, 1.2, 1.2, 1.2, 1.0, 1.0]

# WBF параметры
IOU_THR = 0.6
SKIP_BOX_THR = 0.01

# Soft-NMS параметры
SOFT_NMS_IOU_THR = 0.45
SOFT_NMS_SIGMA = 0.5
CONF_THR_FINAL = 0.30
```

## 📝 Зависимости

```
ultralytics>=8.4.0
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
pandas>=2.0.0
numpy>=1.24.0
ensemble-boxes>=1.0.4
Pillow>=9.0.0
```
