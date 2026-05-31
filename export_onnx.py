import os
import torch
import torch.nn as nn
from torchvision import models

import config

def export_classifier():
    print("=" * 55)
    print("Экспорт классификатора EfficientNet-B0 → ONNX")
    print("=" * 55)

    for path, name in [
        (config.CLASSIFIER_WEIGHTS, "Веса классификатора"),
        (config.CLASSES_FILE,       "Файл классов"),
    ]:
        if not os.path.exists(path):
            print(f"[ОШИБКА] {name} не найден: {path}")
            print("Сначала запустите train_classifier.py")
            return None

    with open(config.CLASSES_FILE, encoding="utf-8") as f:
        classes = [line.strip() for line in f if line.strip()]
    num_classes = len(classes)

    print(f"Классов брендов: {num_classes}")
    print(f"Размер входа:    1 × 3 × {config.CLS_IMG_SIZE} × {config.CLS_IMG_SIZE}")

    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(1280, num_classes)
    model.load_state_dict(
        torch.load(config.CLASSIFIER_WEIGHTS, map_location="cpu")
    )
    model.eval()

    dummy = torch.randn(1, 3, config.CLS_IMG_SIZE, config.CLS_IMG_SIZE)

    out_path = os.path.join(config.WORK_DIR, "brand_classifier.onnx")

    torch.onnx.export(
        model,
        dummy,
        out_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={
            "image":  {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        verbose=False,
    )

    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"\n✔ Классификатор экспортирован: {out_path}")
    print(f"  Размер файла: {size_mb:.1f} МБ")
    return out_path

def export_detector():
    print("\n" + "=" * 55)
    print("Экспорт детектора YOLOv8 → ONNX")
    print("=" * 55)

    if not os.path.exists(config.DETECTOR_WEIGHTS):
        print(f"[ОШИБКА] Веса детектора не найдены: {config.DETECTOR_WEIGHTS}")
        return None

    try:
        from ultralytics import YOLO
        model = YOLO(config.DETECTOR_WEIGHTS)
        out = model.export(format="onnx", imgsz=config.DET_IMGSZ,
                           dynamic=True, simplify=True)
        print(f"✔ Детектор экспортирован: {out}")
        return out
    except Exception as e:
        print(f"[ОШИБКА] Экспорт детектора: {e}")
        return None

def validate_onnx(onnx_path: str):
    try:
        import onnx
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        print(f"  Валидация ONNX: ОК ({onnx_path})")
    except ImportError:
        print("  Установи onnx для валидации: pip install onnx")
    except Exception as e:
        print(f"  [ПРЕДУПРЕЖДЕНИЕ] Валидация: {e}")

if __name__ == "__main__":
    os.makedirs(config.WORK_DIR, exist_ok=True)

    cls_path = export_classifier()
    if cls_path:
        validate_onnx(cls_path)

    export_detector()

    print("\n  Запусти 'pip install onnxruntime' для CPU-инференса без PyTorch.")
    print("  Для GPU: 'pip install onnxruntime-gpu'")