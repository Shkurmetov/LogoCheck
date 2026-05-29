import os
import torch
from ultralytics import YOLO

import config

def train():
    print("=" * 50)
    print("Обучение детектора YOLOv8")
    print("=" * 50)

    yaml_path = os.path.join(config.DET_DATASET, "data.yaml")
    if not os.path.exists(yaml_path):
        print(f"[ОШИБКА] data.yaml не найден: {yaml_path}")
        print("Сначала запустите prepare_data.py")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Устройство: {device}")

    model = YOLO("yolov8n.pt")

    model.train(
        data=yaml_path,
        epochs=config.DET_EPOCHS,
        imgsz=config.DET_IMGSZ,
        batch=config.DET_BATCH,
        device=device,
        lr0=config.DET_LR,
        mosaic=1.0,
        mixup=0.1,
        patience=config.DET_PATIENCE,
        project=os.path.join(config.WORK_DIR, "runs", "detect"),
        name="logo_detector",
        seed=config.RANDOM_SEED,
    )

    print(f"\n✔ Обучение завершено. Веса: {config.DETECTOR_WEIGHTS}")
    print("Следующий шаг: запустите prepare_data.py -> make_crops()")

if __name__ == "__main__":
    train()