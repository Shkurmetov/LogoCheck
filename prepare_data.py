import os
import random
import shutil
import xml.etree.ElementTree as ET
from tqdm import tqdm

import config

def convert_bbox(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return x * dw, y * dh, w * dw, h * dh

def prepare_detector_dataset():
    print("=" * 50)
    print("Подготовка датасета для детектора...")
    print("=" * 50)

    for split in ["train", "val"]:
        os.makedirs(os.path.join(config.DET_DATASET, "images", split), exist_ok=True)
        os.makedirs(os.path.join(config.DET_DATASET, "labels", split), exist_ok=True)

    # Собираем все XML-файлы
    samples = []
    for brand in os.listdir(config.SOURCE_DIR):
        brand_path = os.path.join(config.SOURCE_DIR, brand)
        if not os.path.isdir(brand_path):
            continue
        for f in os.listdir(brand_path):
            if f.endswith(".xml"):
                samples.append((brand_path, f))

    random.seed(config.RANDOM_SEED)
    random.shuffle(samples)

    split_idx = int(len(samples) * config.TRAIN_VAL_SPLIT)
    splits = {
        "train": samples[:split_idx],
        "val":   samples[split_idx:]
    }

    for split, files in splits.items():
        skipped = 0
        for brand_path, xml_file in tqdm(files, desc=f"  {split}"):
            xml_path = os.path.join(brand_path, xml_file)
            tree = ET.parse(xml_path)
            root = tree.getroot()

            filename = root.find("filename").text
            img_path = os.path.join(brand_path, filename)
            if not os.path.exists(img_path):
                skipped += 1
                continue

            size_node = root.find("size")
            w = int(size_node.find("width").text)
            h = int(size_node.find("height").text)

            labels = []
            for obj in root.findall("object"):
                bbox = obj.find("bndbox")
                box = (
                    float(bbox.find("xmin").text),
                    float(bbox.find("xmax").text),
                    float(bbox.find("ymin").text),
                    float(bbox.find("ymax").text),
                )
                cx, cy, bw, bh = convert_bbox((w, h), box)
                labels.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            if not labels:
                skipped += 1
                continue

            dst_img = os.path.join(config.DET_DATASET, "images", split, filename)
            dst_lbl = os.path.join(
                config.DET_DATASET, "labels", split,
                os.path.splitext(filename)[0] + ".txt"
            )
            shutil.copy(img_path, dst_img)
            with open(dst_lbl, "w") as out:
                out.write("\n".join(labels))

        if skipped:
            print(f"  Пропущено ({split}): {skipped} файлов (изображение не найдено / нет bbox)")

    # Создаём data.yaml для YOLO
    yaml_path = os.path.join(config.DET_DATASET, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(
            f"path: {config.DET_DATASET}\n"
            f"train: images/train\n"
            f"val: images/val\n\n"
            f"names:\n"
            f"  0: logo\n"
        )

    print(f"\n✔ Детекторный датасет готов: {config.DET_DATASET}")
    print(f"  train: {len(splits['train'])} | val: {len(splits['val'])}")

def make_crops():
    import cv2
    from ultralytics import YOLO

    print("\n" + "=" * 50)
    print("Вырезаем кропы логотипов для классификатора...")
    print("=" * 50)

    if not os.path.exists(config.DETECTOR_WEIGHTS):
        print(f"[ОШИБКА] Веса детектора не найдены: {config.DETECTOR_WEIGHTS}")
        print("Сначала запустите train_detector.py")
        return

    os.makedirs(config.CROPS_DIR, exist_ok=True)
    model = YOLO(config.DETECTOR_WEIGHTS)

    total_crops = 0
    for brand in tqdm(os.listdir(config.SOURCE_DIR), desc="  Бренды"):
        brand_path = os.path.join(config.SOURCE_DIR, brand)
        if not os.path.isdir(brand_path):
            continue

        brand_dir = os.path.join(config.CROPS_DIR, brand)
        os.makedirs(brand_dir, exist_ok=True)

        for img_name in os.listdir(brand_path):
            if not img_name.lower().endswith(".jpg"):
                continue

            img_path = os.path.join(brand_path, img_name)
            image = cv2.imread(img_path)
            if image is None:
                continue

            results = model(img_path, verbose=False)[0]
            for i, box in enumerate(results.boxes.xyxy):
                x1, y1, x2, y2 = map(int, box)
                crop = image[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                stem = os.path.splitext(img_name)[0]
                save_path = os.path.join(brand_dir, f"{stem}_{i}.jpg")
                cv2.imwrite(save_path, crop)
                total_crops += 1

    # Удаляем бренды без кропов
    removed = []
    for brand in os.listdir(config.CROPS_DIR):
        brand_path = os.path.join(config.CROPS_DIR, brand)
        if not os.path.isdir(brand_path):
            continue
        images = [f for f in os.listdir(brand_path)
                  if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        if len(images) == 0:
            shutil.rmtree(brand_path)
            removed.append(brand)

    print(f"\n✔ Кропов создано: {total_crops}")
    if removed:
        print(f"  Удалено пустых классов: {len(removed)}: {removed[:5]}{'...' if len(removed) > 5 else ''}")

if __name__ == "__main__":
    prepare_detector_dataset()
    make_crops()