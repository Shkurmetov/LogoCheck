import argparse
import os
import torch
import cv2
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn
from ultralytics import YOLO

import config

def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(config.CLASSES_FILE):
        raise FileNotFoundError(f"Файл классов не найден: {config.CLASSES_FILE}\n"
                                f"Запустите сначала train_classifier.py")
    with open(config.CLASSES_FILE, encoding="utf-8") as f:
        classes = [line.strip() for line in f if line.strip()]

    if not os.path.exists(config.DETECTOR_WEIGHTS):
        raise FileNotFoundError(f"Веса детектора не найдены: {config.DETECTOR_WEIGHTS}")
    detector = YOLO(config.DETECTOR_WEIGHTS)

    if not os.path.exists(config.CLASSIFIER_WEIGHTS):
        raise FileNotFoundError(f"Веса классификатора не найдены: {config.CLASSIFIER_WEIGHTS}")

    classifier = models.efficientnet_b0(weights=None)
    classifier.classifier[1] = nn.Linear(1280, len(classes))
    classifier.load_state_dict(torch.load(config.CLASSIFIER_WEIGHTS, map_location=device))
    classifier.eval()
    classifier.to(device)

    return detector, classifier, classes, device

def get_transform():
    return transforms.Compose([
        transforms.Resize((config.CLS_IMG_SIZE, config.CLS_IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

def detect_and_classify(image_path, detector, classifier, classes, device,
                         conf: float = 0.3):
    transform = get_transform()

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Не удалось открыть изображение: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    results = detector(image_path, conf=conf, verbose=False)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = img_rgb[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        tensor = transform(Image.fromarray(crop)).unsqueeze(0).to(device)
        with torch.no_grad():
            prob = torch.softmax(classifier(tensor), dim=1)
            cls_id = prob.argmax(1).item()
            score  = prob.max().item()

        brand = classes[cls_id]
        label = f"{brand} {score:.2f}"

        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(img_bgr, (x1, y1 - th - 6), (x1 + tw, y1), (0, 255, 0), -1)
        cv2.putText(img_bgr, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    return img_bgr

def run_on_folder(folder, detector, classifier, classes, device, conf, show):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    files = [f for f in os.listdir(folder) if f.lower().endswith(exts)]
    if not files:
        print("Изображения не найдены в папке:", folder)
        return

    out_folder = os.path.join(folder, "results")
    os.makedirs(out_folder, exist_ok=True)

    for fname in files:
        path = os.path.join(folder, fname)
        result = detect_and_classify(path, detector, classifier, classes, device, conf)
        out_path = os.path.join(out_folder, fname)
        cv2.imwrite(out_path, result)
        print(f"  Сохранено: {out_path}")

        if show:
            cv2.imshow(fname, result)
            cv2.waitKey(0)

    cv2.destroyAllWindows()
    print(f"\n✔ Готово. Результаты в: {out_folder}")

def main():
    parser = argparse.ArgumentParser(description="Logo detection + classification")
    parser.add_argument("--image",  type=str, help="Путь к одному изображению")
    parser.add_argument("--folder", type=str, help="Путь к папке с изображениями")
    parser.add_argument("--conf",   type=float, default=0.3,
                        help="Минимальный conf детектора (default: 0.3)")
    parser.add_argument("--show",   action="store_true",
                        help="Показывать результат в окне (требует GUI)")
    args = parser.parse_args()

    if not args.image and not args.folder:
        parser.error("Укажите --image или --folder")

    detector, classifier, classes, device = load_models()
    print(f"Загружено {len(classes)} классов брендов | устройство: {device}")

    if args.image:
        result = detect_and_classify(
            args.image, detector, classifier, classes, device, args.conf)
        out_path = os.path.splitext(args.image)[0] + "_result.jpg"
        cv2.imwrite(out_path, result)
        print(f"✔ Результат сохранён: {out_path}")
        if args.show:
            cv2.imshow("Result", result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    elif args.folder:
        run_on_folder(args.folder, detector, classifier, classes, device, args.conf, args.show)

if __name__ == "__main__":
    main()