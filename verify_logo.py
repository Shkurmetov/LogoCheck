import argparse
import os
import cv2
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models
from ultralytics import YOLO

import config


class EmbeddingExtractor(nn.Module):
    def __init__(self, weights_path, num_classes):
        super().__init__()
        base = models.efficientnet_b0(weights=None)
        base.classifier[1] = nn.Linear(1280, num_classes)
        base.load_state_dict(torch.load(weights_path, map_location="cpu"))
        self.features   = base.features
        self.avgpool    = base.avgpool
        self.classifier = base.classifier

    def forward_embed(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return x / x.norm(dim=1, keepdim=True)

    def forward_class(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.classifier(x)


def get_transform():
    return transforms.Compose([
        transforms.Resize((config.CLS_IMG_SIZE, config.CLS_IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def load_all(threshold=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for path, name in [
        (config.DETECTOR_WEIGHTS,     "Веса детектора"),
        (config.CLASSIFIER_WEIGHTS,   "Веса классификатора"),
        (config.CLASSES_FILE,         "Файл классов"),
        (config.REFERENCE_EMBEDDINGS, "Эталонные эмбеддинги"),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} не найден: {path}")

    with open(config.CLASSES_FILE, encoding="utf-8") as f:
        classes = [line.strip() for line in f if line.strip()]

    detector = YOLO(config.DETECTOR_WEIGHTS)

    model = EmbeddingExtractor(config.CLASSIFIER_WEIGHTS, len(classes))
    model.eval()
    model.to(device)

    reference = torch.load(config.REFERENCE_EMBEDDINGS, map_location=device)

    thr = threshold if threshold is not None else config.FAKE_THRESHOLD
    return detector, model, classes, reference, device, thr


def classify_crop(crop_rgb, model, classes, reference, device, threshold, transform=None):
    if transform is None:
        transform = get_transform()

    tensor = transform(Image.fromarray(crop_rgb)).unsqueeze(0).to(device)

    with torch.no_grad():
        probs    = torch.softmax(model.forward_class(tensor), dim=1)
        cls_id   = probs.argmax(1).item()
        cls_conf = probs.max().item()
        brand    = classes[cls_id]

        emb = model.forward_embed(tensor).squeeze(0)

        if brand in reference:
            centroids  = reference[brand]
            sims       = centroids @ emb
            similarity = sims.max().item()
        else:
            similarity = 0.0

    return {
        "brand":      brand,
        "cls_conf":   cls_conf,
        "similarity": similarity,
        "is_fake":    similarity < threshold,
    }


def draw_box(img_bgr, box, result):
    x1, y1, x2, y2 = box
    is_fake = result["is_fake"]
    color   = (0, 0, 210) if is_fake else (0, 200, 0)
    verdict = "FAKE" if is_fake else "REAL"
    label   = f"{result['brand']} | {verdict} | {result['similarity']:.2f}"

    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
    cv2.rectangle(img_bgr, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
    cv2.putText(img_bgr, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)


def verify_image(image_path, detector, model, classes, reference, device,
                 conf_det=0.3, threshold=None):
    if threshold is None:
        threshold = config.FAKE_THRESHOLD

    transform = get_transform()

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Не удалось открыть: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    det_results = detector(image_path, conf=conf_det, verbose=False)[0]
    results     = []

    for box in det_results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = img_rgb[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        res = classify_crop(crop, model, classes, reference, device, threshold, transform)
        res["box"] = (x1, y1, x2, y2)
        results.append(res)
        draw_box(img_bgr, (x1, y1, x2, y2), res)

    return img_bgr, results


def main():
    parser = argparse.ArgumentParser(description="Проверка логотипа на подлинность")
    parser.add_argument("--image",     type=str)
    parser.add_argument("--folder",    type=str)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--conf",      type=float, default=0.3)
    parser.add_argument("--show",      action="store_true")
    args = parser.parse_args()

    if not args.image and not args.folder:
        parser.error("Укажите --image или --folder")

    detector, model, classes, reference, device, thr = load_all(args.threshold)
    print(f"Загружено {len(classes)} брендов | порог: {thr} | устройство: {device}")

    if args.image:
        targets = [args.image]
        out_dir = None
    else:
        exts    = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        out_dir = os.path.join(args.folder, "verify_results")
        os.makedirs(out_dir, exist_ok=True)
        targets = [os.path.join(args.folder, f)
                   for f in os.listdir(args.folder)
                   if f.lower().endswith(exts)]

    for img_path in targets:
        print(f"\n── {os.path.basename(img_path)}")
        result_img, detections = verify_image(
            img_path, detector, model, classes, reference, device,
            conf_det=args.conf, threshold=thr
        )

        if not detections:
            print("  Логотипы не найдены")
        for d in detections:
            verdict = "🔴 FAKE" if d["is_fake"] else "🟢 REAL"
            print(f"  {verdict} | {d['brand']} | sim={d['similarity']:.3f} | conf={d['cls_conf']:.3f}")

        out_path = (os.path.splitext(img_path)[0] + "_verify.jpg"
                    if out_dir is None
                    else os.path.join(out_dir, os.path.basename(img_path)))
        cv2.imwrite(out_path, result_img)
        print(f"  Сохранено: {out_path}")

        if args.show:
            cv2.imshow(os.path.basename(img_path), result_img)
            cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()