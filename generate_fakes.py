import argparse
import json
import os
import random
import shutil

import cv2
import numpy as np
from tqdm import tqdm

import config
from verify_logo import load_all, classify_crop, get_transform



def fake_blur(img: np.ndarray) -> np.ndarray:
    k = random.choice([9, 13, 17, 21])
    return cv2.GaussianBlur(img, (k, k), 0)


def fake_hsv(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int32)
    shift = random.randint(40, 100)
    hsv[:, :, 0] = (hsv[:, :, 0] + shift) % 180
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def fake_noise(img: np.ndarray) -> np.ndarray:
    sigma = random.randint(30, 60)
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    noisy = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy


def fake_mirror(img: np.ndarray) -> np.ndarray:
    return cv2.flip(img, random.choice([0, 1, -1]))


def fake_overexpose(img: np.ndarray) -> np.ndarray:
    factor = random.uniform(1.6, 2.5)
    bright = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    return bright


def fake_combined(img: np.ndarray) -> np.ndarray:
    img = fake_hsv(img)
    img = fake_noise(img)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    return img


TRANSFORMS = {
    "blur":       fake_blur,
    "hsv":        fake_hsv,
    "noise":      fake_noise,
    "mirror":     fake_mirror,
    "overexpose": fake_overexpose,
    "combined":   fake_combined,
}



def collect_crops(crops_dir: str, brands: list | None, n_per_brand: int) -> list[str]:
    paths = []
    if not os.path.isdir(crops_dir):
        raise FileNotFoundError(f"CROPS_DIR не найден: {crops_dir}")

    for brand in os.listdir(crops_dir):
        if brands and brand not in brands:
            continue
        brand_dir = os.path.join(crops_dir, brand)
        if not os.path.isdir(brand_dir):
            continue
        imgs = [os.path.join(brand_dir, f)
                for f in os.listdir(brand_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        random.shuffle(imgs)
        paths.extend(imgs[:n_per_brand])

    random.shuffle(paths)
    return paths


def load_image_bgr(path: str) -> np.ndarray | None:
    img = cv2.imread(path)
    return img if img is not None and img.size > 0 else None


def annotate_image(img_bgr: np.ndarray, label: str,
                   similarity: float, is_fake: bool) -> np.ndarray:
    out = img_bgr.copy()
    color = (0, 0, 210) if is_fake else (0, 200, 0)
    verdict = "FAKE ✓" if is_fake else "REAL ✗"
    text = f"{label} | {verdict} | sim={similarity:.2f}"
    cv2.rectangle(out, (0, 0), (out.shape[1]-1, out.shape[0]-1), color, 2)
    cv2.putText(out, text, (4, out.shape[0] - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return out



def compute_metrics(y_true: list, y_scores: list,
                    threshold: float) -> dict:
    tp = fp = tn = fn = 0
    for gt, sim in zip(y_true, y_scores):
        pred_fake = int(sim < threshold)
        if gt == 1 and pred_fake == 1: tp += 1
        elif gt == 0 and pred_fake == 1: fp += 1
        elif gt == 0 and pred_fake == 0: tn += 1
        else: fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    accuracy  = (tp + tn) / len(y_true) if y_true else 0.0
    fpr       = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return dict(tp=tp, fp=fp, tn=tn, fn=fn,
                precision=round(precision, 4),
                recall=round(recall, 4),
                f1=round(f1, 4),
                accuracy=round(accuracy, 4),
                fpr=round(fpr, 4))


def compute_roc(y_true: list, y_scores: list) -> tuple[list, list, list]:
    thresholds = sorted(set(y_scores), reverse=True)
    fprs, tprs = [], []
    for thr in thresholds:
        m = compute_metrics(y_true, y_scores, thr)
        fprs.append(m["fpr"])
        tprs.append(m["recall"])
    return fprs, tprs, thresholds


def find_best_threshold(y_true: list, y_scores: list) -> tuple[float, float]:
    fprs, tprs, thresholds = compute_roc(y_true, y_scores)
    best_j = -1.0
    best_t = thresholds[0] if thresholds else 0.5
    for fpr, tpr, t in zip(fprs, tprs, thresholds):
        j = tpr - fpr
        if j > best_j:
            best_j, best_t = j, t
    return round(best_t, 4), round(best_j, 4)


def save_roc_png(fprs: list, tprs: list, best_thr: float,
                 auc: float, out_path: str):
    W, H, PAD = 640, 480, 60
    img = np.full((H, W, 3), 18, dtype=np.uint8)

    cv2.rectangle(img, (PAD, PAD), (W-PAD, H-PAD), (40, 40, 40), -1)
    cv2.line(img, (PAD, H-PAD), (W-PAD, H-PAD), (80, 80, 80), 1)
    cv2.line(img, (PAD, PAD),   (PAD, H-PAD),   (80, 80, 80), 1)
    cv2.line(img, (PAD, H-PAD), (W-PAD, PAD), (60, 60, 60), 1)

    def to_px(fpr, tpr):
        x = PAD + int(fpr * (W - 2*PAD))
        y = (H-PAD) - int(tpr * (H - 2*PAD))
        return x, y

    pts = [to_px(f, t) for f, t in zip(fprs, tprs)]
    for i in range(len(pts)-1):
        cv2.line(img, pts[i], pts[i+1], (197, 240, 58), 2)

    cv2.putText(img, f"ROC Curve  AUC={auc:.3f}",
                (PAD, PAD-10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (197, 240, 58), 1)
    cv2.putText(img, f"Best threshold: {best_thr}",
                (PAD, PAD+18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (58, 255, 160), 1)
    cv2.putText(img, "FPR", (W//2, H-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (130, 130, 130), 1)
    cv2.putText(img, "TPR", (4, H//2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (130, 130, 130), 1)

    cv2.imwrite(out_path, img)


def trapz_auc(fprs: list, tprs: list) -> float:
    pts = sorted(zip(fprs, tprs))
    auc = 0.0
    for i in range(1, len(pts)):
        dx = pts[i][0] - pts[i-1][0]
        dy = (pts[i][1] + pts[i-1][1]) / 2
        auc += dx * dy
    return round(abs(auc), 4)



def run(n_per_brand: int = 30, threshold: float | None = None,
        brands: list | None = None, save_examples: int = 10):

    print("=" * 60)
    print("Генерация синтетических фейков и оценка системы")
    print("=" * 60)

    detector, model, classes, reference, device, thr = load_all(threshold)
    if threshold is None:
        threshold = thr
    print(f"Порог: {threshold} | Устройство: {device}")

    eval_dir  = os.path.join(config.WORK_DIR, "fake_eval")
    ex_dir    = os.path.join(eval_dir, "examples")
    os.makedirs(ex_dir, exist_ok=True)

    transform = get_transform()

    real_paths = collect_crops(config.CROPS_DIR, brands, n_per_brand)
    print(f"Настоящих кропов: {len(real_paths)}")
    if not real_paths:
        print("[ОШИБКА] Кропы не найдены. Запусти prepare_data.py -> make_crops()")
        return


    all_y_true:  list[int]   = []
    all_y_scores: list[float] = []

    per_transform: dict[str, dict] = {t: {"y_true": [], "y_scores": []}
                                       for t in TRANSFORMS}

    examples_saved = 0

    print("\nОбработка настоящих логотипов...")
    for path in tqdm(real_paths, desc="  REAL"):
        img = load_image_bgr(path)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = classify_crop(img_rgb, model, classes, reference,
                            device, threshold, transform)
        all_y_true.append(0)
        all_y_scores.append(res["similarity"])

    print("\nГенерация и обработка фейковых логотипов...")
    for t_name, t_fn in TRANSFORMS.items():
        print(f"  [{t_name}]")
        for path in tqdm(real_paths, desc=f"    {t_name}", leave=False):
            img = load_image_bgr(path)
            if img is None:
                continue
            fake_bgr = t_fn(img)
            fake_rgb = cv2.cvtColor(fake_bgr, cv2.COLOR_BGR2RGB)
            res = classify_crop(fake_rgb, model, classes, reference,
                                device, threshold, transform)

            all_y_true.append(1)
            all_y_scores.append(res["similarity"])
            per_transform[t_name]["y_true"].append(1)
            per_transform[t_name]["y_scores"].append(res["similarity"])

            if examples_saved < save_examples:
                ann = annotate_image(fake_bgr, t_name,
                                     res["similarity"], res["is_fake"])
                stem = os.path.splitext(os.path.basename(path))[0]
                cv2.imwrite(os.path.join(ex_dir, f"{t_name}_{stem}.jpg"), ann)
                examples_saved += 1

    print("\n" + "=" * 60)
    print(f"{'ТРАНСФОРМАЦИЯ':<14} {'PREC':>6} {'REC':>6} {'F1':>6} {'ACC':>6}")
    print("-" * 60)

    transform_metrics = {}
    for t_name, data in per_transform.items():
        y_t = list(np.zeros(len(real_paths[:len(data["y_true"])]), dtype=int)) \
              + data["y_true"]
        y_s = all_y_scores[:len(real_paths[:len(data["y_true"])])] \
              + data["y_scores"]

        m = compute_metrics(y_t, y_s, threshold)
        transform_metrics[t_name] = m
        print(f"  {t_name:<12} {m['precision']:>6.3f} {m['recall']:>6.3f} "
              f"{m['f1']:>6.3f} {m['accuracy']:>6.3f}")

    overall = compute_metrics(all_y_true, all_y_scores, threshold)
    print("-" * 60)
    print(f"  {'OVERALL':<12} {overall['precision']:>6.3f} {overall['recall']:>6.3f} "
          f"{overall['f1']:>6.3f} {overall['accuracy']:>6.3f}")
    print(f"\n  TP={overall['tp']}  FP={overall['fp']}  "
          f"TN={overall['tn']}  FN={overall['fn']}")

    fprs, tprs, thrs = compute_roc(all_y_true, all_y_scores)
    auc = trapz_auc(fprs, tprs)
    best_thr, best_j = find_best_threshold(all_y_true, all_y_scores)

    print(f"\n  AUC: {auc:.4f}")
    print(f"  Оптимальный порог (Youden J={best_j:.3f}): {best_thr}")
    print(f"  Текущий порог в config.py:                {config.FAKE_THRESHOLD}")

    if abs(best_thr - config.FAKE_THRESHOLD) > 0.05:
        print(f"  ⚠ Рекомендуется обновить FAKE_THRESHOLD = {best_thr}")

    roc_path = os.path.join(eval_dir, "roc_curve.png")
    save_roc_png(fprs, tprs, best_thr, auc, roc_path)
    print(f"\n  ROC-кривая: {roc_path}")

    report = {
        "threshold_used":     threshold,
        "optimal_threshold":  best_thr,
        "youden_j":           best_j,
        "auc":                auc,
        "n_real":             len(real_paths),
        "n_fake_per_type":    len(real_paths),
        "overall":            overall,
        "per_transform":      transform_metrics,
    }
    report_path = os.path.join(eval_dir, "metrics.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  Метрики JSON: {report_path}")
    print(f"  Примеры:      {ex_dir}")
    print("\n✔ Готово.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Генерация синтетических фейков и оценка системы")
    parser.add_argument("--n",          type=int,   default=30,
                        help="Кропов на бренд (default: 30)")
    parser.add_argument("--threshold",  type=float, default=None,
                        help="Порог (default: из config.py)")
    parser.add_argument("--brands",     nargs="+",  default=None,
                        help="Список брендов (default: все)")
    parser.add_argument("--examples",   type=int,   default=10,
                        help="Сколько примеров сохранить (default: 10)")
    args = parser.parse_args()

    random.seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    run(n_per_brand   = args.n,
        threshold     = args.threshold,
        brands        = args.brands,
        save_examples = args.examples)