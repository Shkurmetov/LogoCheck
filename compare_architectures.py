import argparse
import json
import os
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
from tqdm import tqdm

import config


ARCHS = {
    "efficientnet_b0": {
        "display": "EfficientNet-B0",
        "params_m": 5.3,
    },
    "resnet50": {
        "display": "ResNet-50",
        "params_m": 25.6,
    },
    "mobilenet_v3_large": {
        "display": "MobileNetV3-L",
        "params_m": 5.5,
    },
}


def build_model(arch: str, num_classes: int) -> nn.Module:
    if arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        m.classifier[1] = nn.Linear(1280, num_classes)

    elif arch == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        m.fc = nn.Linear(2048, num_classes)

    elif arch == "mobilenet_v3_large":
        m = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        m.classifier[-1] = nn.Linear(1280, num_classes)

    else:
        raise ValueError(f"Неизвестная архитектура: {arch}")

    return m


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def weights_path(arch: str) -> str:
    return os.path.join(config.WORK_DIR, f"cmp_{arch}.pt")



def get_loaders(batch_size: int, seed: int):
    train_tf = transforms.Compose([
        transforms.Resize((config.CLS_IMG_SIZE, config.CLS_IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((config.CLS_IMG_SIZE, config.CLS_IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    full = datasets.ImageFolder(config.CROPS_DIR, transform=train_tf)
    num_classes = len(full.classes)

    val_size   = int(len(full) * (1 - config.TRAIN_VAL_SPLIT))
    train_size = len(full) - val_size
    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full, [train_size, val_size], generator=gen)

    val_ds.dataset = datasets.ImageFolder(config.CROPS_DIR, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, num_classes



def train_arch(arch: str, num_classes: int, device: str,
               epochs: int, batch_size: int, lr: float,
               seed: int) -> dict:
    print(f"\n{'='*60}")
    print(f"  Обучение: {ARCHS[arch]['display']}")
    print(f"{'='*60}")

    model = build_model(arch, num_classes).to(device)
    n_params = count_params(model)
    print(f"  Параметров: {n_params:,}  ({n_params/1e6:.1f} млн)")

    train_loader, val_loader, _ = get_loaders(batch_size, seed)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_acc     = 0.0
    wpath        = weights_path(arch)
    epoch_times  = []
    train_losses = []
    val_accs     = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        model.train()
        loss_sum = 0.0
        for imgs, labels in tqdm(train_loader,
                                 desc=f"  [{arch[:12]}] E{epoch:02d} train",
                                 leave=False, ncols=80):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs).argmax(1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)

        epoch_sec = time.time() - t0
        val_acc   = correct / total if total else 0.0
        avg_loss  = loss_sum / len(train_loader)

        epoch_times.append(round(epoch_sec, 1))
        train_losses.append(round(avg_loss, 4))
        val_accs.append(round(val_acc, 4))

        print(f"  E{epoch:02d}/{epochs}  "
              f"loss={avg_loss:.4f}  val_acc={val_acc:.4f}  "
              f"({epoch_sec:.0f}s)")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), wpath)

    size_mb = os.path.getsize(wpath) / 1024 / 1024 if os.path.exists(wpath) else 0.0

    return {
        "arch":         arch,
        "display":      ARCHS[arch]["display"],
        "best_val_acc": round(best_acc, 4),
        "params":       n_params,
        "size_mb":      round(size_mb, 1),
        "epoch_times":  epoch_times,
        "train_losses": train_losses,
        "val_accs":     val_accs,
        "avg_epoch_sec": round(sum(epoch_times) / len(epoch_times), 1),
        "weights_path": wpath,
    }



def measure_inference_ms(arch: str, num_classes: int,
                         wpath: str, n_runs: int = 200) -> float:
    device = "cpu"
    model  = build_model(arch, num_classes)
    model.load_state_dict(torch.load(wpath, map_location="cpu"))
    model.eval()

    dummy = torch.randn(1, 3, config.CLS_IMG_SIZE, config.CLS_IMG_SIZE)

    with torch.no_grad():
        for _ in range(10):
            model(dummy)

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            model(dummy)
    elapsed = (time.perf_counter() - t0) / n_runs * 1000

    return round(elapsed, 2)



def _best_idx(values: list, higher_is_better: bool) -> int:
    return (values.index(max(values)) if higher_is_better
            else values.index(min(values)))


def generate_table_image(results: list[dict]) -> np.ndarray:
    FONT      = cv2.FONT_HERSHEY_SIMPLEX
    FS        = 0.55
    FT        = 1
    PAD       = 14
    ROW_H     = 40
    COL_W     = [260, 180, 180, 180]

    BG_HDR    = (40,  40,  40)
    BG_ROW0   = (255, 255, 255)
    BG_ROW1   = (240, 243, 248)
    C_HDR     = (255, 255, 255)
    C_NORMAL  = (30,  30,  30)
    C_BEST    = (0,   140, 0)
    C_BORDER  = (180, 180, 180)

    arch_names = [r["display"] for r in results]

    rows_data = [
        {
            "label": "Val Accuracy",
            "values": [f"{r['best_val_acc']*100:.2f}%"  for r in results],
            "raw":    [r["best_val_acc"]                 for r in results],
            "higher": True,
        },
        {
            "label": "Параметры (млн)",
            "values": [f"{r['params']/1e6:.2f}"          for r in results],
            "raw":    [r["params"]                        for r in results],
            "higher": False,
        },
        {
            "label": "Размер файла (МБ)",
            "values": [f"{r['size_mb']:.1f}"             for r in results],
            "raw":    [r["size_mb"]                       for r in results],
            "higher": False,
        },
        {
            "label": "Ср. время эпохи (с)",
            "values": [f"{r['avg_epoch_sec']:.0f}"        for r in results],
            "raw":    [r["avg_epoch_sec"]                  for r in results],
            "higher": False,
        },
        {
            "label": "Инференс CPU (мс)",
            "values": [f"{r.get('inference_ms', '—')}"   for r in results],
            "raw":    [r.get("inference_ms", 1e9)          for r in results],
            "higher": False,
        },
    ]

    n_cols   = 1 + len(results)
    total_w  = sum(COL_W[:n_cols])
    n_rows   = 1 + len(rows_data)
    total_h  = n_rows * ROW_H + 2

    img = np.ones((total_h, total_w, 3), dtype=np.uint8) * 255

    def draw_cell(row, col, text, bg, fg, bold=False):
        x0 = sum(COL_W[:col])
        y0 = row * ROW_H
        x1 = x0 + COL_W[col]
        y1 = y0 + ROW_H
        img[y0:y1, x0:x1] = bg
        tw = cv2.getTextSize(text, FONT, FS, FT)[0][0]
        tx = x0 + PAD if col == 0 else x0 + (COL_W[col] - tw) // 2
        ty = y0 + ROW_H // 2 + 7
        thickness = 2 if bold else FT
        cv2.putText(img, text, (tx, ty), FONT, FS, fg, thickness, cv2.LINE_AA)
        cv2.line(img, (x1-1, y0), (x1-1, y1), C_BORDER, 1)
        cv2.line(img, (x0, y1-1), (x1, y1-1), C_BORDER, 1)

    draw_cell(0, 0, "Метрика", BG_HDR, C_HDR, bold=True)
    for j, name in enumerate(arch_names):
        draw_cell(0, j+1, name, BG_HDR, C_HDR, bold=True)

    for i, row in enumerate(rows_data):
        bg = BG_ROW0 if i % 2 == 0 else BG_ROW1
        draw_cell(i+1, 0, row["label"], bg, C_NORMAL)

        try:
            best_j = _best_idx(row["raw"], row["higher"])
        except Exception:
            best_j = -1

        for j, val in enumerate(row["values"]):
            fg = C_BEST if j == best_j else C_NORMAL
            draw_cell(i+1, j+1, val, bg, fg, bold=(j == best_j))

    cv2.rectangle(img, (0, 0), (total_w-1, total_h-1), C_BORDER, 2)

    return img


def generate_curves_image(results: list[dict]) -> np.ndarray:
    W, H   = 700, 400
    PAD    = 60
    COLORS = [(200, 80,  0),
              (0,   100, 200),
              (0,   160, 0)]
    FONT   = cv2.FONT_HERSHEY_SIMPLEX

    img = np.ones((H, W, 3), dtype=np.uint8) * 255

    cv2.line(img, (PAD, PAD),       (PAD, H-PAD),   (100,100,100), 1)
    cv2.line(img, (PAD, H-PAD),     (W-PAD, H-PAD), (100,100,100), 1)
    cv2.putText(img, "Val Accuracy", (PAD, 25), FONT, 0.5, (50,50,50), 1, cv2.LINE_AA)

    max_epochs = max(len(r["val_accs"]) for r in results)
    def to_px(epoch_idx, acc):
        x = PAD + int(epoch_idx / max(max_epochs-1, 1) * (W - 2*PAD))
        y = H - PAD - int(acc * (H - 2*PAD))
        return (x, y)

    for v in [0.0, 0.25, 0.5, 0.75, 1.0]:
        _, py = to_px(0, v)
        cv2.line(img, (PAD-4, py), (W-PAD, py), (220,220,220), 1)
        cv2.putText(img, f"{v:.2f}", (4, py+4), FONT, 0.38, (100,100,100), 1)

    for idx, r in enumerate(results):
        accs  = r["val_accs"]
        color = COLORS[idx % len(COLORS)]
        pts   = [to_px(e, a) for e, a in enumerate(accs)]
        for i in range(1, len(pts)):
            cv2.line(img, pts[i-1], pts[i], color, 2, cv2.LINE_AA)
        ly = PAD + 20 + idx * 22
        cv2.line(img, (W-PAD-80, ly), (W-PAD-60, ly), color, 2)
        cv2.putText(img, r["display"], (W-PAD-56, ly+5),
                    FONT, 0.42, color, 1, cv2.LINE_AA)

    cv2.putText(img, "Эпохи", (W//2-20, H-10), FONT, 0.45, (80,80,80), 1)

    return img



def run(archs_to_run: list[str],
        epochs: int,
        batch_size: int,
        lr: float,
        skip_existing: bool,
        table_only: bool):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Устройство: {device}")
    print(f"Эпох: {epochs}  Батч: {batch_size}  LR: {lr}")

    os.makedirs(config.WORK_DIR, exist_ok=True)

    if not os.path.exists(config.CROPS_DIR):
        print(f"[ОШИБКА] CROPS_DIR не найден: {config.CROPS_DIR}")
        print("Запустите prepare_data.py → make_crops() сначала.")
        return

    num_classes = len([d for d in os.listdir(config.CROPS_DIR)
                       if os.path.isdir(os.path.join(config.CROPS_DIR, d))])
    print(f"Классов брендов: {num_classes}\n")

    results_path = os.path.join(config.WORK_DIR, "comparison_results.json")

    existing: dict[str, dict] = {}
    if os.path.exists(results_path):
        with open(results_path, encoding="utf-8") as f:
            for r in json.load(f):
                existing[r["arch"]] = r

    results = []

    for arch in archs_to_run:
        wpath = weights_path(arch)

        if table_only or (skip_existing and os.path.exists(wpath)
                          and arch in existing):
            print(f"  Пропуск обучения {arch} (веса найдены)")
            r = existing.get(arch, {
                "arch":         arch,
                "display":      ARCHS[arch]["display"],
                "best_val_acc": 0.0,
                "params":       0,
                "size_mb":      round(os.path.getsize(wpath)/1024/1024, 1) if os.path.exists(wpath) else 0.0,
                "avg_epoch_sec": 0.0,
                "epoch_times":  [],
                "train_losses": [],
                "val_accs":     [],
                "weights_path": wpath,
            })
        else:
            r = train_arch(arch, num_classes, device,
                           epochs, batch_size, lr, config.RANDOM_SEED)

        if os.path.exists(wpath):
            print(f"  Замер инференса {arch}...")
            r["inference_ms"] = measure_inference_ms(arch, num_classes, wpath)
            print(f"  → {r['inference_ms']} мс/изображение")
        else:
            r["inference_ms"] = None

        results.append(r)

    if not results:
        print("Нет результатов для отображения.")
        return

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n✔ Результаты сохранены: {results_path}")

    table_img   = generate_table_image(results)
    table_path  = os.path.join(config.WORK_DIR, "comparison_table.jpg")
    cv2.imwrite(table_path, table_img)
    print(f"✔ Таблица сохранена:    {table_path}")

    if any(r.get("val_accs") for r in results):
        curves_img  = generate_curves_image(results)
        curves_path = os.path.join(config.WORK_DIR, "comparison_curves.jpg")
        cv2.imwrite(curves_path, curves_img)
        print(f"✔ Кривые сохранены:     {curves_path}")

    print("\n" + "─"*60)
    print(f"{'Архитектура':<20} {'Val Acc':>9} {'Params':>10} "
          f"{'Size MB':>8} {'Эпоха(с)':>9} {'Inference':>10}")
    print("─"*60)
    for r in results:
        inf = f"{r['inference_ms']} мс" if r.get("inference_ms") else "—"
        print(f"{r['display']:<20} "
              f"{r['best_val_acc']*100:>8.2f}% "
              f"{r['params']/1e6:>9.1f}M "
              f"{r['size_mb']:>7.1f} "
              f"{r['avg_epoch_sec']:>8.0f}с "
              f"{inf:>10}")
    print("─"*60)

    best = max(results, key=lambda x: x["best_val_acc"])
    print(f"\n  Лучшая архитектура по val_acc: {best['display']} "
          f"({best['best_val_acc']*100:.2f}%)")
    print(f"\n  Веса каждой архитектуры сохранены как cmp_<arch>.pt")
    print(f"  Основной классификатор (brand_classifier.pt) не изменён.")
    print(f"  Это сравнение только для диплома — результаты в comparison_results.json")



def main():
    parser = argparse.ArgumentParser(
        description="Сравнительный анализ архитектур классификатора логотипов")

    parser.add_argument(
        "--archs", nargs="+",
        default=list(ARCHS.keys()),
        choices=list(ARCHS.keys()),
        help="Архитектуры для сравнения (default: все три)"
    )
    parser.add_argument(
        "--epochs", type=int, default=config.CLS_EPOCHS,
        help=f"Эпох обучения (default: {config.CLS_EPOCHS})"
    )
    parser.add_argument(
        "--batch", type=int, default=config.CLS_BATCH,
        help=f"Размер батча (default: {config.CLS_BATCH})"
    )
    parser.add_argument(
        "--lr", type=float, default=config.CLS_LR,
        help=f"Скорость обучения (default: {config.CLS_LR})"
    )
    parser.add_argument(
        "--skip_existing", action="store_true",
        help="Не переобучать архитектуру если её веса уже существуют"
    )
    parser.add_argument(
        "--table_only", action="store_true",
        help="Только построить таблицу по уже существующим весам"
    )

    args = parser.parse_args()

    run(
        archs_to_run  = args.archs,
        epochs        = args.epochs,
        batch_size    = args.batch,
        lr            = args.lr,
        skip_existing = args.skip_existing,
        table_only    = args.table_only,
    )


if __name__ == "__main__":
    main()