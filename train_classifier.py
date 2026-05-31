import csv
import os
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms

import config


def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((config.CLS_IMG_SIZE, config.CLS_IMG_SIZE)),

        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),

        transforms.RandomPerspective(distortion_scale=0.2, p=0.4),

        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.05,
        ),

        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),

        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((config.CLS_IMG_SIZE, config.CLS_IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    return train_transform, val_transform


def save_csv(log_path: str, rows: list[dict]):
    if not rows:
        return
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

def save_curves_png(log_rows: list[dict], out_path: str):
    W, H, PAD = 720, 440, 60
    img = np.full((H, W, 3), 245, dtype=np.uint8)

    epochs     = [r["epoch"]    for r in log_rows]
    losses     = [r["train_loss"] for r in log_rows]
    val_accs   = [r["val_acc"]  for r in log_rows]
    n          = len(epochs)

    if n < 2:
        return

    def px_loss(i, v):
        x = PAD + int(i / (n - 1) * (W - 2 * PAD))
        y = PAD + int((1 - v / (max(losses) + 1e-6)) * (H // 2 - PAD - 10))
        return x, max(PAD, min(H // 2 - 5, y))

    def px_acc(i, v):
        x = PAD + int(i / (n - 1) * (W - 2 * PAD))
        y = H // 2 + PAD + int((1 - v) * (H // 2 - PAD - 10))
        return x, max(H // 2 + PAD, min(H - PAD, y))

    FONT = cv2.FONT_HERSHEY_SIMPLEX

    cv2.rectangle(img, (PAD, PAD), (W - PAD, H // 2 - 5), (255, 255, 255), -1)
    cv2.rectangle(img, (PAD, PAD), (W - PAD, H // 2 - 5), (200, 200, 200),  1)
    cv2.putText(img, "Train Loss", (PAD + 6, PAD + 18),
                FONT, 0.5, (60, 60, 60), 1, cv2.LINE_AA)

    pts_loss = [px_loss(i, v) for i, v in enumerate(losses)]
    for i in range(1, len(pts_loss)):
        cv2.line(img, pts_loss[i-1], pts_loss[i], (200, 80, 0), 2, cv2.LINE_AA)

    min_loss_i = losses.index(min(losses))
    cv2.circle(img, pts_loss[min_loss_i], 4, (200, 80, 0), -1)
    cv2.putText(img, f"min={losses[min_loss_i]:.4f}",
                (pts_loss[min_loss_i][0] + 6, pts_loss[min_loss_i][1] - 6),
                FONT, 0.38, (200, 80, 0), 1)

    mid = H // 2 + 5
    cv2.rectangle(img, (PAD, mid), (W - PAD, H - PAD), (255, 255, 255), -1)
    cv2.rectangle(img, (PAD, mid), (W - PAD, H - PAD), (200, 200, 200),  1)
    cv2.putText(img, "Val Accuracy", (PAD + 6, mid + 18),
                FONT, 0.5, (60, 60, 60), 1, cv2.LINE_AA)

    best_acc   = max(val_accs)
    best_acc_i = val_accs.index(best_acc)

    _, py_best = px_acc(0, best_acc)
    for x in range(PAD, W - PAD, 8):
        cv2.line(img, (x, py_best), (min(x + 4, W - PAD), py_best),
                 (160, 200, 160), 1)

    pts_acc = [px_acc(i, v) for i, v in enumerate(val_accs)]
    for i in range(1, len(pts_acc)):
        cv2.line(img, pts_acc[i-1], pts_acc[i], (0, 140, 0), 2, cv2.LINE_AA)

    cv2.circle(img, pts_acc[best_acc_i], 5, (0, 180, 0), -1)
    cv2.putText(img, f"best={best_acc:.4f} (ep{best_acc_i+1})",
                (pts_acc[best_acc_i][0] + 7, pts_acc[best_acc_i][1] - 6),
                FONT, 0.38, (0, 140, 0), 1)

    for i in [0, n // 4, n // 2, 3 * n // 4, n - 1]:
        px, _ = px_acc(i, 0)
        cv2.putText(img, str(epochs[i]), (px - 6, H - PAD + 16),
                    FONT, 0.36, (100, 100, 100), 1)

    cv2.putText(img, "Epoch", (W // 2 - 20, H - 10),
                FONT, 0.42, (100, 100, 100), 1)

    cv2.imwrite(out_path, img)


def train():
    print("=" * 55)
    print("Обучение классификатора EfficientNet-B0")
    print("=" * 55)

    if not os.path.exists(config.CROPS_DIR):
        print(f"[ОШИБКА] Папка с кропами не найдена: {config.CROPS_DIR}")
        print("Сначала запустите prepare_data.py -> make_crops()")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Устройство:  {device}")
    print(f"Эпох:        {config.CLS_EPOCHS}")
    print(f"Батч:        {config.CLS_BATCH}")
    print(f"LR старт:    {config.CLS_LR}")

    train_transform, val_transform = get_transforms()

    full_dataset = datasets.ImageFolder(config.CROPS_DIR, transform=train_transform)
    num_classes  = len(full_dataset.classes)
    print(f"Классов:     {num_classes}")
    print(f"Изображений: {len(full_dataset)}")

    os.makedirs(config.WORK_DIR, exist_ok=True)
    with open(config.CLASSES_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(full_dataset.classes))
    print(f"Классы сохранены: {config.CLASSES_FILE}")

    val_size   = int(len(full_dataset) * (1 - config.TRAIN_VAL_SPLIT))
    train_size = len(full_dataset) - val_size
    generator  = torch.Generator().manual_seed(config.RANDOM_SEED)
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=generator
    )
    val_dataset.dataset = datasets.ImageFolder(
        config.CROPS_DIR, transform=val_transform
    )
    print(f"Train: {train_size}  |  Val: {val_size}")

    train_loader = DataLoader(train_dataset, batch_size=config.CLS_BATCH,
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=config.CLS_BATCH,
                              shuffle=False, num_workers=2, pin_memory=True)

    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(1280, num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = optim.Adam(model.parameters(), lr=config.CLS_LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.CLS_EPOCHS, eta_min=1e-6
    )

    best_val_acc    = 0.0
    patience_count  = 0
    es_patience     = getattr(config, "CLS_PATIENCE", 10)

    log_rows   = []
    t_total    = time.perf_counter()

    print(f"\n{'─'*55}")
    print(f"{'Ep':>4}  {'Loss':>8}  {'Val Acc':>8}  "
          f"{'LR':>10}  {'Best':>8}  {'Time':>6}")
    print(f"{'─'*55}")

    for epoch in range(1, config.CLS_EPOCHS + 1):
        t_ep = time.perf_counter()

        model.train()
        train_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_loss = train_loss / len(train_loader)
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds    = model(imgs).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)

        val_acc   = correct / total if total > 0 else 0.0
        ep_time   = time.perf_counter() - t_ep
        is_best   = val_acc > best_val_acc

        if is_best:
            best_val_acc   = val_acc
            patience_count = 0
            torch.save(model.state_dict(), config.CLASSIFIER_WEIGHTS)
        else:
            patience_count += 1

        print(f"{epoch:>4}  {avg_loss:>8.4f}  {val_acc:>8.4f}  "
              f"{current_lr:>10.2e}  {best_val_acc:>8.4f}  "
              f"{ep_time:>5.0f}с"
              + ("  ✔" if is_best else ""))

        log_rows.append({
            "epoch":      epoch,
            "train_loss": round(avg_loss, 4),
            "val_acc":    round(val_acc,  4),
            "lr":         round(current_lr, 8),
            "is_best":    int(is_best),
            "epoch_sec":  round(ep_time, 1),
        })

        if patience_count >= es_patience:
            print(f"\n  Early stopping на эпохе {epoch} "
                  f"(нет улучшения {es_patience} эпох подряд)")
            break

    total_time = time.perf_counter() - t_total
    print(f"{'─'*55}")
    print(f"\n✔ Обучение завершено за {total_time/60:.1f} мин")
    print(f"  Лучшая val_acc: {best_val_acc:.4f}")
    print(f"  Веса: {config.CLASSIFIER_WEIGHTS}")

    log_path   = os.path.join(config.WORK_DIR, "training_log.csv")
    curves_path = os.path.join(config.WORK_DIR, "training_curves.png")

    save_csv(log_path, log_rows)
    print(f"  Метрики CSV: {log_path}")

    save_curves_png(log_rows, curves_path)
    print(f"  График PNG:  {curves_path}")

if __name__ == "__main__":
    train()