import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.cluster import KMeans

import config

class EmbeddingExtractor(nn.Module):
    def __init__(self, weights_path, num_classes):
        super().__init__()
        base = models.efficientnet_b0(weights=None)
        base.classifier[1] = nn.Linear(1280, num_classes)
        base.load_state_dict(torch.load(weights_path, map_location="cpu"))
        self.features = base.features
        self.avgpool  = base.avgpool

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return x.flatten(1)   # (batch, 1280), ненормализованный

def build():
    print("=" * 55)
    print("Построение эталонной базы (KMeans-кластеры)")
    print("=" * 55)

    for path, name in [
        (config.CROPS_DIR,          "Папка с кропами (CROPS_DIR)"),
        (config.CLASSIFIER_WEIGHTS, "Веса классификатора"),
        (config.CLASSES_FILE,       "Файл классов"),
    ]:
        if not os.path.exists(path):
            print(f"[ОШИБКА] {name} не найден: {path}")
            return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Устройство:  {device}")
    print(f"Кластеров на бренд: {config.REFERENCE_N_CLUSTERS}\n")

    with open(config.CLASSES_FILE, encoding="utf-8") as f:
        classes = [line.strip() for line in f if line.strip()]
    num_classes = len(classes)

    extractor = EmbeddingExtractor(config.CLASSIFIER_WEIGHTS, num_classes)
    extractor.eval()
    extractor.to(device)

    transform = transforms.Compose([
        transforms.Resize((config.CLS_IMG_SIZE, config.CLS_IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    dataset = datasets.ImageFolder(config.CROPS_DIR, transform=transform)
    loader  = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)

    all_embeddings = [[] for _ in range(num_classes)]

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="  Извлечение эмбеддингов"):
            embs = extractor(imgs.to(device)).cpu()
            for emb, lbl in zip(embs, labels):
                all_embeddings[lbl.item()].append(emb)

    reference = {}
    stats = {"kmeans": 0, "single": 0, "empty": 0}

    print("\n  KMeans кластеризация...")
    for i, brand in enumerate(tqdm(classes, desc="  Бренды")):
        embs = all_embeddings[i]

        if len(embs) == 0:
            stats["empty"] += 1
            continue

        embs_tensor = torch.stack(embs)              # (N, 1280)
        k = min(config.REFERENCE_N_CLUSTERS, len(embs))

        if k == 1:
            centroid = embs_tensor[0]
            centroid = centroid / centroid.norm()
            centroids = centroid.unsqueeze(0)        # (1, 1280)
            stats["single"] += 1
        else:
            km = KMeans(n_clusters=k, random_state=config.RANDOM_SEED,
                        n_init=10, max_iter=300)
            km.fit(embs_tensor.numpy())
            centroids = torch.tensor(km.cluster_centers_, dtype=torch.float32)
            centroids = centroids / centroids.norm(dim=1, keepdim=True)
            stats["kmeans"] += 1

        reference[brand] = centroids                 # (k, 1280)

    os.makedirs(config.WORK_DIR, exist_ok=True)
    torch.save(reference, config.REFERENCE_EMBEDDINGS)

    print(f"\n✔ Готово.")
    print(f"  Брендов с KMeans ({config.REFERENCE_N_CLUSTERS} кластеров): {stats['kmeans']}")
    print(f"  Брендов с 1 кропом: {stats['single']}")
    print(f"  Пропущено (нет кропов): {stats['empty']}")
    print(f"  Файл: {config.REFERENCE_EMBEDDINGS}")
    print(f"\n  Теперь запусти verify_logo.py или verify_video.py.")

if __name__ == "__main__":
    build()