import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

import config
from verify_logo import load_all, classify_crop, get_transform

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    HAS_GRADCAM = True
except ImportError:
    HAS_GRADCAM = False



class ClassifierForCAM(nn.Module):
    def __init__(self, weights_path: str, num_classes: int):
        super().__init__()
        base = models.efficientnet_b0(weights=None)
        base.classifier[1] = nn.Linear(1280, num_classes)
        base.load_state_dict(
            torch.load(weights_path, map_location="cpu", weights_only=True)
        )
        self.features   = base.features   # свёрточный бэкбон
        self.avgpool    = base.avgpool
        self.classifier = base.classifier

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.classifier(x)          # логиты, форма: (batch, num_classes)

    @property
    def target_layer(self):
        return self.features[-1]



def tensor_to_rgb(tensor: torch.Tensor) -> np.ndarray:
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    img = tensor.squeeze(0).permute(1, 2, 0).numpy()   # (H, W, 3)
    img = img * std + mean                              # денормализация
    img = np.clip(img, 0, 1).astype(np.float32)
    return img


def run_gradcam_manual(cam_model: nn.Module,
                       input_tensor: torch.Tensor,
                       target_class: int) -> np.ndarray:
    activations = {}
    gradients   = {}

    def fwd_hook(module, input, output):
        activations["feat"] = output.detach()

    def bwd_hook(module, grad_input, grad_output):
        gradients["feat"] = grad_output[0].detach()

    target = cam_model.target_layer
    h1 = target.register_forward_hook(fwd_hook)
    h2 = target.register_full_backward_hook(bwd_hook)

    cam_model.eval()
    logits = cam_model(input_tensor)            # (1, num_classes)

    cam_model.zero_grad()
    score = logits[0, target_class]
    score.backward()

    h1.remove()
    h2.remove()

    # GAP градиентов → веса
    grads = gradients["feat"]                   # (1, C, H, W)
    acts  = activations["feat"]                 # (1, C, H, W)
    weights = grads.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

    # Взвешенная сумма + ReLU
    cam = (weights * acts).sum(dim=1).squeeze(0)    # (H, W)
    cam = torch.relu(cam).numpy()

    # Нормализация
    if cam.max() > 0:
        cam = cam / cam.max()

    # Resize до размера входа (CLS_IMG_SIZE × CLS_IMG_SIZE)
    cam = cv2.resize(cam, (config.CLS_IMG_SIZE, config.CLS_IMG_SIZE))
    return cam.astype(np.float32)


def apply_colormap(cam: np.ndarray, rgb_img: np.ndarray) -> np.ndarray:
    heatmap_bgr = cv2.applyColorMap(
        (cam * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    overlay = (0.45 * heatmap_rgb + 0.55 * rgb_img * 255).astype(np.uint8)
    return cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)


def add_verdict_banner(img_bgr: np.ndarray, result: dict) -> np.ndarray:
    is_fake    = result["is_fake"]
    color_bgr  = (0, 0, 210) if is_fake else (0, 180, 0)
    verdict    = "FAKE" if is_fake else "REAL"
    text       = (f"{result['brand'].upper()}  |  {verdict}  |  "
                  f"sim={result['similarity']:.3f}  conf={result['cls_conf']:.3f}")

    out = img_bgr.copy()
    h, w = out.shape[:2]

    # Полоса внизу
    banner_h = 28
    cv2.rectangle(out, (0, h - banner_h), (w, h), color_bgr, -1)
    cv2.putText(out, text, (6, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return out



def visualize_gradcam(image_path: str,
                      cam_model: nn.Module,
                      verify_model,
                      classes: list,
                      reference: dict,
                      device: str,
                      threshold: float,
                      out_dir: str | None = None,
                      show: bool = False) -> str:
    transform = get_transform()

    # Читаем изображение
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Не удалось открыть: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Подготовка тензора
    pil_img    = Image.fromarray(img_rgb).resize(
        (config.CLS_IMG_SIZE, config.CLS_IMG_SIZE)
    )
    input_tensor = transform(pil_img).unsqueeze(0).to(device)

    # Верификация
    crop_rgb = np.array(pil_img)
    result   = classify_crop(crop_rgb, verify_model, classes, reference,
                             device, threshold, transform)

    brand_idx = classes.index(result["brand"])

    # GradCAM
    cam_model.eval()

    if HAS_GRADCAM:
        # Официальная библиотека pytorch-grad-cam
        with GradCAM(
            model=cam_model,
            target_layers=[cam_model.target_layer],
        ) as cam_ctx:
            targets = [ClassifierOutputTarget(brand_idx)]
            grayscale_cam = cam_ctx(
                input_tensor=input_tensor,
                targets=targets,
            )[0]                           # (H, W), float32 [0,1]

        rgb_float = tensor_to_rgb(input_tensor.cpu())
        overlay_rgb = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)
        overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)

    else:
        # Ручная реализация без библиотеки
        print("  [INFO] pytorch-grad-cam не установлен, "
              "используется встроенная реализация.")
        grayscale_cam = run_gradcam_manual(
            cam_model, input_tensor.cpu(), brand_idx
        )
        rgb_float  = tensor_to_rgb(input_tensor.cpu())
        overlay_bgr = apply_colormap(grayscale_cam, rgb_float)

    # Добавляем баннер с вердиктом
    result_img = add_verdict_banner(overlay_bgr, result)

    # Сохранение
    stem     = os.path.splitext(os.path.basename(image_path))[0]
    out_name = stem + "_gradcam.jpg"

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, out_name)
    else:
        out_path = os.path.join(os.path.dirname(image_path), out_name)

    cv2.imwrite(out_path, result_img)
    print(f"  Сохранено: {out_path}")

    if show:
        cv2.imshow(f"GradCAM — {os.path.basename(image_path)}", result_img)
        cv2.waitKey(0)

    return out_path



def run_on_folder(folder: str,
                  cam_model: nn.Module,
                  verify_model,
                  classes: list,
                  reference: dict,
                  device: str,
                  threshold: float,
                  out_dir: str | None = None,
                  show: bool = False):

    exts  = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(exts) and "_gradcam" not in f
    ]

    if not files:
        print(f"Изображения не найдены в папке: {folder}")
        return

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        print(f"  Результаты → {out_dir}")

    print(f"  Найдено {len(files)} изображений")
    for path in files:
        try:
            print(f"\n── {os.path.basename(path)}")
            visualize_gradcam(
                path, cam_model, verify_model,
                classes, reference, device, threshold,
                out_dir=out_dir, show=show,
            )
        except Exception as e:
            print(f"  [ОШИБКА] {e}")

    cv2.destroyAllWindows()



def load_cam_model(num_classes: int, device: str) -> ClassifierForCAM:
    model = ClassifierForCAM(config.CLASSIFIER_WEIGHTS, num_classes)
    model.eval()
    model.to(device)
    return model



def main():
    parser = argparse.ArgumentParser(
        description="GradCAM-визуализация для системы верификации логотипов"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image",  type=str, help="Путь к изображению")
    group.add_argument("--folder", type=str, help="Путь к папке с изображениями")

    parser.add_argument("--threshold", type=float, default=None,
                        help="Порог верификации (default: из config.py)")
    parser.add_argument("--outdir",    type=str,   default=None,
                        help="Папка для сохранения результатов "
                             "(default: WORK_DIR/gradcam_results/)")
    parser.add_argument("--show", action="store_true",
                        help="Показывать результат в окне")
    args = parser.parse_args()

    # Папка для результатов: явная или дефолтная
    out_dir = args.outdir or os.path.join(config.WORK_DIR, "gradcam_results")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Результаты будут сохранены в: {out_dir}")

    # Проверяем наличие pytorch-grad-cam
    if not HAS_GRADCAM:
        print("=" * 60)
        print("ВНИМАНИЕ: pytorch-grad-cam не установлен.")
        print("Установи его командой:  pip install grad-cam")
        print("Будет использована встроенная реализация GradCAM.")
        print("=" * 60)

    # Загружаем verify-систему (EmbeddingExtractor + детектор + эталоны)
    detector, verify_model, classes, reference, device, thr = load_all(
        args.threshold
    )
    if args.threshold is not None:
        thr = args.threshold

    print(f"Брендов: {len(classes)} | Порог: {thr} | Устройство: {device}")
    if not HAS_GRADCAM:
        print("Режим: встроенный GradCAM (без pytorch-grad-cam)")
    else:
        print("Режим: pytorch-grad-cam")

    # Загружаем модель специально для GradCAM
    cam_model = load_cam_model(len(classes), device)

    if args.image:
        if not os.path.exists(args.image):
            print(f"[ОШИБКА] Файл не найден: {args.image}")
            return
        print(f"\n── {os.path.basename(args.image)}")
        visualize_gradcam(
            args.image, cam_model, verify_model,
            classes, reference, device, thr,
            out_dir=out_dir, show=args.show,
        )

    elif args.folder:
        if not os.path.isdir(args.folder):
            print(f"[ОШИБКА] Папка не найдена: {args.folder}")
            return
        run_on_folder(
            args.folder, cam_model, verify_model,
            classes, reference, device, thr,
            out_dir=out_dir, show=args.show,
        )

    cv2.destroyAllWindows()
    print(f"\n✔ Готово. Все результаты в: {out_dir}")


if __name__ == "__main__":
    main()