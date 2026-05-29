import argparse
import os
from collections import defaultdict, deque

import cv2
import torch
from tqdm import tqdm

import config
from verify_logo import load_all, classify_crop, draw_box, get_transform


class TrackState:
    def __init__(self, window: int):
        self.window   = window
        self.history  = defaultdict(lambda: deque(maxlen=window))
        self.last_res = {}

    def update(self, track_id: int, result: dict) -> dict:
        self.history[track_id].append(result["is_fake"])
        self.last_res[track_id] = result

        hist     = self.history[track_id]
        smoothed = result.copy()
        smoothed["is_fake"] = sum(hist) > len(hist) / 2   # majority vote
        return smoothed

    def get_last(self, track_id: int):
        return self.last_res.get(track_id)


def process_frame(frame_bgr, detector, model, classes, reference, device,
                  transform, threshold, conf_det, track_state, every, frame_idx,
                  last_boxes):
    img_out = frame_bgr.copy()

    if frame_idx % every == 0:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        track_results = detector.track(
            frame_rgb, persist=True, conf=conf_det, verbose=False
        )[0]

        last_boxes.clear()

        if track_results.boxes.id is None:
            return img_out

        for box, track_id_t in zip(track_results.boxes, track_results.boxes.id.int()):
            track_id = track_id_t.item()
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = frame_rgb[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            raw      = classify_crop(crop, model, classes, reference, device, threshold, transform)
            smoothed = track_state.update(track_id, raw)
            last_boxes[track_id] = ((x1, y1, x2, y2), smoothed)

    for (x1, y1, x2, y2), res in last_boxes.values():
        draw_box(img_out, (x1, y1, x2, y2), res)

    return img_out


def process_video(video_path, detector, model, classes, reference, device,
                  conf_det=0.3, threshold=None, every=5, show=False, output_path=None):

    if threshold is None:
        threshold = config.FAKE_THRESHOLD

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видео: {video_path}")

    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"  {os.path.basename(video_path)}  {width}×{height}  {fps:.0f}fps  {total} кадров")

    if output_path is None:
        base, ext = os.path.splitext(video_path)
        output_path = base + "_verify" + ext

    writer      = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    transform   = get_transform()
    track_state = TrackState(config.VIDEO_SMOOTH_WINDOW)
    last_boxes  = {}

    pbar = tqdm(total=total if total > 0 else None, desc="  Обработка", unit="кадр")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated = process_frame(
            frame, detector, model, classes, reference, device,
            transform, threshold, conf_det, track_state, every, frame_idx, last_boxes
        )
        writer.write(annotated)

        if show:
            cv2.imshow("verify_video — Q для выхода", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print(f"\n✔ Готово: {output_path}")
    return output_path


def process_webcam(detector, model, classes, reference, device,
                   conf_det=0.3, threshold=None, every=3):
    if threshold is None:
        threshold = config.FAKE_THRESHOLD

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise ValueError("Веб-камера не найдена")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  Камера {w}×{h}  |  Q — выход  |  S — скриншот")

    transform    = get_transform()
    track_state  = TrackState(config.VIDEO_SMOOTH_WINDOW)
    last_boxes   = {}
    frame_idx    = 0
    screenshot_n = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated = process_frame(
            frame, detector, model, classes, reference, device,
            transform, threshold, conf_det, track_state, every, frame_idx, last_boxes
        )

        cv2.imshow("LOGOCHECK — Q: выход  S: скриншот", annotated)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("s"):
            screenshot_n += 1
            path = os.path.join(config.WORK_DIR, f"screenshot_{screenshot_n}.jpg")
            cv2.imwrite(path, annotated)
            print(f"  Скриншот: {path}")

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video",  type=str)
    group.add_argument("--webcam", action="store_true")

    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--conf",      type=float, default=0.3)
    parser.add_argument("--every",     type=int,   default=5)
    parser.add_argument("--show",      action="store_true")
    args = parser.parse_args()

    detector, model, classes, reference, device, thr = load_all(args.threshold)
    print(f"Загружено {len(classes)} брендов | порог: {thr} | устройство: {device}\n")

    if args.webcam:
        process_webcam(detector, model, classes, reference, device,
                       conf_det=args.conf, threshold=thr, every=args.every)
    else:
        if not os.path.exists(args.video):
            print(f"[ОШИБКА] Файл не найден: {args.video}")
            return
        process_video(args.video, detector, model, classes, reference, device,
                      conf_det=args.conf, threshold=thr, every=args.every, show=args.show)


if __name__ == "__main__":
    main()