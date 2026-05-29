import os
import uuid
import cv2
from flask import Flask, render_template, request, jsonify, send_from_directory

from werkzeug.utils import secure_filename

import config
import db
from verify_logo import load_all, verify_image
from verify_video import process_video

#  Папки для загрузок и результатов
UPLOAD_FOLDER  = os.path.join(config.WORK_DIR, "web_uploads")
RESULTS_FOLDER = os.path.join(config.WORK_DIR, "web_results")
os.makedirs(UPLOAD_FOLDER,  exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

ALLOWED_IMAGES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALLOWED_VIDEOS = {".mp4", ".avi", ".mov", ".mkv"}

#  Flask app
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB

#  Инициализация БД и загрузка моделей
db.init_db()
print("Загрузка моделей...")
_detector, _model, _classes, _reference, _device, _ = load_all()
print(f"Готово. Брендов в базе: {len(_classes)}")

#  Вспомогательные функции
def ext(filename: str) -> str:
    return os.path.splitext(filename.lower())[1]

def is_image(filename: str) -> bool:
    return ext(filename) in ALLOWED_IMAGES

def is_video(filename: str) -> bool:
    return ext(filename) in ALLOWED_VIDEOS

#  Маршруты
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/history")
def history():
    return render_template("history.html")

@app.route("/api/result/<filename>")
def serve_result(filename):
    return send_from_directory(RESULTS_FOLDER, filename)

@app.route("/api/history")
def api_history():
    limit = int(request.args.get("limit", 200))
    rows  = db.get_history(limit)
    stats = db.get_stats()
    return jsonify({"rows": rows, "stats": stats})

@app.route("/api/verify", methods=["POST"])
def verify():
    if "files" not in request.files:
        return jsonify({"error": "Файлы не переданы"}), 400

    threshold = float(request.form.get("threshold", config.FAKE_THRESHOLD))
    conf      = float(request.form.get("conf", 0.3))

    files   = request.files.getlist("files")
    results = []

    for file in files:
        if not file.filename:
            continue

        safe_name  = secure_filename(file.filename)
        uid        = uuid.uuid4().hex[:8]
        saved_name = f"{uid}_{safe_name}"
        saved_path = os.path.join(UPLOAD_FOLDER, saved_name)
        file.save(saved_path)

        try:
            if is_image(safe_name):
                result = _process_image(saved_path, safe_name, threshold, conf)
            elif is_video(safe_name):
                result = _process_video(saved_path, safe_name, threshold, conf)
            else:
                result = {"filename": safe_name,
                          "error": "Неподдерживаемый формат файла"}
        except Exception as e:
            result = {"filename": safe_name, "error": str(e)}

        # Логируем в SQLite
        if "error" not in result:
            db.log_check(
                filename  = safe_name,
                file_type = result.get("type", "image"),
                detections = result.get("detections", []),
                threshold  = threshold,
            )

        results.append(result)

    return jsonify({"results": results})

def _process_image(saved_path, saved_name, threshold, conf):
    result_name = f"result_{saved_name}"
    result_path = os.path.join(RESULTS_FOLDER, result_name)

    result_img, detections = verify_image(
        saved_path, _detector, _model, _classes, _reference, _device,
        conf_det=conf, threshold=threshold,
    )
    cv2.imwrite(result_path, result_img)

    return {
        "filename":   saved_name,
        "type":       "image",
        "result_url": f"/api/result/{result_name}",
        "detections": [
            {
                "brand":      d["brand"],
                "is_fake":    d["is_fake"],
                "similarity": round(d["similarity"], 3),
                "cls_conf":   round(d["cls_conf"], 3),
            }
            for d in detections
        ],
    }

def _process_video(saved_path, saved_name, threshold, conf):
    result_name = f"result_{os.path.splitext(saved_name)[0]}.mp4"
    result_path = os.path.join(RESULTS_FOLDER, result_name)

    process_video(
        saved_path, _detector, _model, _classes, _reference, _device,
        conf_det=conf, threshold=threshold,
        every=5, show=False,
        output_path=result_path,
    )

    return {
        "filename":   saved_name,
        "type":       "video",
        "result_url": f"/api/result/{result_name}",
        "detections": [],
    }

#  Точка входа
if __name__ == "__main__":
    print("\n  Открой браузер: http://localhost:5000\n")
    app.run(debug=False, host="0.0.0.0", port=5000)