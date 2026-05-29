FROM python:3.11-slim

# Системные зависимости для OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY Requirements.txt .
RUN pip install --no-cache-dir -r Requirements.txt

COPY *.py ./
COPY templates/ templates/

RUN mkdir -p /data/web_uploads /data/web_results

EXPOSE 5000

ENV FLASK_ENV=production

CMD ["python", "app.py"]