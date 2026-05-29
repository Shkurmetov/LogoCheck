import os

SOURCE_DIR = r"D:\DIPLOM\datasets\Clothes"
WORK_DIR   = os.environ.get("LOGO_WORK_DIR", r"D:\DIPLOM\logo_recognition_work_v2")

DET_DATASET          = os.path.join(WORK_DIR, "logo_detector_dataset")
CROPS_DIR            = os.path.join(WORK_DIR, "logo_crops", "train")
DETECTOR_WEIGHTS     = os.path.join(WORK_DIR, "runs", "detect", "logo_detector", "weights", "best.pt")
CLASSIFIER_WEIGHTS   = os.path.join(WORK_DIR, "brand_classifier.pt")
CLASSES_FILE         = os.path.join(WORK_DIR, "classes.txt")
REFERENCE_EMBEDDINGS = os.path.join(WORK_DIR, "reference_embeddings.pt")

DET_EPOCHS   = 300
DET_IMGSZ    = 416
DET_BATCH    = 8
DET_LR       = 0.001
DET_PATIENCE = 50

CLS_EPOCHS      = 30
CLS_BATCH       = 32
CLS_LR          = 1e-4
CLS_IMG_SIZE    = 224
TRAIN_VAL_SPLIT = 0.8
CLS_PATIENCE    = 10   # early stopping

FAKE_THRESHOLD       = 0.50
REFERENCE_N_CLUSTERS = 5
VIDEO_SMOOTH_WINDOW  = 7

RANDOM_SEED = 42