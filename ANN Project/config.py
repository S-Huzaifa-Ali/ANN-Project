"""
Central configuration for the Food Detection & Classification system.
All paths are relative so the project is portable.
Models load on CPU by default.
"""

import os

# ----- Base Directory (project root) -----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ----- Model Paths (relative to project root) -----
DETECTION_MODEL_PATH = os.path.join(BASE_DIR, "Food_Detection_best.pt")
CLASSIFICATION_MODEL_PATH = os.path.join(BASE_DIR, "Food_Classification_best.pth")

# ----- Nutrition Data -----
NUTRITION_DATA_PATH = os.path.join(BASE_DIR, "food_items_ordered.json")

# ----- Device Configuration -----
# Use CPU by default; change to "cuda" if GPU is available
DEVICE = "cpu"

# ----- Detection Settings -----
DETECTION_CONFIDENCE_THRESHOLD = 0.25  # Minimum confidence for YOLO detections
DETECTION_IOU_THRESHOLD = 0.45         # IoU threshold for NMS

# ----- Classification Settings -----
CLASSIFICATION_IMAGE_SIZE = 224        # Input size for the classification model
NUM_CLASSES = 748                       # Number of food classes
TOP_K = 3                              # Number of top predictions to return

# ----- Classification Model Architecture -----
# Options: "efficientnet_b0", "resnet50", "resnet101", etc. (timm model names)
# Change this when swapping the classification model
CLASSIFIER_ARCHITECTURE = "efficientnet_b0"

# ----- Upload Settings -----
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max upload size
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}

# ----- Server Settings -----
HOST = "0.0.0.0"
PORT = 5000
DEBUG = True

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
