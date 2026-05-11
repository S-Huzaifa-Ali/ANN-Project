"""
Central configuration for the Food Detection & Classification system.
All paths are relative so the project is portable.
Models load on CPU by default.
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DETECTION_MODEL_PATH = os.path.join(BASE_DIR, "Food_Detection_best.pt")
CLASSIFICATION_MODEL_PATH = os.path.join(BASE_DIR, "Food_Classification_best.pth")

NUTRITION_DATA_PATH = os.path.join(BASE_DIR, "food_items_ordered.json")

DEVICE = "cpu"

DETECTION_CONFIDENCE_THRESHOLD = 0.25  
DETECTION_IOU_THRESHOLD = 0.45  

CLASSIFICATION_IMAGE_SIZE = 224  
NUM_CLASSES = 748  
TOP_K = 3    

CLASSIFIER_ARCHITECTURE = "efficientnet_b0"

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
MAX_CONTENT_LENGTH = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}

HOST = "0.0.0.0"
PORT = 5000
DEBUG = True

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
