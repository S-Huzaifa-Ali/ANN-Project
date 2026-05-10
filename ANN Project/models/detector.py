"""
Food Detector module — wraps Ultralytics YOLO for food detection.

Usage:
    detector = FoodDetector(model_path="Food_Detection_best.pt")
    results = detector.detect(image)
    # results = list of dicts with keys: bbox, confidence, crop
"""

import os
import numpy as np
from PIL import Image

# Import config for defaults
import config


class FoodDetector:
    """
    Modular food detection wrapper using Ultralytics YOLO.
    Can be replaced with any detection model by implementing
    the same detect() interface.
    """

    def __init__(self, model_path=None, device=None, conf_threshold=None, iou_threshold=None):
        """
        Initialize the food detector.

        Args:
            model_path (str): Path to the YOLO .pt model file.
            device (str): Device to run inference on ("cpu" or "cuda").
            conf_threshold (float): Minimum confidence for detections.
            iou_threshold (float): IoU threshold for non-max suppression.
        """
        self.model_path = model_path or config.DETECTION_MODEL_PATH
        self.device = device or config.DEVICE
        self.conf_threshold = conf_threshold or config.DETECTION_CONFIDENCE_THRESHOLD
        self.iou_threshold = iou_threshold or config.DETECTION_IOU_THRESHOLD
        self.model = None

        # Load the model
        self._load_model()

    def _load_model(self):
        """Load the YOLO model from disk."""
        try:
            from ultralytics import YOLO
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(
                    f"Detection model not found at: {self.model_path}. "
                    "Please ensure 'Food_Detection_best.pt' is in the project root."
                )
            self.model = YOLO(self.model_path)
            print(f"[Detector] Loaded YOLO model from {self.model_path}")
        except ImportError:
            raise ImportError(
                "Ultralytics is not installed. Run: pip install ultralytics"
            )

    def detect(self, image):
        """
        Run food detection on an image.

        Args:
            image: PIL Image or numpy array or file path.

        Returns:
            list of dict: Each dict contains:
                - 'bbox': [x1, y1, x2, y2] bounding box coordinates
                - 'confidence': float detection confidence
                - 'crop': PIL Image of the cropped detection
                - 'class_name': str class name from YOLO (if available)
                - 'bbox_area_ratio': float ratio of bbox area to image area
        """
        if self.model is None:
            raise RuntimeError("Detection model is not loaded.")

        # Convert to PIL Image if needed
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # Get image dimensions for area ratio calculation
        img_width, img_height = image.size
        img_area = img_width * img_height

        # Run YOLO inference
        results = self.model.predict(
            source=image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False
        )

        detections = []

        # Parse results
        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue

            for box in result.boxes:
                # Extract bounding box coordinates (xyxy format)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()

                # Extract confidence score
                confidence = float(box.conf[0].cpu().numpy())

                # Get class name if available
                cls_id = int(box.cls[0].cpu().numpy())
                class_name = result.names.get(cls_id, "food") if result.names else "food"

                # Crop the detected region from the original image
                # Add small padding for better classification
                pad = 10
                crop_x1 = max(0, int(x1) - pad)
                crop_y1 = max(0, int(y1) - pad)
                crop_x2 = min(img_width, int(x2) + pad)
                crop_y2 = min(img_height, int(y2) + pad)
                crop = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))

                # Calculate bounding box area ratio (used for portion estimation)
                bbox_area = (x2 - x1) * (y2 - y1)
                bbox_area_ratio = bbox_area / img_area if img_area > 0 else 0

                detections.append({
                    "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    "confidence": round(confidence, 4),
                    "crop": crop,
                    "class_name": class_name,
                    "bbox_area_ratio": round(bbox_area_ratio, 4)
                })

        # Sort by confidence (highest first)
        detections.sort(key=lambda d: d["confidence"], reverse=True)

        print(f"[Detector] Found {len(detections)} food item(s)")
        return detections
