import os
import numpy as np
from PIL import Image

# Import config for defaults
import config


class FoodDetector:

    def __init__(self, model_path=None, device=None, conf_threshold=None, iou_threshold=None):
        self.model_path = model_path or config.DETECTION_MODEL_PATH
        self.device = device or config.DEVICE
        self.conf_threshold = conf_threshold or config.DETECTION_CONFIDENCE_THRESHOLD
        self.iou_threshold = iou_threshold or config.DETECTION_IOU_THRESHOLD
        self.model = None

        self._load_model()

    def _load_model(self):
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
        if self.model is None:
            raise RuntimeError("Detection model is not loaded.")

        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        img_width, img_height = image.size
        img_area = img_width * img_height

        results = self.model.predict(
            source=image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False
        )

        detections = []

        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue

            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()

                confidence = float(box.conf[0].cpu().numpy())

                cls_id = int(box.cls[0].cpu().numpy())
                class_name = result.names.get(cls_id, "food") if result.names else "food"

                pad = 10
                crop_x1 = max(0, int(x1) - pad)
                crop_y1 = max(0, int(y1) - pad)
                crop_x2 = min(img_width, int(x2) + pad)
                crop_y2 = min(img_height, int(y2) + pad)
                crop = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))

                bbox_area = (x2 - x1) * (y2 - y1)
                bbox_area_ratio = bbox_area / img_area if img_area > 0 else 0

                detections.append({
                    "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    "confidence": round(confidence, 4),
                    "crop": crop,
                    "class_name": class_name,
                    "bbox_area_ratio": round(bbox_area_ratio, 4)
                })

        detections.sort(key=lambda d: d["confidence"], reverse=True)

        print(f"[Detector] Found {len(detections)} food item(s)")
        return detections
