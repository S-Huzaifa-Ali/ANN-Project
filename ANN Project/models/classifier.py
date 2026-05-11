import os
import json
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

import config


class FoodClassifier:

    def __init__(self, model_path=None, device=None, architecture=None):
        self.model_path = model_path or config.CLASSIFICATION_MODEL_PATH
        self.device = device or config.DEVICE
        self.architecture = architecture or config.CLASSIFIER_ARCHITECTURE
        self.model = None
        self.class_names = []
        self.num_classes = config.NUM_CLASSES

        self.transform = transforms.Compose([
            transforms.Resize((config.CLASSIFICATION_IMAGE_SIZE, config.CLASSIFICATION_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self._load_model()

    def _load_class_names_from_json(self):
        """Fallback: load class names from food_items_ordered.json, sorted by id."""
        try:
            with open(config.NUTRITION_DATA_PATH, "r", encoding="utf-8") as f:
                food_data = json.load(f)
            return [item["food_item"] for item in sorted(food_data, key=lambda x: x["id"])]
        except FileNotFoundError:
            return [f"class_{i}" for i in range(self.num_classes)]

    def _build_model(self, num_classes, architecture=None):
        arch = (architecture or self.architecture).lower()

        try:
            import timm
            model = timm.create_model(arch, pretrained=False, num_classes=num_classes)
            print(f"[Classifier] Built {arch} model via timm with {num_classes} classes")
            return model
        except ImportError:
            print("[Classifier] timm not installed, falling back to torchvision")
        except Exception as e:
            print(f"[Classifier] timm failed for '{arch}': {e}, trying torchvision")

        from torchvision import models
        import torch.nn as nn

        if "resnet50" in arch:
            model = models.resnet50(weights=None)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif "resnet18" in arch:
            model = models.resnet18(weights=None)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif "efficientnet_b0" in arch:
            model = models.efficientnet_b0(weights=None)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        else:
            raise ValueError(f"Unsupported architecture: '{arch}'. Install timm: pip install timm")

        print(f"[Classifier] Built {arch} model via torchvision with {num_classes} classes")
        return model

    def _load_model(self):
        """Load the PyTorch classification model from the checkpoint file."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Classification model not found at: {self.model_path}. "
                "Please ensure 'Food_Classification_best.pth' is in the project root."
            )

        try:
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)

            if isinstance(checkpoint, dict):
                if "classes" in checkpoint:
                    self.class_names = checkpoint["classes"]
                    print(f"[Classifier] Loaded {len(self.class_names)} class names from checkpoint")

                if "num_classes" in checkpoint:
                    self.num_classes = checkpoint["num_classes"]

                epoch = checkpoint.get("epoch", "?")
                val_acc = checkpoint.get("val_acc", "?")
                if isinstance(val_acc, float):
                    val_acc = f"{val_acc:.2%}"
                print(f"[Classifier] Checkpoint info: epoch={epoch}, val_acc={val_acc}")

                if "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                elif "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint

            if not self.class_names:
                self.class_names = self._load_class_names_from_json()
                print(f"[Classifier] Loaded {len(self.class_names)} class names from JSON")

            keys_str = " ".join(state_dict.keys())
            if "conv_stem" in keys_str and "blocks" in keys_str:
                self.architecture = "efficientnet_b0"
                print("[Classifier] Auto-detected architecture: timm EfficientNet-B0")
            elif "fc.weight" in state_dict and "layer4" in keys_str:
                self.architecture = "resnet50"
                print("[Classifier] Auto-detected architecture: ResNet50")

            self.model = self._build_model(self.num_classes, self.architecture)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()

            print(f"[Classifier] Model loaded successfully from {self.model_path}")
            print(f"[Classifier] Note: Model is at early training (epoch {epoch}) — predictions may be inaccurate")

        except Exception as e:
            raise RuntimeError(f"Failed to load classification model: {e}")

    def classify(self, image, top_k=None):
        if self.model is None:
            raise RuntimeError("Classification model is not loaded.")

        top_k = top_k or config.TOP_K

        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert("RGB")

        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = F.softmax(output, dim=1)

        top_probs, top_indices = torch.topk(probabilities, k=min(top_k, self.num_classes), dim=1)

        predictions = []
        for i in range(top_probs.shape[1]):
            idx = top_indices[0, i].item()
            prob = top_probs[0, i].item()
            name = self.class_names[idx] if idx < len(self.class_names) else f"Unknown_{idx}"

            predictions.append({
                "class_index": idx,
                "class_name": name,
                "confidence": round(prob, 4)
            })

        return predictions
