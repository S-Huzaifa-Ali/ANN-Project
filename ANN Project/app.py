"""
Flask web application for Food Detection & Classification.
Serves the frontend and provides REST API endpoints for:
  - /api/detect: Upload image → detect food → classify → return results
  - /api/nutrition: Get nutritional info for a specific food + portion
"""

import os
import io
import base64
import uuid
import traceback
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image

import config
from models.detector import FoodDetector
from models.classifier import FoodClassifier
from utils.nutrition import NutritionLookup

# ----- Initialize Flask App -----
app = Flask(__name__, static_folder="static", static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH

# ----- Load Models (once at startup) -----
print("=" * 60)
print("  Food Detection & Classification System")
print("=" * 60)

# Initialize models with error handling
detector = None
classifier = None
nutrition = None

try:
    detector = FoodDetector()
except Exception as e:
    print(f"[WARNING] Could not load detection model: {e}")

try:
    classifier = FoodClassifier()
except Exception as e:
    print(f"[WARNING] Could not load classification model: {e}")

try:
    nutrition = NutritionLookup()
except Exception as e:
    print(f"[WARNING] Could not load nutrition data: {e}")

print("=" * 60)


def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in config.ALLOWED_EXTENSIONS


def pil_to_base64(pil_image, fmt="JPEG"):
    """Convert a PIL Image to a base64-encoded string."""
    buffer = io.BytesIO()
    pil_image.save(buffer, format=fmt)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# ----- Serve Frontend -----
@app.route("/")
def index():
    """Serve the main HTML page."""
    return send_from_directory("static", "index.html")


# ----- API: Detect & Classify -----
@app.route("/api/detect", methods=["POST"])
def api_detect():
    """
    Upload an image, detect food items, classify each, and return results.

    Request: multipart/form-data with 'image' file
    Response: JSON with detections, classifications, and nutritional data
    """
    # Validate models are loaded
    if detector is None:
        return jsonify({"error": "Detection model not loaded. Check server logs."}), 503
    if classifier is None:
        return jsonify({"error": "Classification model not loaded. Check server logs."}), 503

    # Validate image upload
    if "image" not in request.files:
        return jsonify({"error": "No image file provided. Use 'image' field."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": f"Invalid file type. Allowed: {config.ALLOWED_EXTENSIONS}"}), 400

    try:
        # Read and validate image
        image = Image.open(file.stream).convert("RGB")
        img_width, img_height = image.size

        # Step 1: Run YOLO detection
        detections = detector.detect(image)

        if len(detections) == 0:
            # Return the image but with no detections
            return jsonify({
                "success": True,
                "message": "No food items detected in the image.",
                "image_width": img_width,
                "image_height": img_height,
                "detections": [],
                "original_image": pil_to_base64(image)
            })

        # Step 2: Classify each detected crop and get nutrition
        results = []
        for i, det in enumerate(detections):
            # Classify the cropped food image
            predictions = classifier.classify(det["crop"], top_k=config.TOP_K)

            # Get nutrition for the top prediction
            top_food = predictions[0]["class_name"]
            nutrition_info = None
            if nutrition is not None:
                nutrition_info = nutrition.get_nutrition(
                    top_food,
                    bbox_area_ratio=det["bbox_area_ratio"]
                )

            # Convert crop to base64 for frontend display
            crop_b64 = pil_to_base64(det["crop"])

            results.append({
                "id": i,
                "bbox": det["bbox"],
                "detection_confidence": det["confidence"],
                "bbox_area_ratio": det["bbox_area_ratio"],
                "crop_image": crop_b64,
                "predictions": predictions,
                "nutrition": nutrition_info
            })

        # Encode original image as base64
        original_b64 = pil_to_base64(image)

        return jsonify({
            "success": True,
            "message": f"Detected {len(results)} food item(s).",
            "image_width": img_width,
            "image_height": img_height,
            "detections": results,
            "original_image": original_b64
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500


# ----- API: Get Nutrition -----
@app.route("/api/nutrition", methods=["POST"])
def api_nutrition():
    """
    Get nutritional info for a specific food with portion adjustment.

    Request JSON: { "food_name": str, "portion_multiplier": float }
    Response JSON: nutritional data
    """
    if nutrition is None:
        return jsonify({"error": "Nutrition data not loaded."}), 503

    data = request.get_json()
    if not data or "food_name" not in data:
        return jsonify({"error": "Missing 'food_name' in request body."}), 400

    food_name = data["food_name"]
    portion_multiplier = data.get("portion_multiplier", 1.0)

    result = nutrition.get_nutrition(food_name, portion_multiplier=portion_multiplier)
    if result is None:
        return jsonify({"error": f"Food item '{food_name}' not found in database."}), 404

    return jsonify({"success": True, "nutrition": result})


# ----- API: Search Foods -----
@app.route("/api/search", methods=["GET"])
def api_search():
    """Search food items by name."""
    query = request.args.get("q", "")
    if not query:
        return jsonify({"error": "Missing query parameter 'q'."}), 400

    if nutrition is None:
        return jsonify({"error": "Nutrition data not loaded."}), 503

    results = nutrition.search_foods(query)
    return jsonify({"success": True, "results": results})


# ----- Run Server -----
if __name__ == "__main__":
    print(f"\n  Starting server at http://localhost:{config.PORT}")
    print(f"  Open your browser and navigate to http://localhost:{config.PORT}\n")
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG)
