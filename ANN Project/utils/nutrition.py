"""
Nutrition lookup utility — maps predicted food labels to nutritional data.
Supports portion size estimation and fuzzy matching.
"""

import json
import os
import re
from difflib import SequenceMatcher
import config


class NutritionLookup:
    """Loads nutrition data and provides lookup by food name with fuzzy matching."""

    PORTION_MULTIPLIERS = {
        "small": 0.5, "medium": 1.0, "large": 1.5, "extra_large": 2.0
    }

    def __init__(self, data_path=None):
        self.data_path = data_path or config.NUTRITION_DATA_PATH
        self.food_data = []
        self.food_name_map = {}
        self._load_data()

    def _load_data(self):
        """Load nutrition data from JSON file."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Nutrition data not found at: {self.data_path}")
        with open(self.data_path, "r", encoding="utf-8") as f:
            self.food_data = json.load(f)
        for item in self.food_data:
            key = item["food_item"].lower().strip()
            self.food_name_map[key] = item
            self.food_name_map[key.replace("_", " ")] = item
        print(f"[Nutrition] Loaded {len(self.food_data)} food items")

    def _parse_serving_weight_grams(self, serving_size_str):
        """Parse weight in grams from serving_size string. E.g. '3 pieces (~75g)' → 75.0"""
        match = re.search(r"~\s*(\d+\.?\d*)\s*g", serving_size_str, re.IGNORECASE)
        return float(match.group(1)) if match else None

    def _fuzzy_match(self, query, threshold=0.6):
        """Find best matching food item using fuzzy string matching."""
        query_lower = query.lower().strip().replace("_", " ")
        if query_lower in self.food_name_map:
            return self.food_name_map[query_lower]
        best_match, best_score = None, 0
        for name, item in self.food_name_map.items():
            score = SequenceMatcher(None, query_lower, name).ratio()
            if score > best_score:
                best_score = score
                best_match = item
        return best_match if best_score >= threshold else None

    def estimate_portion_size(self, bbox_area_ratio):
        """Estimate portion category from bounding box area ratio."""
        if bbox_area_ratio < 0.05:
            return "small", 0.5
        elif bbox_area_ratio < 0.15:
            return "medium", 1.0
        elif bbox_area_ratio < 0.35:
            return "large", 1.5
        else:
            return "extra_large", 2.0

    def get_nutrition(self, food_name, portion_multiplier=1.0, bbox_area_ratio=None):
        """
        Get nutritional info for a food item, adjusted for portion size.
        Returns dict with adjusted and reference values, or None if not found.
        """
        item = self._fuzzy_match(food_name)
        if item is None:
            return None
        portion_size = "medium"
        if bbox_area_ratio is not None and portion_multiplier == 1.0:
            portion_size, portion_multiplier = self.estimate_portion_size(bbox_area_ratio)
        ref_weight = self._parse_serving_weight_grams(item["serving_size"])
        return {
            "food_item": item["food_item"],
            "serving_size": item["serving_size"],
            "portion_size": portion_size,
            "portion_multiplier": round(portion_multiplier, 2),
            "calories": round(item["calories"] * portion_multiplier, 1),
            "protein_g": round(item["protein_g"] * portion_multiplier, 1),
            "carbs_g": round(item["carbs_g"] * portion_multiplier, 1),
            "fat_g": round(item["fat_g"] * portion_multiplier, 1),
            "reference_calories": item["calories"],
            "reference_protein_g": item["protein_g"],
            "reference_carbs_g": item["carbs_g"],
            "reference_fat_g": item["fat_g"],
            "reference_weight_g": ref_weight,
            "estimated_weight_g": round(ref_weight * portion_multiplier, 1) if ref_weight else None
        }

    def search_foods(self, query, limit=10):
        """Search for food items matching a query string."""
        query_lower = query.lower().strip().replace("_", " ")
        results = []
        for item in self.food_data:
            name = item["food_item"].lower().replace("_", " ")
            score = SequenceMatcher(None, query_lower, name).ratio()
            if query_lower in name:
                score = max(score, 0.8)
            if score > 0.3:
                results.append({"food_item": item["food_item"], "score": round(score, 3),
                                "calories": item["calories"], "serving_size": item["serving_size"]})
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]
