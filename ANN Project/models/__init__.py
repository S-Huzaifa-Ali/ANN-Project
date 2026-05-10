"""
Models package — contains modular detector and classifier wrappers.
Designed for easy model swapping as training progresses.
"""

from .detector import FoodDetector
from .classifier import FoodClassifier

__all__ = ["FoodDetector", "FoodClassifier"]
