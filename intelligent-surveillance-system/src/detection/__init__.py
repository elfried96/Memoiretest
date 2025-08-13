"""Package de détection et tracking pour la surveillance."""

from .yolo_detector import YOLODetector
from .tracking.byte_tracker import BYTETracker, TrackedObject

__all__ = ["YOLODetector", "BYTETracker", "TrackedObject"]