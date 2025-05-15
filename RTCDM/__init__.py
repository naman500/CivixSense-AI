"""
Real-Time Crowd Detection System (RTCDM)
A system for real-time crowd detection and monitoring using computer vision.
"""

__version__ = '1.0.0'

# Import main components for easier access
from .detection.hybrid_detector import HybridDetector
from .detection.yolo_model import YOLOModel
from .detection.detection_utils import FrameDimensionHandler
from .detection.model_manager import ModelManager
from .detection.model_utils import ModelManager as ModelUtils
from .cameras.camera_manager import CameraManager
from .alerts.alert_manager import AlertManager
from .dashboard.dashboard import Dashboard 