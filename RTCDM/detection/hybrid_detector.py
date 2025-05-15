import cv2
import numpy as np
import torch
import os
import threading
from ultralytics import YOLO
from .model_manager import ModelManager
from .detection_utils import FrameDimensionHandler, draw_roi
import time


class HybridDetector:
    def __init__(self, yolo_model_path, density_threshold=20, yolo_model_size='n'):
        """
        Initialize the hybrid detector with automatic GPU/CPU selection
        
        Args:
            yolo_model_path: Path to the YOLO model file
            density_threshold: Threshold count to switch from YOLOv8n to YOLOv8m
            yolo_model_size: Size of YOLO model to use ('n', 's', 'm', 'l', 'x')
        """
        # Initialize model manager and dimension handler
        self.model_manager = ModelManager()
        self.dimension_handler = FrameDimensionHandler()
        
        # Initialize model paths
        self.yolo_model_path = yolo_model_path
        self.yolo_m_model_path = yolo_model_path.replace('n.pt', 'm.pt').replace('n.onnx', 'm.onnx')
        if not os.path.exists(self.yolo_m_model_path):
            print(f"Warning: YOLOv8m model not found at {self.yolo_m_model_path}")
            self.yolo_m_model_path = None
        
        # Initialize other parameters
        self.yolo_model_size = yolo_model_size
        self.density_threshold = density_threshold
        self.lower_threshold = max(1, int(density_threshold * 0.6))
        self.current_mode = 'yolo_n'
        self.transition_weight = 0.0
        self.transition_speed = 0.1  # Reduced for smoother transitions
        self.recent_counts = []
        self.max_recent_counts = 5
        self.min_mode_duration = 2.0  # Minimum seconds in each mode
        self.last_mode_switch = time.time()
        
        # Initialize models
        self.yolo_model = None
        self.yolo_m_model = None
        self.device = None
        
        # Initialize threading lock for thread safety
        self.lock = threading.Lock()
        
        # Load models
        self._initialize_models()

    def _initialize_models(self):
        """Initialize both YOLO models with proper error handling"""
        try:
            # Check GPU availability
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if self.device.type == 'cuda':
                print(f"Using GPU: {torch.cuda.get_device_name(0)}")
                print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
                torch.cuda.set_device(0)
                torch.backends.cudnn.benchmark = True
            else:
                print("GPU not available. Using CPU.")
                torch.set_num_threads(4)
            
            # Load primary model
            self.yolo_model = YOLO(self.yolo_model_path, task='detect')
            self.yolo_model.conf = 0.6
            
            # Load YOLOv8m model if available
            if self.yolo_m_model_path and os.path.exists(self.yolo_m_model_path):
                self.yolo_m_model = YOLO(self.yolo_m_model_path, task='detect')
                self.yolo_m_model.conf = 0.6
                print("Successfully loaded both YOLOv8n and YOLOv8m models")
            else:
                print("YOLOv8m model not available, running in single-model mode")
            
            # Monitor initial performance
            self.model_manager.monitor_performance('yolo', self.device.type)
            
        except Exception as e:
            print(f"Error initializing models: {str(e)}")
            # Try fallback mechanisms
            try:
                fallback_device = self.model_manager.implement_fallback('cuda_error' if 'CUDA' in str(e) else 'onnx_error', self.yolo_model_path)
                if fallback_device == 'cpu':
                    self.device = torch.device('cpu')
                    print("Falling back to CPU")
                    # Retry loading models on CPU
                    self.yolo_model = YOLO(self.yolo_model_path, task='detect')
                    self.yolo_model.conf = 0.6
                    if self.yolo_m_model_path and os.path.exists(self.yolo_m_model_path):
                        self.yolo_m_model = YOLO(self.yolo_m_model_path, task='detect')
                        self.yolo_m_model.conf = 0.6
            except Exception as e2:
                print(f"Fatal error during model initialization: {str(e2)}")
                raise
        
        # Note: All parameters are initialized in the constructor, no need to reinitialize here

    def set_density_threshold(self, threshold):
        """
        Update the density threshold and lower threshold, thread-safe
        """
        with self.lock:
            if threshold < 1:
                raise ValueError("Density threshold must be at least 1")
            self.density_threshold = threshold
            self.lower_threshold = max(1, int(threshold * 0.6))
            print(f"Density threshold updated to {threshold}, lower threshold set to {self.lower_threshold}")
    
    def change_yolo_model(self, model_path, model_size='n'):
        """
        Change the YOLO model being used, thread-safe
        """
        with self.lock:
            try:
                # Load new model with explicit task parameter
                new_model = YOLO(model_path, task='detect')
                new_model.conf = self.yolo_model.conf if hasattr(self, 'yolo_model') else 0.6
                
                # Update model if successfully loaded
                self.yolo_model = new_model
                self.yolo_model_path = model_path
                self.yolo_model_size = model_size
                print(f"Successfully changed YOLO model to {model_size}")
                return True
            except Exception as e:
                print(f"Error changing YOLO model: {str(e)}")
                return False
            
    def load_yolo_m_model(self, model_path):
        """
        Load YOLOv8m model for high-density scenarios, thread-safe
        """
        with self.lock:
            try:
                self.yolo_m_model = YOLO(model_path, task='detect')
                self.yolo_m_model.conf = 0.6
                self.yolo_m_model_path = model_path
                print("Successfully loaded YOLOv8m model")
                return True
            except Exception as e:
                print(f"Error loading YOLOv8m model: {str(e)}")
                self.yolo_m_model = None
                return False
        
    def detect(self, frame, roi_points=None):
        """Optimized detection with automatic dimension handling and ROI support, thread-safe model switching"""
        with self.lock:
            try:
                # Update dimension handler with current frame
                self.dimension_handler.set_source_dimensions(frame)
                
                # Convert ROI points if provided
                yolo_roi = None
                if roi_points and len(roi_points) >= 3:
                    # Debug log original ROI
                    print(f"[DEBUG] Original ROI points: {roi_points}")
                    # Convert ROI points to YOLO input size
                    yolo_roi = self.dimension_handler.convert_roi_points(roi_points, source='source', target='yolo')
                    print(f"[DEBUG] Converted ROI points for YOLO: {yolo_roi}")
                    yolo_roi = np.array(yolo_roi, dtype=np.int32)
                
                # Prepare frame for YOLO
                yolo_frame = self.dimension_handler.resize_frame(frame, target='yolo')
                print(f"[DEBUG] YOLO input frame shape: {yolo_frame.shape}")
                
                # Draw the converted ROI on the YOLO input frame for debugging
                if yolo_roi is not None and len(yolo_roi) >= 3:
                    debug_frame = yolo_frame.copy()
                    cv2.polylines(debug_frame, [yolo_roi], isClosed=True, color=(0,0,255), thickness=2)
                    cv2.imwrite("debug_roi_yolo_frame.jpg", debug_frame)
                    print("[DEBUG] Saved debug ROI overlay image as debug_roi_yolo_frame.jpg")
                
                # Run YOLO detection
                yolo_results = self.yolo_model(yolo_frame)[0]
                
                # Filter detections by ROI if provided
                if yolo_roi is not None:
                    boxes = yolo_results.boxes.xyxy.cpu().numpy()
                    filtered_boxes = []
                    for box in boxes:
                        center_x = (box[0] + box[2]) / 2
                        center_y = (box[1] + box[3]) / 2
                        inside = cv2.pointPolygonTest(yolo_roi, (center_x, center_y), False) >= 0
                        print(f"[DEBUG] Detection center: ({center_x:.1f}, {center_y:.1f}), inside ROI: {inside}")
                        if inside:
                            filtered_boxes.append(box)
                    yolo_count = len(filtered_boxes)
                else:
                    yolo_count = len(yolo_results.boxes)
                
                # Store recent count for stability
                self.recent_counts.append(yolo_count)
                if len(self.recent_counts) > self.max_recent_counts:
                    self.recent_counts.pop(0)
                
                # Use median for stability
                stable_count = int(np.median(self.recent_counts))
                
                # Debug print for transition logic
                print(f"[HybridDetector] Mode: {self.current_mode}, Transition: {self.transition_weight:.2f}, Stable Count: {stable_count}, Threshold: {self.density_threshold}, Lower Threshold: {self.lower_threshold}")
                
                # --- Transition logic similar to CSRNet hybrid ---
                if stable_count > self.density_threshold and self.yolo_m_model is not None:
                    # Prepare for transition to yolo_m
                    if self.current_mode == 'yolo_n' or self.current_mode == 'transitioning':
                        self.transition_weight = min(1.0, self.transition_weight + self.transition_speed)
                        if self.transition_weight >= 1.0:
                            self.current_mode = 'yolo_m'
                        else:
                            self.current_mode = 'transitioning'
                    # Run both models for smooth transition
                    yolo_m_results = self.yolo_m_model(yolo_frame)[0]
                    if yolo_roi is not None:
                        boxes = yolo_m_results.boxes.xyxy.cpu().numpy()
                        filtered_boxes = []
                        for box in boxes:
                            center_x = (box[0] + box[2]) / 2
                            center_y = (box[1] + box[3]) / 2
                            if cv2.pointPolygonTest(yolo_roi, (center_x, center_y), False) >= 0:
                                filtered_boxes.append(box)
                        yolo_m_count = len(filtered_boxes)
                    else:
                        yolo_m_count = len(yolo_m_results.boxes)
                    final_count = int(yolo_count * (1 - self.transition_weight) + yolo_m_count * self.transition_weight)
                    mode_str = 'yolo_m' if self.transition_weight >= 1.0 else 'transitioning'
                    return final_count, None, {
                        'mode': mode_str,
                        'model': f'YOLOv8{self.yolo_model_size.upper()} → YOLOv8M',
                        'threshold': self.density_threshold,
                        'transition': self.transition_weight,
                        'hybrid': True
                    }
                elif stable_count < self.lower_threshold:
                    # Prepare for transition back to yolo_n
                    if self.current_mode == 'yolo_m' or self.current_mode == 'transitioning':
                        self.transition_weight = max(0.0, self.transition_weight - self.transition_speed)
                        if self.transition_weight <= 0.0:
                            self.current_mode = 'yolo_n'
                        else:
                            self.current_mode = 'transitioning'
                        # Run both models for smooth transition
                        if self.yolo_m_model is not None:
                            yolo_m_results = self.yolo_m_model(yolo_frame)[0]
                            if yolo_roi is not None:
                                boxes = yolo_m_results.boxes.xyxy.cpu().numpy()
                                filtered_boxes = []
                                for box in boxes:
                                    center_x = (box[0] + box[2]) / 2
                                    center_y = (box[1] + box[3]) / 2
                                    if cv2.pointPolygonTest(yolo_roi, (center_x, center_y), False) >= 0:
                                        filtered_boxes.append(box)
                                yolo_m_count = len(filtered_boxes)
                            else:
                                yolo_m_count = len(yolo_m_results.boxes)
                            final_count = int(yolo_count * (1 - self.transition_weight) + yolo_m_count * self.transition_weight)
                        else:
                            final_count = yolo_count
                        mode_str = 'yolo_n' if self.transition_weight <= 0.0 else 'transitioning'
                        return final_count, None, {
                            'mode': mode_str,
                            'model': f'YOLOv8{self.yolo_model_size.upper()} → YOLOv8M',
                            'threshold': self.density_threshold,
                            'transition': self.transition_weight,
                            'hybrid': True
                        }
                    else:
                        self.current_mode = 'yolo_n'
                        return yolo_count, None, {
                            'mode': 'yolo_n',
                            'model': f'YOLOv8{self.yolo_model_size.upper()}',
                            'threshold': self.density_threshold,
                            'transition': 0.0,
                            'hybrid': False
                        }
                else:
                    # In between thresholds - maintain current state
                    if self.current_mode == 'yolo_m':
                        return yolo_count, None, {
                            'mode': 'yolo_m',
                            'model': f'YOLOv8M',
                            'threshold': self.density_threshold,
                            'transition': 1.0,
                            'hybrid': True
                        }
                    else:
                        return yolo_count, None, {
                            'mode': 'yolo_n',
                            'model': f'YOLOv8{self.yolo_model_size.upper()}',
                            'threshold': self.density_threshold,
                            'transition': 0.0,
                            'hybrid': False
                        }
                
            except Exception as e:
                print(f"Error in detection: {str(e)}")
                # Return safe fallback values
                return 0, None, {
                    'mode': 'error',
                    'model': 'Error in detection',
                    'threshold': self.density_threshold,
                    'transition': 0.0
                }
            
    def visualize(self, frame, count, density_map=None, mode=None, roi_points=None):
        """Visualize detection results with ROI and model information"""
        try:
            # Draw ROI if provided
            if roi_points:
                frame, area_percent = draw_roi(frame, roi_points)
            
            # Add detection mode and count
            if isinstance(mode, dict):
                # Display model information
                model_text = f"Model: {mode['model']}"
                threshold_text = f"Threshold: {mode['threshold']}"
                if mode['transition'] > 0:
                    transition_text = f"Transition: {int(mode['transition'] * 100)}%"
                else:
                    transition_text = ""
                
                # Add text to frame
                cv2.putText(frame, model_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, threshold_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                if transition_text:
                    cv2.putText(frame, transition_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Count: {count}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                # Fallback for old format
                mode_text = f"Mode: {mode.upper()}" if mode else ""
                cv2.putText(frame, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Count: {count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            return frame
            
        except Exception as e:
            print(f"Error in visualization: {str(e)}")
            return frame