import cv2
import numpy as np
import torch
import os
from ultralytics import YOLO
from .csrnet_model import CSRNetLight
from .model_manager import ModelManager
from .detection_utils import FrameDimensionHandler
import time

class HybridDetector:
    def __init__(self, yolo_model_path, csrnet_model_path, density_threshold=20, yolo_model_size='n'):
        """
        Initialize the hybrid detector with automatic GPU/CPU selection
        
        Args:
            yolo_model_path: Path to the YOLO model file
            csrnet_model_path: Path to the CSRNet model file
            density_threshold: Threshold count to switch from YOLO to hybrid mode
            yolo_model_size: Size of YOLO model to use ('n', 's', 'm', 'l', 'x')
        """
        # Initialize model manager and dimension handler
        self.model_manager = ModelManager()
        self.dimension_handler = FrameDimensionHandler()
        
        # Setup CUDA-enabled ONNX Runtime if needed
        if yolo_model_path.endswith('.onnx'):
            onnx_success = self.model_manager.setup_onnx_runtime()
            if not onnx_success:
                print("ONNX Runtime setup failed, falling back to PyTorch...")
                pt_path = self.model_manager.convert_to_pytorch(yolo_model_size)
                if pt_path:
                    yolo_model_path = pt_path
                else:
                    raise RuntimeError("Failed to set up both ONNX Runtime and PyTorch fallback")
        
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
        
        try:
            # Initialize YOLO model with performance monitoring
            if yolo_model_path.endswith('.onnx'):
                try:
                    # Load ONNX model with proper providers
                    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device.type == 'cuda' else ['CPUExecutionProvider']
                    print(f"Loading {yolo_model_path} with providers: {providers}")
                    self.yolo_model = YOLO(yolo_model_path, task='detect')
                    self.yolo_model.conf = 0.3
                    # Test ONNX model
                    dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
                    _ = self.yolo_model.predict(source=dummy_image, device=self.device.type)
                    print(f"Successfully loaded ONNX model on {self.device.type.upper()}")
                except Exception as e:
                    print(f"ONNX model error: {str(e)}. Converting to PyTorch...")
                    pt_path = self.model_manager.convert_to_pytorch(yolo_model_size)
                    if pt_path:
                        self.yolo_model = YOLO(pt_path, task='detect')
                        self.yolo_model.conf = 0.3
                    else:
                        raise RuntimeError("Failed to convert ONNX model to PyTorch")
            else:
                self.yolo_model = YOLO(yolo_model_path, task='detect')
                self.yolo_model.conf = 0.3
            
            # Monitor initial performance
            self.model_manager.monitor_performance('yolo', self.device.type)
            print(f"Successfully loaded YOLO model on {self.device.type.upper()}")
            
        except Exception as e:
            print(f"Error loading YOLO model: {str(e)}")
            # Try fallback mechanisms
            try:
                fallback_device = self.model_manager.implement_fallback('cuda_error' if 'CUDA' in str(e) else 'onnx_error', yolo_model_path)
                if fallback_device == 'cpu':
                    self.device = torch.device('cpu')
                    print("Falling back to CPU")
                # Try one more time with PyTorch model
                pt_path = self.model_manager.convert_to_pytorch(yolo_model_size)
                if pt_path:
                    self.yolo_model = YOLO(pt_path, task='detect')
                    self.yolo_model.conf = 0.3
                else:
                    raise RuntimeError("Failed to load YOLO model after all fallback attempts")
            except Exception as e2:
                print(f"Fatal error: {str(e2)}")
                raise
        
        try:
            # Initialize CSRNet model with performance monitoring
            self.csrnet_model = CSRNetLight().to(self.device)
            if os.path.exists(csrnet_model_path):
                state_dict = torch.load(csrnet_model_path, map_location=self.device)
                self.csrnet_model.load_state_dict(state_dict)
            else:
                print(f"Warning: CSRNet model not found at {csrnet_model_path}")
                print("Using initialized weights")
                os.makedirs(os.path.dirname(csrnet_model_path), exist_ok=True)
                torch.save(self.csrnet_model.state_dict(), csrnet_model_path)
            
            self.csrnet_model.eval()
            
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
                self.csrnet_model = torch.nn.DataParallel(self.csrnet_model)
            
            # Monitor initial performance
            self.model_manager.monitor_performance('csrnet', self.device.type)
            print(f"Successfully loaded CSRNet model on {self.device.type.upper()}")
            
        except Exception as e:
            print(f"Error loading CSRNet model: {str(e)}")
            fallback_device = self.model_manager.implement_fallback('memory_error', csrnet_model_path)
            if fallback_device == 'cpu':
                self.device = torch.device('cpu')
                print("Falling back to CPU")
            print("Using initialized weights")
            os.makedirs(os.path.dirname(csrnet_model_path), exist_ok=True)
            torch.save(self.csrnet_model.state_dict(), csrnet_model_path)
        
        # Initialize other parameters
        self.yolo_model_size = yolo_model_size
        self.yolo_model_path = yolo_model_path
        self.density_threshold = density_threshold
        self.lower_threshold = max(1, int(density_threshold * 0.8))
        self.current_mode = 'yolo'
        self.transition_weight = 0.0
        self.transition_speed = 0.2
        self.recent_counts = []
        self.max_recent_counts = 5
        
    def set_density_threshold(self, threshold):
        """
        Update the density threshold
        
        Args:
            threshold: New threshold value
        """
        if threshold < 1:
            raise ValueError("Density threshold must be at least 1")
        self.density_threshold = threshold
        print(f"Density threshold updated to {threshold}")
    
    def change_yolo_model(self, model_path, model_size='n'):
        """
        Change the YOLO model being used
        
        Args:
            model_path: Path to the new YOLO model file
            model_size: Size of the new model ('n', 's', 'm', 'l', 'x')
        
        Returns:
            bool: Success status
        """
        try:
            # Load new model with explicit task parameter
            new_model = YOLO(model_path, task='detect')
            new_model.conf = self.yolo_model.conf
            
            # Update model if successfully loaded
            self.yolo_model = new_model
            self.yolo_model_path = model_path
            self.yolo_model_size = model_size
            print(f"Successfully changed YOLO model to {model_size}")
            return True
        except Exception as e:
            print(f"Error changing YOLO model: {str(e)}")
            return False
        
    def preprocess_for_csrnet(self, frame):
        """Preprocess frame for CSRNet input"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Use dimension handler to resize
        resized = self.dimension_handler.resize_frame(gray, target='csrnet')
        # Normalize
        normalized = resized / 255.0
        # Add batch and channel dimensions
        tensor = torch.from_numpy(normalized).float()
        tensor = tensor.unsqueeze(0).unsqueeze(0)
        return tensor
    
    def detect(self, frame, roi_points=None):
        """Optimized detection with automatic dimension handling and ROI support"""
        try:
            # Update dimension handler with current frame
            self.dimension_handler.set_source_dimensions(frame)
            
            # Convert ROI points if provided
            yolo_roi = None
            csrnet_roi = None
            if roi_points:
                yolo_roi = self.dimension_handler.convert_roi_points(roi_points, source='source', target='yolo')
                csrnet_roi = self.dimension_handler.convert_roi_points(roi_points, source='source', target='csrnet')
            
            # Prepare frame for YOLO
            yolo_frame = self.dimension_handler.resize_frame(frame, target='yolo')
            
            # Run YOLO detection
            yolo_results = self.yolo_model(yolo_frame)[0]
            
            # Filter detections by ROI if provided
            if yolo_roi:
                # Convert detections to points and check if they're in ROI
                boxes = yolo_results.boxes.xyxy.cpu().numpy()
                filtered_boxes = []
                for box in boxes:
                    center_x = (box[0] + box[2]) / 2
                    center_y = (box[1] + box[3]) / 2
                    if cv2.pointPolygonTest(np.array(yolo_roi), (center_x, center_y), False) >= 0:
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
            
            # Determine if we should use hybrid mode
            if stable_count > self.density_threshold:
                # Prepare frame for CSRNet
                csrnet_input = self.preprocess_for_csrnet(frame)
                
                # Move to device
                csrnet_input = csrnet_input.to(self.device)
                
                # Get density map
                with torch.no_grad():
                    density_map = self.csrnet_model(csrnet_input)
                
                # Apply ROI mask if provided
                if csrnet_roi:
                    mask = np.zeros(density_map.shape[2:], dtype=np.float32)
                    cv2.fillPoly(mask, [np.array(csrnet_roi)], 1)
                    density_map = density_map * torch.from_numpy(mask).to(self.device)
                
                # Calculate crowd count from density map
                csrnet_count = int(density_map.sum().item())
                
                # Smooth transition between modes
                if self.current_mode == 'yolo':
                    self.transition_weight = min(1.0, self.transition_weight + self.transition_speed)
                self.current_mode = 'hybrid'
                
                # Combine counts with transition weight
                final_count = int(yolo_count * (1 - self.transition_weight) + csrnet_count * self.transition_weight)
                
                return final_count, density_map.cpu().numpy()[0, 0], 'hybrid'
            else:
                # Reset transition when returning to YOLO mode
                if self.current_mode == 'hybrid':
                    self.transition_weight = max(0.0, self.transition_weight - self.transition_speed)
                if self.transition_weight > 0:
                    # Still in transition
                    csrnet_input = self.preprocess_for_csrnet(frame)
                    csrnet_input = csrnet_input.to(self.device)
                    with torch.no_grad():
                        density_map = self.csrnet_model(csrnet_input)
                    csrnet_count = int(density_map.sum().item())
                    final_count = int(yolo_count * (1 - self.transition_weight) + csrnet_count * self.transition_weight)
                    return final_count, density_map.cpu().numpy()[0, 0], 'transitioning'
                else:
                    self.current_mode = 'yolo'
                    return yolo_count, None, 'yolo'
                    
        except Exception as e:
            print(f"Error in detection: {str(e)}")
            return 0, None, 'error'
            
    def visualize(self, frame, count, density_map=None, mode=None, roi_points=None):
        """Visualize detection results with ROI"""
        try:
            # Draw ROI if provided
            if roi_points:
                frame, area_percent = draw_roi(frame, roi_points)
                
            # Add detection mode and count
            mode_text = f"Mode: {mode.upper()}" if mode else ""
            cv2.putText(frame, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Count: {count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Overlay density map if available
            if density_map is not None and mode in ['hybrid', 'transitioning']:
                # Resize density map to match frame size
                density_vis = cv2.resize(density_map, (frame.shape[1], frame.shape[0]))
                density_vis = (density_vis - density_vis.min()) / (density_vis.max() - density_vis.min() + 1e-8)
                density_vis = (density_vis * 255).astype(np.uint8)
                density_vis = cv2.applyColorMap(density_vis, cv2.COLORMAP_JET)
                
                # Create overlay
                overlay = frame.copy()
                cv2.addWeighted(density_vis, 0.3, frame, 0.7, 0, overlay)
                
                # Apply ROI mask if provided
                if roi_points:
                    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                    cv2.fillPoly(mask, [np.array(roi_points)], 255)
                    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                    frame = cv2.bitwise_and(frame, mask)
                    overlay = cv2.bitwise_and(overlay, mask)
                    frame = cv2.add(frame, cv2.bitwise_and(overlay, cv2.bitwise_not(mask)))
                else:
                    frame = overlay
            
            return frame
            
        except Exception as e:
            print(f"Error in visualization: {str(e)}")
            return frame 