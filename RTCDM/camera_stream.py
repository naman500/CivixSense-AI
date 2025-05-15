from .detection.detection_utils import FrameDimensionHandler
import cv2
from flask import request

class CameraStream:
    def __init__(self, camera_id, url, detector, fps=30, threshold=None):
        self.camera_id = camera_id
        self.url = url
        self.detector = detector
        self.target_fps = fps
        self.threshold = threshold
        self.dimension_handler = FrameDimensionHandler()
        self.roi_points = None
        self.is_running = False
        self.frame_count = 0
        self.total_processing_time = 0
        self.last_frame_time = None
        self.current_fps = 0
        self.current_latency = 0
        
    def update_roi(self, roi_points):
        """Update ROI points for the stream"""
        self.roi_points = roi_points
        
    def process_frame(self, frame):
        """Process a single frame with ROI support"""
        try:
            if frame is None:
                return None, None
                
            # Update dimension handler with current frame
            self.dimension_handler.set_source_dimensions(frame)
            
            # Convert ROI points to model input space if ROI exists
            model_roi_points = None
            if self.roi_points:
                model_roi_points = self.dimension_handler.convert_roi_points(
                    self.roi_points,
                    source='source',
                    target='yolo'
                )
            
            # Perform detection with converted ROI
            detections, density_map = self.detector.detect(
                frame, 
                roi_points=model_roi_points,
                threshold=self.threshold
            )
            
            # Visualize results with original ROI points
            annotated_frame = self.detector.visualize(
                frame,
                detections,
                density_map,
                roi_points=self.roi_points
            )
            
            return detections, annotated_frame
            
        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            return None, None
            
    def get_performance_metrics(self):
        """Get current performance metrics"""
        return {
            'fps': {
                'target': self.target_fps,
                'actual': round(self.current_fps, 1)
            },
            'latency': self.current_latency
        }
        
    def update_performance_metrics(self, processing_time):
        """Update performance metrics"""
        self.frame_count += 1
        self.total_processing_time += processing_time
        
        # Update FPS calculation
        if self.frame_count >= 30:  # Calculate every 30 frames
            self.current_fps = self.frame_count / self.total_processing_time
            self.current_latency = self.total_processing_time / self.frame_count
            self.frame_count = 0
            self.total_processing_time = 0

    def get_static_frame(self):
        """Get a static frame with optimization support"""
        try:
            # Capture frame
            frame = self.capture_frame()
            if frame is None:
                return None
                
            # Check if resize is needed
            if 'resize' in request.args:
                max_dimension = int(request.args.get('resize', 1280))
                height, width = frame.shape[:2]
                if width > max_dimension or height > max_dimension:
                    # Calculate new dimensions while maintaining aspect ratio
                    ratio = min(max_dimension / width, max_dimension / height)
                    new_width = int(width * ratio)
                    new_height = int(height * ratio)
                    frame = cv2.resize(frame, (new_width, new_height), 
                                    interpolation=cv2.INTER_AREA)
            
            # Apply JPEG compression if quality parameter is provided
            if 'quality' in request.args:
                quality = int(request.args.get('quality', 85))
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
                _, buffer = cv2.imencode('.jpg', frame, encode_param)
                frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
            
            return frame
            
        except Exception as e:
            print(f"Error getting static frame: {str(e)}")
            return None 