# YOLO Model for Crowd Detection
import cv2
import numpy as np
import os
from ultralytics import YOLO

class YOLOModel:
    def __init__(self, model_path='models/yolov8n.onnx'):
        """Initialize YOLOv8 model in ONNX format"""
        try:
            self.model = YOLO(model_path)  # Load ONNX model
            self.model.conf = 0.6  # confidence threshold
            self.model_loaded = True
            print(f"Successfully loaded YOLO model from {model_path}")
        except Exception as e:
            print(f"Error loading YOLO model: {str(e)}")
            self.model_loaded = False
    
    def is_model_loaded(self):
        """Check if the model is loaded and ready to use"""
        return self.model_loaded
    
    def detect(self, frame, roi=None):
        """
        Detect people in a frame using YOLOv8 ONNX model with improved ROI handling
        
        Parameters:
            frame: Input image
            roi: Region of interest polygon coordinates (can be [x,y] lists or {x,y} dicts)
            
        Returns:
            count: Number of people detected
            boxes: Bounding boxes of detected people
        """
        if not self.model_loaded:
            return 0, []
        
        try:
            # Create a mask for ROI if provided
            mask = None
            if roi and len(roi) > 2:
                # Convert ROI points to standard format
                formatted_points = []
                for point in roi:
                    try:
                        if isinstance(point, dict):
                            x = int(point.get('x', 0))
                            y = int(point.get('y', 0))
                        elif isinstance(point, (list, tuple)) and len(point) >= 2:
                            x = int(point[0])
                            y = int(point[1])
                        else:
                            print(f"Warning: Invalid ROI point format: {point}")
                            continue
                        formatted_points.append([x, y])
                    except (ValueError, TypeError, IndexError) as e:
                        print(f"Warning: Invalid ROI point: {str(e)}")
                        continue
                
                if len(formatted_points) >= 3:
                    # Create mask in model input space
                    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                    roi_points = np.array(formatted_points, dtype=np.int32)
                    cv2.fillPoly(mask, [roi_points], 255)
                    
                    # Apply mask to frame
                    frame = cv2.bitwise_and(frame, frame, mask=mask)
            
            # Run YOLOv8 inference
            results = self.model(frame)[0]
            
            # Filter detections by ROI
            person_boxes = []
            for box in results.boxes:
                if box.cls == 0:  # person class
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    
                    # Convert coordinates to integers
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    
                    # If ROI is defined, check if the detection center is within ROI
                    if mask is not None:
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        if mask[center_y, center_x] == 0:
                            continue
                    
                    person_boxes.append([x1, y1, x2 - x1, y2 - y1])
            
            return len(person_boxes), person_boxes
            
        except Exception as e:
            print(f"Error in YOLO detection: {str(e)}")
            return 0, []
    
    def draw_detections(self, frame, boxes):
        """
        Draw bounding boxes on the frame
        
        Parameters:
            frame: Input frame
            boxes: List of bounding boxes [x,y,w,h]
            
        Returns:
            frame: Frame with bounding boxes drawn
        """
        for box in boxes:
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, 'Person', (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame
