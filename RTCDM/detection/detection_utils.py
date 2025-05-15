# Utility functions for crowd detection

import cv2
import numpy as np
import time

def draw_roi(frame, roi_points, color=(0, 255, 0), thickness=2):
    """
    Draw ROI polygon on the frame with enhanced visualization
    
    Parameters:
        frame: Input frame
        roi_points: List of [x,y] coordinates
        color: BGR color tuple
        thickness: Line thickness
        
    Returns:
        frame: Frame with ROI drawn
        area_percent: Percentage of frame area covered by ROI
    """
    if not roi_points or len(roi_points) < 3:
        return frame, 0.0
        
    try:
        # Convert points to numpy array
        points = np.array(roi_points, dtype=np.int32)
        
        # Calculate ROI area as percentage of frame
        frame_area = frame.shape[0] * frame.shape[1]
        roi_area = cv2.contourArea(points)
        area_percent = (roi_area / frame_area) * 100
        
        # Create a copy of the frame
        frame_copy = frame.copy()
        
        # Draw semi-transparent overlay
        overlay = frame_copy.copy()
        cv2.fillPoly(overlay, [points], color)
        alpha = 0.2  # Transparency factor
        cv2.addWeighted(overlay, alpha, frame_copy, 1 - alpha, 0, frame_copy)
        
        # Draw polygon outline
        cv2.polylines(frame_copy, [points], True, color, thickness)
        
        # Draw points with numbers
        for i, point in enumerate(points):
            # Draw point marker
            cv2.circle(frame_copy, tuple(point), 8, (255, 255, 255), -1)
            cv2.circle(frame_copy, tuple(point), 8, color, 2)
            
            # Draw point number
            cv2.putText(frame_copy, str(i + 1), 
                       (point[0] - 4, point[1] + 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (0, 0, 0), 2)
        
        return frame_copy, area_percent
        
    except Exception as e:
        print(f"Error drawing ROI: {str(e)}")
        return frame, 0.0

def draw_crowd_info(frame, count, mode=None, roi_points=None, threshold=None):
    """
    Draw crowd information on the frame with enhanced visualization
    
    Parameters:
        frame: Input frame
        count: Number of people detected
        mode: Current detection mode (YOLOv8n or YOLOv8m)
        roi_points: ROI points for visualization
        threshold: Threshold value for alerts
        
    Returns:
        frame: Frame with information drawn
    """
    # Draw ROI if provided
    if roi_points:
        frame, area_percent = draw_roi(frame, roi_points)
    
    # Create background rectangle for text
    cv2.rectangle(frame, (10, 10), (300, 100), (0, 0, 0), -1)
    
    # Draw count with color based on threshold
    color = (0, 255, 0)  # Green by default
    if threshold and count > threshold:
        color = (0, 0, 255)  # Red if over threshold
    
    cv2.putText(frame, f'Count: {count}', (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    # Draw mode if provided
    if mode:
        cv2.putText(frame, f'Mode: {mode}', (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Draw threshold if provided
    if threshold:
        cv2.putText(frame, f'Threshold: {threshold}', (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    return frame

def combine_detections(detections1, detections2):
    """
    Combine two lists of detections while removing duplicates based on IoU
    
    Parameters:
        detections1: First list of detections [x,y,w,h]
        detections2: Second list of detections [x,y,w,h]
        
    Returns:
        combined_detections: List of unique detections
    """
    if not detections1:
        return detections2
    if not detections2:
        return detections1
        
    combined = detections1.copy()
    
    for det2 in detections2:
        is_duplicate = False
        for det1 in detections1:
            # Calculate IoU
            x1 = max(det1[0], det2[0])
            y1 = max(det1[1], det2[1])
            x2 = min(det1[0] + det1[2], det2[0] + det2[2])
            y2 = min(det1[1] + det1[3], det2[1] + det2[3])
            
            if x2 < x1 or y2 < y1:
                continue
                
            intersection = (x2 - x1) * (y2 - y1)
            area1 = det1[2] * det1[3]
            area2 = det2[2] * det2[3]
            union = area1 + area2 - intersection
            
            iou = intersection / union if union > 0 else 0
            
            if iou > 0.5:  # If IoU > 0.5, consider it a duplicate
                is_duplicate = True
                break
                
        if not is_duplicate:
            combined.append(det2)
            
    return combined

def calculate_performance_metrics(start_time, frame_count):
    """
    Calculate performance metrics with enhanced statistics
    
    Parameters:
        start_time: Start time of processing
        frame_count: Number of frames processed
        
    Returns:
        fps: Frames per second
        processing_time: Average processing time per frame
        total_time: Total processing time
    """
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time if elapsed_time > 0 else 0
    processing_time = elapsed_time / frame_count if frame_count > 0 else 0
    return fps, processing_time, elapsed_time

class FrameDimensionHandler:
    def __init__(self):
        # Model dimensions
        self.yolo_input_size = (640, 640)  # Standard YOLO input size
        
        # Source dimensions (will be set automatically)
        self.source_width = None
        self.source_height = None
        
        # Conversion matrices
        self.source_to_yolo = None
        self.yolo_to_source = None
        
        # Performance tracking
        self.last_frame_time = None
        self.frame_times = []

    def set_source_dimensions(self, frame):
        """
        Automatically detect and set source frame dimensions
        
        Args:
            frame: Source video frame
        """
        if frame is None:
            raise ValueError("Invalid frame provided")
        
        height, width = frame.shape[:2]
        print(f"[DEBUG] FrameDimensionHandler source_width: {width}, source_height: {height}")
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid frame dimensions: {width}x{height}")
        
        self.source_width = width
        self.source_height = height
        
        # Update conversion matrices
        self._update_conversion_matrices()
        
        return width, height

    def _update_conversion_matrices(self):
        """Update conversion matrices for coordinate transformations"""
        if not self.source_width or not self.source_height:
            return
            
        # For 16:9 (1280x720) to square (640x640) conversion:
        # 1. Scale the width to 640
        # 2. Calculate corresponding height (360)
        # 3. Add padding to make it 640x640
        
        # First scale width to target
        scale = self.yolo_input_size[0] / self.source_width
        scaled_height = int(self.source_height * scale)
        
        # Calculate padding needed to make it square
        pad_y = (self.yolo_input_size[1] - scaled_height) // 2
        
        # Store conversion factors and padding
        self.source_to_yolo = scale
        self.yolo_padding = pad_y
        self.scaled_height = scaled_height
        
        # Store inverse scale for converting back
        self.yolo_to_source = 1/scale

    def convert_roi_points(self, points, source='source', target='yolo'):
        """
        Convert ROI points between different coordinate spaces
        
        Args:
            points: List of [x, y] coordinates
            source: Source coordinate space ('source', 'yolo')
            target: Target coordinate space ('source', 'yolo')
            
        Returns:
            Converted points in the target coordinate space
        """
        if not points:
            return points
            
        if not self.source_width or not self.source_height:
            raise ValueError("Source dimensions not set. Call set_source_dimensions first.")
            
        # Get conversion factors
        if source == target:
            return points
            
        # Convert points
        converted_points = []
        for point in points:
            if isinstance(point, dict):
                x, y = point['x'], point['y']
            else:
                x, y = point[0], point[1]
                
            if target == 'yolo':
                # Convert to YOLO space:
                # 1. Scale coordinates
                # 2. Add padding to y-coordinate
                new_x = int(x * self.source_to_yolo)
                new_y = int(y * self.source_to_yolo) + self.yolo_padding
            else:
                # Convert back to source space:
                # 1. Remove padding from y-coordinate
                # 2. Scale back to source dimensions
                new_x = int(x * self.yolo_to_source)
                new_y = int((y - self.yolo_padding) * self.yolo_to_source)
            
            # Return in the same format as input
            if isinstance(point, dict):
                converted_points.append({'x': new_x, 'y': new_y})
            else:
                converted_points.append([new_x, new_y])
                
        return converted_points

    def resize_frame(self, frame, target='yolo'):
        """
        Resize frame to target dimensions while maintaining aspect ratio with padding
        
        Args:
            frame: Input frame
            target: Target space ('yolo', 'source')
            
        Returns:
            Resized frame with padding if needed
        """
        if target == 'yolo':
            # Scale width to target size
            scale = self.yolo_input_size[0] / self.source_width
            new_width = self.yolo_input_size[0]
            new_height = int(self.source_height * scale)
            
            # Resize frame
            resized = cv2.resize(frame, (new_width, new_height))
            
            # Add padding to make it square
            pad_y = (self.yolo_input_size[1] - new_height) // 2
            if pad_y > 0:
                padded = cv2.copyMakeBorder(
                    resized,
                    pad_y, pad_y,  # Top, bottom
                    0, 0,  # Left, right
                    cv2.BORDER_CONSTANT,
                    value=(0, 0, 0)  # Black padding
                )
                return padded
            
            return resized
        else:
            raise ValueError(f"Unsupported target space: {target}")

    def get_average_processing_time(self):
        """Get average processing time for the last 100 frames"""
        if not self.frame_times:
            return 0
        return sum(self.frame_times) / len(self.frame_times)
