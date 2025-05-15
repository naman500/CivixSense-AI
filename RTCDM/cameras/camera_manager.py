# Camera Manager for Real-Time Crowd Detection System
import json
import cv2
import os
import numpy as np
import time
import threading
from typing import Dict, Any, Optional, List, Tuple
from .stream_utils import StreamTester

class CameraManager:
    def __init__(self, config_file: str = 'cameras.json'):
        # Use the absolute path to the cameras.json file in the RTCDM root directory
        self.config_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cameras.json')
        print(f"CameraManager (from RTCDM/cameras/camera_manager.py) using config file: {self.config_file}")
        self.cameras: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()
        self.stream_tester = StreamTester()
        self.load_cameras()
    
    def load_cameras(self) -> None:
        """Load camera configurations from file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    self.cameras = json.load(f)
        except Exception as e:
            print(f"Error loading cameras: {str(e)}")
            self.cameras = {}
    
    def save_cameras(self) -> None:
        """Save camera configurations to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.cameras, f, indent=4)
        except Exception as e:
            print(f"Error saving cameras: {str(e)}")
    
    def add_camera(self, camera_info: Dict[str, Any]) -> bool:
        """
        Add a new camera to the system
        
        Parameters:
            camera_info: Dictionary containing camera configuration
            
        Returns:
            bool: True if camera was added successfully
        """
        try:
            # Get camera ID
            camera_id = camera_info.get('id')
            if not camera_id:
                print("Error: Camera ID is required")
                return False
            
            # Get camera URL
            camera_url = camera_info.get('url')
            if not camera_url:
                print("Error: Camera URL is required")
                return False
            
            # Get or set default values
            camera_info.setdefault('name', f"Camera {camera_id}")
            camera_info.setdefault('threshold', 50)
            camera_info.setdefault('density_threshold', 20)
            camera_info.setdefault('priority', 3)
            camera_info.setdefault('fps', 30)  # Default FPS
            
            # Convert FPS to integer
            try:
                camera_info['fps'] = int(camera_info['fps'])
            except (ValueError, TypeError):
                print(f"Warning: Invalid FPS value for camera {camera_id}, using default 30")
                camera_info['fps'] = 30
            
            # Add camera to configuration
            self.cameras[camera_id] = camera_info
            
            # Save configuration
            self.save_cameras()
            
            print(f"Added camera {camera_id} with URL: {camera_url}")
            print(f"Camera settings: FPS={camera_info['fps']}, Threshold={camera_info['threshold']}, Priority={camera_info['priority']}")
            
            return True
            
        except Exception as e:
            print(f"Error adding camera: {str(e)}")
            return False
    
    def remove_camera(self, camera_id: str) -> bool:
        """Remove a camera from the system"""
        with self.lock:
            if camera_id in self.cameras:
                del self.cameras[camera_id]
                self.save_cameras()
                return True
        return False
    
    def get_camera(self, camera_id: str) -> Optional[Dict[str, Any]]:
        """Get camera information"""
        return self.cameras.get(camera_id)
    
    def get_cameras(self) -> Dict[str, Dict[str, Any]]:
        """Get all camera configurations, ensuring all required fields are present"""
        result = {}
        for cam_id, cam in self.cameras.items():
            cam_copy = cam.copy()
            # Ensure all required fields are present
            cam_copy.setdefault('name', f"Camera {cam_id}")
            cam_copy.setdefault('threshold', 50)
            cam_copy.setdefault('density_threshold', 20)
            cam_copy.setdefault('priority', 3)
            cam_copy.setdefault('fps', 30)
            cam_copy.setdefault('roi_points', [])
            result[cam_id] = cam_copy
        return result
    
    def update_camera(self, camera_id: str, updates: Dict[str, Any]) -> bool:
        """Update camera configuration"""
        print(f"RTCDM/cameras/camera_manager.py: update_camera called for {camera_id} with {updates}")
        with self.lock:
            if camera_id in self.cameras:
                # Get current camera settings
                current_settings = self.cameras[camera_id].copy()
                
                # Update only the provided fields
                for key, value in updates.items():
                    if value is not None:  # Only update if value is provided
                        if key == 'priority':
                            # Ensure priority is an integer between 1 and 6
                            try:
                                priority_value = int(value)
                                if 1 <= priority_value <= 6:
                                    current_settings[key] = priority_value
                                else:
                                    print(f"Warning: Priority value {value} out of range (1-6), using default 3")
                                    current_settings[key] = 3
                            except (ValueError, TypeError):
                                print(f"Warning: Invalid priority value {value}, using default 3")
                                current_settings[key] = 3
                        else:
                            current_settings[key] = value
                
                # Update the camera settings
                self.cameras[camera_id] = current_settings
                
                # Save to configuration file
                self.save_cameras()
                print(f"RTCDM/cameras/camera_manager.py: Camera {camera_id} updated: {current_settings}")
                return True
        return False
    
    def test_camera(self, camera_id: str) -> Tuple[bool, Dict[str, Any]]:
        """Test camera connection"""
        camera = self.get_camera(camera_id)
        if not camera:
            return False, {'error': 'Camera not found'}
        
        return self.stream_tester.test_stream(camera['url'])
    
    def test_all_cameras(self) -> Dict[str, Tuple[bool, Dict[str, Any]]]:
        """Test all camera connections"""
        results = {}
        for camera_id, camera in self.cameras.items():
            results[camera_id] = self.stream_tester.test_stream(camera['url'])
        return results
    
    def get_best_stream_url(self, camera_id: str) -> Optional[str]:
        """Get the best performing stream URL for a camera"""
        camera = self.get_camera(camera_id)
        if not camera:
            return None
        
        # If camera has multiple URLs, test them all
        if isinstance(camera['url'], list):
            return self.stream_tester.get_best_url(camera['url'])
        
        # If single URL, test it
        success, _ = self.stream_tester.test_stream(camera['url'])
        return camera['url'] if success else None
    
    def reset_cameras(self) -> None:
        """Reset all camera configurations"""
        with self.lock:
            self.cameras = {}
            self.save_cameras()

    def test_all_camera_formats(self, base_url):
        """
        Test multiple URL formats for a camera to find which works
        
        Parameters:
            base_url (str): Base URL of the camera (http://ip:port)
            
        Returns:
            dict: Results of testing each URL format
        """
        # Handle numeric camera indices for local webcams
        if base_url.isdigit():
            camera_index = int(base_url)
            print(f"Testing local webcam at index {camera_index}")
            try:
                cap = cv2.VideoCapture(camera_index)
                time.sleep(1)
                opened = cap.isOpened()
                ret = False
                
                if opened:
                    ret, _ = cap.read()
                    
                cap.release()
                
                results = {
                    f"Local Webcam (index {camera_index})": {
                        "opened": opened,
                        "frame_read": ret
                    }
                }
                
                return {
                    "results": results,
                    "best_url": str(camera_index) if (opened or ret) else None
                }
            except Exception as e:
                print(f"Error testing local webcam {camera_index}: {str(e)}")
                return {
                    "results": {
                        f"Local Webcam (index {camera_index})": {
                            "opened": False,
                            "frame_read": False,
                            "error": str(e)
                        }
                    },
                    "best_url": None
                }
        
        # For network cameras, test multiple formats
        formats = [
            base_url,
            f"{base_url}/video",
            f"{base_url}/videostream.cgi",
            f"{base_url}/shot.jpg",
            f"rtsp://{base_url.split('://')[-1]}/h264_ulaw.sdp"  # Common RTSP format
        ]
        
        results = {}
        
        for url_format in formats:
            try:
                print(f"Testing camera URL: {url_format}")
                cap = cv2.VideoCapture(url_format)
                time.sleep(1)
                opened = cap.isOpened()
                ret = False
                
                if opened:
                    ret, _ = cap.read()
                    
                cap.release()
                
                results[url_format] = {
                    "opened": opened,
                    "frame_read": ret
                }
                
                print(f"URL {url_format}: Opened={opened}, Frame read={ret}")
                
            except Exception as e:
                print(f"Error testing URL {url_format}: {str(e)}")
                results[url_format] = {
                    "opened": False,
                    "frame_read": False,
                    "error": str(e)
                }
        
        # Find the best URL (one that can read frames)
        best_url = None
        for url, result in results.items():
            if result.get("frame_read"):
                best_url = url
                break
                
        # If no URL can read frames, try one that at least opened
        if not best_url:
            for url, result in results.items():
                if result.get("opened"):
                    best_url = url
                    break
        
        return {
            "results": results,
            "best_url": best_url
        }

    def get_total_crowd_count(self):
        """Get the total crowd count across all active cameras"""
        total = 0
        for camera in self.cameras.values():
            if camera.get('active', False) and 'crowd_count' in camera:
                total += camera['crowd_count']
        return total

    def camera_exists(self, camera_id):
        """
        Check if a camera exists in the configuration
        
        Parameters:
            camera_id (str): ID of the camera to check
            
        Returns:
            bool: True if camera exists, False otherwise
        """
        with self.lock:  # Use lock for thread safety
            return camera_id in self.cameras

    def set_roi(self, camera_id, roi_points):
        """
        Set Region of Interest for a camera
        Args:
            camera_id: ID of the camera
            roi_points: List of [x, y] coordinates defining the ROI polygon
        """
        try:
            if not self.camera_exists(camera_id):
                raise ValueError(f"Camera {camera_id} does not exist")
            if not isinstance(roi_points, list) or len(roi_points) < 3:
                raise ValueError("ROI points must be a list of at least 3 points")
            # Validate points format
            for point in roi_points:
                if not isinstance(point, list) or len(point) != 2:
                    raise ValueError("Each ROI point must be a list of [x, y] coordinates")
            # Store ROI points in original frame coordinates
            self.cameras[camera_id]['roi_points'] = roi_points
            # Update configuration file
            self.save_cameras()
            print(f"ROI set successfully for camera {camera_id}: {roi_points}")
            return True
        except Exception as e:
            print(f"Error setting ROI: {str(e)}")
            return False

    def get_roi(self, camera_id):
        """
        Get the ROI points for a camera
        Args:
            camera_id: ID of the camera
        Returns:
            List of [x, y] coordinates or None if no ROI set
        """
        try:
            if not self.camera_exists(camera_id):
                raise ValueError(f"Camera {camera_id} does not exist")
            return self.cameras[camera_id].get('roi_points', None)
        except Exception as e:
            print(f"Error getting ROI: {str(e)}")
            return None

    def clear_roi(self, camera_id):
        """
        Clear the ROI for a camera
        Args:
            camera_id: ID of the camera
        """
        try:
            if not self.camera_exists(camera_id):
                raise ValueError(f"Camera {camera_id} does not exist")
            if 'roi_points' in self.cameras[camera_id]:
                del self.cameras[camera_id]['roi_points']
            self.save_cameras()
            print(f"ROI cleared for camera {camera_id}")
            return True
        except Exception as e:
            print(f"Error clearing ROI: {str(e)}")
            return False

    def set_threshold(self, camera_id, threshold=None, density_threshold=None, priority=None):
        """
        Set threshold, density_threshold, and priority for a camera.
        Args:
            camera_id: ID of the camera
            threshold: (optional) new threshold value
            density_threshold: (optional) new density threshold value
            priority: (optional) new priority value
        Returns:
            bool: True if updated, False otherwise
        """
        if not self.camera_exists(camera_id):
            print(f"Camera {camera_id} does not exist")
            return False
        updated = False
        if threshold is not None:
            self.cameras[camera_id]['threshold'] = threshold
            updated = True
        if density_threshold is not None:
            self.cameras[camera_id]['density_threshold'] = density_threshold
            updated = True
        if priority is not None:
            self.cameras[camera_id]['priority'] = priority
            updated = True
        if updated:
            self.save_cameras()
            print(f"Updated thresholds for camera {camera_id}: threshold={threshold}, density_threshold={density_threshold}, priority={priority}")
        return updated
