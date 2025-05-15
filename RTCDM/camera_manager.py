# Camera Manager for Real-Time Crowd Detection System
import json
import cv2
import os
import numpy as np
import time
import threading

class CameraManager:
    def __init__(self):
        # Initialize camera management
        self.cameras = {}
        self.camera_threads = {}  # Initialize camera_threads dictionary
        self.config_file = os.path.join(os.path.dirname(__file__), 'cameras.json')
        print(f"CameraManager (from RTCDM/camera_manager.py) using config file: {self.config_file}")
        self.load_cameras()
        self.lock = threading.Lock()  # Add thread lock for thread-safe operations
    
    def load_cameras(self):
        """Load cameras from configuration file"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    self.cameras = json.load(f)
            except json.JSONDecodeError:
                self.cameras = {}
    
    def save_cameras(self):
        """Save cameras to configuration file"""
        with open(self.config_file, 'w') as f:
            json.dump(self.cameras, f, indent=4)

    def add_camera(self, camera_info):
        """
        Add a new camera to the system
        
        Parameters:
            camera_info (dict): Dictionary containing camera information
                - url: URL or IP address of the camera
                - name: Display name for the camera
                - id: Unique identifier for the camera
                - threshold: Crowd count threshold for alerts
                - density_threshold: Threshold to switch to hybrid mode
                - priority: Priority level (1-6, with 6 being highest priority)
        
        Returns:
            bool: True if camera was added successfully, False otherwise
        """
        if 'id' not in camera_info or 'url' not in camera_info:
            return False
        
        camera_id = camera_info['id']
        camera_url = camera_info['url']
        
        # Check if a camera with this ID already exists
        if camera_id in self.cameras:
            print(f"Camera with ID {camera_id} already exists, updating configuration.")
        
        # Add camera to the dictionary without strict validation
        # This allows adding cameras even when connection might be temporarily problematic
        self.cameras[camera_id] = {
            'url': camera_url,
            'name': camera_info.get('name', f"Camera {camera_id}"),
            'roi': camera_info.get('roi', None),
            'threshold': camera_info.get('threshold', 50),  # Default alert threshold
            'density_threshold': camera_info.get('density_threshold', 20),  # Default density threshold
            'priority': camera_info.get('priority', 3)  # Default priority (1-6, with 6 being highest)
        }
        
        # Try to verify the camera, but don't prevent adding it if verification fails
        verified = self._verify_camera_connection(camera_id, camera_url)
        
        # Update verified status
        self.cameras[camera_id]['verified'] = verified
        
        print(f"Camera {camera_id} added to configuration (Connection verified: {verified}, Priority: {self.cameras[camera_id]['priority']})")
        self.save_cameras()
        return True
        
    def _verify_camera_connection(self, camera_id, camera_url):
        """
        Try to verify camera connection
        
        Parameters:
            camera_id (str): ID of the camera
            camera_url (str): URL of the camera
            
        Returns:
            bool: True if connection verified, False otherwise
        """
        try:
            print(f"Trying to verify camera {camera_id} at URL: {camera_url}")
            
            # Handle numeric camera indices (0, 1, etc. for local webcams)
            if camera_url.isdigit():
                camera_index = int(camera_url)
                print(f"Converting string '{camera_url}' to integer index for local webcam")
                
                # Try to connect to the specified camera
                cap = cv2.VideoCapture(camera_index)
                if not cap.isOpened():
                    print(f"Camera {camera_id}: Failed to open webcam at index {camera_index}")
                    cap.release()
                    return False
                
                # Try to read a frame with a longer timeout
                for _ in range(5):  # Try 5 times
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        print(f"Camera {camera_id}: Successfully verified webcam connection")
                        cap.release()
                        return True
                    time.sleep(1)  # Wait a second between attempts
                
                print(f"Camera {camera_id}: Could not read frame from webcam")
                cap.release()
                return False
            
            # For network cameras, use the existing code
            cap = cv2.VideoCapture(camera_url)
            
            if not cap.isOpened():
                print(f"Camera {camera_id}: Failed to open capture")
                cap.release()
                return False
                
            # Try to read just one frame with timeout
            for _ in range(3):
                ret, frame = cap.read()
                if ret:
                    cap.release()
                    print(f"Camera {camera_id}: Successfully verified connection")
                    return True
                time.sleep(1)
            
            print(f"Camera {camera_id}: Opened but couldn't read frame")
            cap.release()
            
            if 'shot.jpg' in camera_url:
                print(f"Camera {camera_id}: MJPEG stream, assuming valid")
                return True
                
            return False
            
        except Exception as e:
            print(f"Warning: Camera {camera_id} verification failed: {str(e)}")
            return False
            
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
            f"rtsp://{base_url.split('://')[-1]}/h264_ulaw.sdp",  # Common RTSP format
            f"{base_url}/stream1"  # Add /stream1 for Tapo and similar cameras
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

    def remove_camera(self, camera_id):
        """
        Remove a camera from the system
        
        Parameters:
            camera_id (str): ID of the camera to remove
            
        Returns:
            bool: True if camera was removed successfully, False otherwise
        """
        with self.lock:  # Use lock for thread safety
            if camera_id not in self.cameras:
                print(f"Camera {camera_id} not found")
                return False
            
            # Remove from our configuration
            del self.cameras[camera_id]
            
            # Save configuration
            self.save_cameras()
            
            print(f"Camera {camera_id} removed from configuration")
            return True

    def get_cameras(self):
        """
        Get all cameras
        
        Returns:
            dict: Dictionary of all cameras
        """
        return self.cameras
    
    def get_camera(self, camera_id):
        """
        Get a specific camera
        
        Parameters:
            camera_id (str): ID of the camera
            
        Returns:
            dict: Camera information or None if not found
        """
        return self.cameras.get(camera_id)
    
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
            # Notify active streams to update ROI (if active_streams exists)
            if hasattr(self, 'active_streams') and camera_id in getattr(self, 'active_streams', {}):
                self.active_streams[camera_id].update_roi(roi_points)
            print(f"ROI set successfully for camera {camera_id}")
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
            # Update configuration file
            self.save_cameras()
            # Notify active streams
            if hasattr(self, 'active_streams') and camera_id in getattr(self, 'active_streams', {}):
                self.active_streams[camera_id].update_roi(None)
            print(f"ROI cleared for camera {camera_id}")
            return True
        except Exception as e:
            print(f"Error clearing ROI: {str(e)}")
            return False

    def set_threshold(self, camera_id, threshold=None, density_threshold=None, priority=None):
        """
        Set crowd threshold, density threshold, and/or priority for a camera
        Parameters:
            camera_id (str): ID of the camera
            threshold (int): Crowd count threshold
            density_threshold (int): Density threshold
            priority (int): Priority level (1-6)
        Returns:
            bool: True if updated, False if camera not found
        """
        if camera_id in self.cameras:
            if threshold is not None:
                self.cameras[camera_id]['threshold'] = threshold
            if density_threshold is not None:
                self.cameras[camera_id]['density_threshold'] = density_threshold
            if priority is not None:
                self.cameras[camera_id]['priority'] = priority
            self.save_cameras()
            return True
        return False

    def reset_cameras(self):
        """
        Reset all camera configurations and stop all camera threads
        
        Returns:
            bool: True if successful
        """
        with self.lock:  # Use lock for thread-safe operations
            # Stop and clean up all camera threads
            for camera_id in list(self.camera_threads.keys()):
                if camera_id in self.camera_threads:
                    thread = self.camera_threads[camera_id]
                    if hasattr(thread, "stop_processing"):
                        thread.stop_processing = True
                    thread.join(timeout=5)  # Wait up to 5 seconds for thread to stop
                    del self.camera_threads[camera_id]
            
            # Clear the cameras dictionary
            self.cameras = {}
            
            # Save the empty configuration to file
            self.save_cameras()
            
            print("All camera configurations have been reset and threads stopped")
            return True

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

    def update_camera(self, camera_id, updates):
        """
        Update camera settings (threshold, density_threshold, priority, etc.)
        Parameters:
            camera_id (str): ID of the camera
            updates (dict): Dictionary of fields to update
        Returns:
            bool: True if updated, False if camera not found
        """
        print(f"RTCDM/camera_manager.py: update_camera called for {camera_id} with {updates}")
        if camera_id not in self.cameras:
            return False
        for key, value in updates.items():
            self.cameras[camera_id][key] = value
        self.save_cameras()
        print(f"RTCDM/camera_manager.py: Camera {camera_id} updated: {self.cameras[camera_id]}")
        return True