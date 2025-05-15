# Dashboard for Monitoring Crowd Detection
import os
import json
import threading
import time
from datetime import datetime
from flask import Flask, render_template, jsonify, request, Response
import cv2
import numpy as np
import psutil
import torch
try:
    import GPUtil
except ImportError:
    GPUtil = None

class Dashboard:
    def __init__(self, camera_manager=None, alert_manager=None):
        # Initialize dashboard
        self.dashboard_dir = os.path.dirname(os.path.abspath(__file__))
        
        # For compatibility, set up Flask to look in either:
        # 1. The current directory (dashboard_dir) 
        # 2. A templates subdirectory if it exists
        template_dir = self.dashboard_dir
        templates_subdir = os.path.join(self.dashboard_dir, 'templates')
        if os.path.exists(templates_subdir) and os.path.isdir(templates_subdir):
            template_dir = templates_subdir
        
        # Initialize the Flask app - note: this app is NOT used when run through main.py
        # It's only used for standalone testing
        self.app = Flask("rtcdm_dashboard", 
                         static_folder=os.path.join(self.dashboard_dir, 'static'),
                         template_folder=template_dir)
        
        # Register routes on the local app instance (for standalone testing)
        self._register_routes()
        
        # Store references to managers
        self.camera_manager = camera_manager
        self.alert_manager = alert_manager
        
        # Dashboard data
        self.crowd_data = {}  # camera_id -> count, timestamp, etc.
        self.performance_data = {"latency": [], "fps": []}
        self.streaming_frames = {}  # camera_id -> latest frame
        
        # Lock for thread safety
        self.data_lock = threading.Lock()
    
    def _register_routes(self, external_app=None):
        """Register Flask routes for the dashboard"""
        # Determine which app to use
        app_to_use = external_app if external_app is not None else self.app
        
        # Register routes on the appropriate Flask app
        app_to_use.route("/")(self.index)
        app_to_use.route("/api/cameras")(self.get_cameras)
        app_to_use.route("/api/add_camera", methods=["POST"])(self.add_camera)
        app_to_use.route("/api/remove_camera", methods=["POST"])(self.remove_camera_api)
        app_to_use.route("/api/set_roi", methods=["POST"])(self.set_roi)
        app_to_use.route("/api/set_threshold", methods=["POST"])(self.set_threshold)
        app_to_use.route("/api/crowd_data")(self.get_crowd_data)
        app_to_use.route("/api/alerts")(self.get_alerts)
        app_to_use.route("/api/clear_alerts", methods=["POST"])(self.clear_alerts)
        app_to_use.route("/api/performance")(self.get_performance)
        app_to_use.route("/video_feed/<camera_id>")(self.video_feed)
        app_to_use.route("/api/test_camera", methods=["POST"])(self.test_camera_connection)
        app_to_use.route("/api/reset_cameras", methods=["POST"])(self.reset_cameras)
        # Add check_feed route
        app_to_use.route("/api/check_feed", methods=["POST"])(self.check_feed)
        app_to_use.route("/api/resource_usage")(self.get_resource_usage)
    
    def index(self):
        """Serve the main dashboard page"""
        try:
            # First try with Flask app's render_template
            from flask import render_template
            return render_template("dashboard_template.html")
        except Exception as e:
            # Fallback: read template file directly
            dashboard_dir = os.path.dirname(os.path.abspath(__file__))
            template_paths = [
                os.path.join(dashboard_dir, "dashboard_template.html"),  # Try direct path
                os.path.join(dashboard_dir, "templates", "dashboard_template.html")  # Try templates folder
            ]
            
            for template_path in template_paths:
                if os.path.exists(template_path):
                    try:
                        with open(template_path, 'r', encoding='utf-8') as file:
                            return file.read()
                    except Exception as file_error:
                        print(f"Error reading template file: {str(file_error)}")
            
            # If all attempts fail, return a basic HTML response
            return """
            <html>
            <head><title>RTCDM Dashboard</title></head>
            <body>
                <h1>Dashboard Error</h1>
                <p>Could not load dashboard template.</p>
                <p>Error: {}</p>
            </body>
            </html>
            """.format(str(e))
    
    def get_cameras(self):
        """API endpoint to get all cameras"""
        if self.camera_manager:
            return jsonify(self.camera_manager.get_cameras())
        return jsonify({})
    
    def add_camera(self):
        """API endpoint to add a camera"""
        if not self.camera_manager:
            return jsonify({"success": False, "message": "Camera manager not available"}), 500
        
        data = request.json
        if not data or "url" not in data or "id" not in data:
            return jsonify({"success": False, "message": "Missing required parameters"}), 400

        # Handle numeric camera indices (local webcams)
        original_url = data["url"]
        if original_url.isdigit():
            try:
                # For local webcams, just use the index directly
                test_data = data.copy()
                success = self.camera_manager.add_camera(test_data)
                if success:
                    print(f"Successfully added local webcam with index: {original_url}")
                    # Notify the main application to start a camera thread for this new camera
                    if hasattr(self, 'on_camera_added') and callable(self.on_camera_added):
                        camera_info = self.camera_manager.get_camera(data["id"])
                        self.on_camera_added(data["id"], camera_info)
                    return jsonify({"success": True, "url_used": original_url})
                else:
                    return jsonify({"success": False, "message": "Failed to connect to local webcam"}), 400
            except Exception as e:
                return jsonify({"success": False, "message": f"Error connecting to local webcam: {str(e)}"}), 500
                
        # For network cameras, try different URL formats
        urls_to_try = [original_url]
        
        # Only generate alternatives if it's an IP address without path
        if '://' in original_url and original_url.count('/') == 2:
            base_url = original_url.rstrip('/')
            urls_to_try = [
                base_url,
                f"{base_url}/video",
                f"{base_url}/shot.jpg",
                f"{base_url}/videostream.cgi"
            ]
            print(f"Will try these URLs: {urls_to_try}")
        
        # Try each URL until one works
        last_error = None
        for url in urls_to_try:
            try:
                test_data = data.copy()
                test_data["url"] = url
                
                # Add default density threshold if not provided
                if "density_threshold" not in test_data:
                    test_data["density_threshold"] = 20
                
                success = self.camera_manager.add_camera(test_data)
                if success:
                    print(f"Successfully added camera with URL: {url}")
                    # Notify the main application to start a camera thread for this new camera
                    if hasattr(self, 'on_camera_added') and callable(self.on_camera_added):
                        camera_info = self.camera_manager.get_camera(data["id"])
                        self.on_camera_added(data["id"], camera_info)
                    return jsonify({"success": True, "url_used": url})
            except Exception as e:
                last_error = str(e)
                print(f"Failed to connect to {url}: {str(e)}")
                continue
        
        # If all URLs failed, return with the original URL (which was already tried)
        error_message = f"Failed to connect to camera with any URL format. Last error: {last_error}" if last_error else "Failed to connect to camera with any URL format"
        return jsonify({
            "success": False, 
            "message": error_message, 
            "urls_tried": urls_to_try
        }), 400
    
    def remove_camera(self, camera_id):
        """Remove a camera from the dashboard"""
        try:
            # Remove camera from camera manager
            if self.camera_manager:
                self.camera_manager.remove_camera(camera_id)
            
            # Clean up dashboard data
            with self.data_lock:
                # Remove streaming frame
                if camera_id in self.streaming_frames:
                    del self.streaming_frames[camera_id]
                
                # Remove crowd data
                if camera_id in self.crowd_data:
                    del self.crowd_data[camera_id]
                
                # Clean up performance data related to this camera
                if hasattr(self, 'performance_data'):
                    if 'latency' in self.performance_data:
                        self.performance_data['latency'] = [lat for lat in self.performance_data['latency'] 
                                                          if lat.get('camera_id') != camera_id]
                    if 'fps' in self.performance_data:
                        self.performance_data['fps'] = [fps for fps in self.performance_data['fps'] 
                                                      if fps.get('camera_id') != camera_id]
                
                # Clean up analytics data
                if hasattr(self, 'analytics_data'):
                    if 'crowd_trends' in self.analytics_data:
                        self.analytics_data['crowd_trends'] = [trend for trend in self.analytics_data['crowd_trends'] 
                                                             if trend.get('camera_id') != camera_id]
                    if 'performance_metrics' in self.analytics_data:
                        self.analytics_data['performance_metrics'] = [metric for metric in self.analytics_data['performance_metrics'] 
                                                                    if metric.get('camera_id') != camera_id]
            
            # Notify alert manager to remove camera alerts
            if self.alert_manager:
                self.alert_manager.remove_camera_alerts(camera_id)
            
            # Call the callback if set
            if hasattr(self, 'on_camera_removed') and self.on_camera_removed:
                self.on_camera_removed(camera_id)
            
            return True, "Camera removed successfully"
            
        except Exception as e:
            print(f"Error removing camera: {str(e)}")
            return False, f"Failed to remove camera: {str(e)}"
    
    def set_roi(self):
        """Set ROI for a camera with improved error handling and validation"""
        try:
            data = request.get_json()
            if not data:
                return jsonify({"success": False, "message": "No data provided"}), 400
            
            camera_id = data.get('id')
            roi_points = data.get('roi')
            
            if not camera_id:
                return jsonify({"success": False, "message": "Camera ID is required"}), 400
            
            if not roi_points or not isinstance(roi_points, list):
                return jsonify({"success": False, "message": "Valid ROI points are required"}), 400
            
            if len(roi_points) < 3:
                return jsonify({"success": False, "message": "At least 3 points are required to define ROI"}), 400
            
            # Get camera manager instance
            if not self.camera_manager:
                return jsonify({"success": False, "message": "Camera manager not initialized"}), 500
            
            # Get camera info to validate ROI points against frame dimensions
            camera_info = self.camera_manager.get_camera(camera_id)
            if not camera_info:
                return jsonify({"success": False, "message": "Camera not found"}), 404
            
            # Convert points to standard format [x,y]
            formatted_points = []
            for point in roi_points:
                try:
                    if isinstance(point, dict):
                        x = int(point.get('x', 0))
                        y = int(point.get('y', 0))
                    elif isinstance(point, list) and len(point) >= 2:
                        x = int(point[0])
                        y = int(point[1])
                    else:
                        return jsonify({"success": False, "message": "Invalid ROI point format"}), 400
                    formatted_points.append([x, y])
                except (ValueError, TypeError) as e:
                    return jsonify({"success": False, "message": f"Invalid point coordinates: {str(e)}"}), 400
            
            # Validate ROI points are within frame bounds
            frame_width = camera_info.get('width', 640)  # Default to common webcam resolution
            frame_height = camera_info.get('height', 480)
            
            for point in formatted_points:
                if point[0] < 0 or point[0] >= frame_width or point[1] < 0 or point[1] >= frame_height:
                    return jsonify({
                        "success": False, 
                        "message": f"ROI points must be within frame bounds (0-{frame_width-1}, 0-{frame_height-1})"
                    }), 400
            
            # Set ROI with formatted points
            success = self.camera_manager.set_roi(camera_id, formatted_points)
            if not success:
                return jsonify({"success": False, "message": "Failed to set ROI"}), 500
            
            # Save the updated configuration
            self.camera_manager.save_cameras()
            
            return jsonify({
                "success": True, 
                "message": "ROI set successfully",
                "roi": formatted_points
            })
            
        except Exception as e:
            print(f"Error setting ROI: {str(e)}")
            return jsonify({
                "success": False, 
                "message": f"Internal server error: {str(e)}"
            }), 500
    
    def set_threshold(self):
        """Set threshold, density threshold, and priority for a camera"""
        try:
            data = request.get_json()
            print(f"DEBUG: Received data in set_threshold: {data}")
            camera_id = data.get('id')
            threshold = data.get('threshold')
            density_threshold = data.get('density_threshold')
            priority = data.get('priority')
            
            print(f"DEBUG: Processing values: threshold={threshold}, density_threshold={density_threshold}, priority={priority}")
            
            if not camera_id:
                return jsonify({'success': False, 'message': 'Camera ID is required'})
            
            # Validate priority
            if priority is not None:
                try:
                    priority = int(priority)
                    if not (1 <= priority <= 6):
                        return jsonify({'success': False, 'message': 'Priority must be between 1 and 6'})
                except (ValueError, TypeError):
                    return jsonify({'success': False, 'message': 'Invalid priority value'})
            
            # Get current camera settings
            current_camera = self.camera_manager.get_camera(camera_id)
            if not current_camera:
                return jsonify({'success': False, 'message': 'Camera not found'})
            
            # Create updates dictionary with only the changed values
            updates = {}
            if threshold is not None:
                updates['threshold'] = int(threshold)
            if density_threshold is not None:
                updates['density_threshold'] = int(density_threshold)
            if priority is not None:
                updates['priority'] = priority
            
            print(f"DEBUG: Applying updates: {updates}")
            
            # Update camera configuration using update_camera
            if self.camera_manager.update_camera(camera_id, updates):
                # Update crowd data with new settings in a thread-safe way
                with self.data_lock:
                    if camera_id in self.crowd_data:
                        self.crowd_data[camera_id].update(updates)
                
                print(f"DEBUG: Updated camera {camera_id} settings: {updates}")
                return jsonify({'success': True})
            else:
                print(f"DEBUG: Failed to update camera {camera_id} settings")
                return jsonify({'success': False, 'message': 'Failed to update camera settings'})
                
        except Exception as e:
            print(f"Error in set_threshold: {str(e)}")
            return jsonify({'success': False, 'message': f'Internal server error: {str(e)}'})
    
    def get_crowd_data(self):
        """API endpoint to get crowd data for all cameras"""
        with self.data_lock:
            return jsonify(self.crowd_data)
    
    def get_alerts(self):
        """API endpoint to get alerts"""
        if self.alert_manager:
            return jsonify(self.alert_manager.get_alert_history())
        return jsonify([])
    
    def get_performance(self):
        """API endpoint to get performance data"""
        try:
            with self.data_lock:
                print("Sending performance data:", self.performance_data)
                return jsonify(self.performance_data)
        except Exception as e:
            print(f"Error getting performance data: {str(e)}")
            return jsonify({"error": str(e)}), 500
    
    def update_crowd_data(self, camera_id, count, timestamp=None, detection_mode=None, fps=None, latency=None):
        """
        Update crowd count data for a camera
        
        Parameters:
            camera_id: ID of the camera
            count: Current crowd count
            timestamp: Timestamp of the update (default: current time)
            detection_mode: Current detection mode (can be dict with model info)
            fps: Current frames per second
            latency: Current processing latency in seconds
        """
        if timestamp is None:
            timestamp = time.time()
            
        with self.data_lock:
            # Get current camera settings from camera manager
            camera_info = self.camera_manager.get_camera(camera_id) if self.camera_manager else None
            
            if camera_id not in self.crowd_data:
                self.crowd_data[camera_id] = {
                    'current_count': 0,
                    'history': [],
                    'detection_mode': 'YOLO',
                    'model_info': '',
                    'density_threshold': '',
                    'priority': camera_info.get('priority', 3) if camera_info else 3,  # Get priority from camera info
                    'fps': 0,
                    'latency': 0
                }
            
            # Update current data
            self.crowd_data[camera_id]['current_count'] = count
            if detection_mode:
                if isinstance(detection_mode, dict):
                    self.crowd_data[camera_id]['detection_mode'] = detection_mode.get('mode', 'YOLO')
                    self.crowd_data[camera_id]['model'] = detection_mode.get('model', '')
                    self.crowd_data[camera_id]['density_threshold'] = detection_mode.get('threshold', '')
                    self.crowd_data[camera_id]['transition'] = detection_mode.get('transition', 0.0)
                    self.crowd_data[camera_id]['hybrid'] = detection_mode.get('hybrid', False)
                else:
                    self.crowd_data[camera_id]['detection_mode'] = detection_mode
                    self.crowd_data[camera_id]['hybrid'] = False
            if fps is not None:
                self.crowd_data[camera_id]['fps'] = fps
            if latency is not None:
                self.crowd_data[camera_id]['latency'] = latency
            
            # Update priority from camera info if available
            if camera_info and 'priority' in camera_info:
                self.crowd_data[camera_id]['priority'] = camera_info['priority']
            
            # Add to history with proper timestamp formatting
            history_entry = {
                'count': count,
                'timestamp': timestamp,
                'datetime': datetime.now().strftime('%I:%M:%S %p'),  # Use real system time
                'detection_mode': self.crowd_data[camera_id]['detection_mode'],
                'model_info': self.crowd_data[camera_id].get('model', ''),
                'density_threshold': self.crowd_data[camera_id]['density_threshold'],
                'priority': self.crowd_data[camera_id]['priority'],  # Include priority in history
                'fps': fps or self.crowd_data[camera_id]['fps'],
                'latency': latency or self.crowd_data[camera_id]['latency']
            }
            
            self.crowd_data[camera_id]['history'].append(history_entry)
            
            # Keep only last 100 history entries
            if len(self.crowd_data[camera_id]['history']) > 100:
                self.crowd_data[camera_id]['history'].pop(0)
    
    def update_performance(self, latency, fps):
        """
        Update performance metrics
        
        Parameters:
            latency: Processing latency in seconds
            fps: Frames per second
        """
        with self.data_lock:
            self.performance_data["latency"].append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "value": latency
            })
            self.performance_data["fps"].append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "value": fps
            })
            
            # Keep only the last 100 data points
            if len(self.performance_data["latency"]) > 100:
                self.performance_data["latency"] = self.performance_data["latency"][-100:]
            if len(self.performance_data["fps"]) > 100:
                self.performance_data["fps"] = self.performance_data["fps"][-100:]
    
    def update_frame(self, camera_id, frame):
        """
        Update the latest frame for a camera
        
        Parameters:
            camera_id: ID of the camera
            frame: Current video frame
        """
        # Use lower quality JPEG encoding for better performance
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 80]  # Use 80% quality instead of default 95%
        _, encoded_frame = cv2.imencode('.jpg', frame, encode_params)
        self.streaming_frames[camera_id] = encoded_frame.tobytes()
    
    def video_feed(self, camera_id):
        """Generate video feed for a specific camera"""
        try:
            def generate():
                while True:
                    try:
                        with self.data_lock:
                            if camera_id not in self.streaming_frames:
                                # If no frame is available, wait a bit and try again
                                time.sleep(0.1)
                                continue
                                
                            frame = self.streaming_frames[camera_id]
                            
                        # Yield the frame in the multipart format
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                        
                        # Small delay to prevent overwhelming the client
                        time.sleep(0.033)  # ~30 FPS
                        
                    except Exception as e:
                        print(f"Error in video feed generation for camera {camera_id}: {str(e)}")
                        # If there's an error, wait a bit before retrying
                        time.sleep(0.5)
                        continue
            
            response = Response(generate(),
                          mimetype='multipart/x-mixed-replace; boundary=frame')
            
            # Add CORS headers
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
            response.headers['Access-Control-Allow-Headers'] = 'Origin, Content-Type'
            response.headers['Access-Control-Expose-Headers'] = 'Content-Type'
            
            return response
            
        except Exception as e:
            print(f"Error setting up video feed for camera {camera_id}: {str(e)}")
            return Response("Error setting up video feed", status=500)
    
    def test_camera_connection(self):
        """API endpoint to test camera connection with different URL formats"""
        if not self.camera_manager:
            return jsonify({"success": False, "message": "Camera manager not available"})
        
        data = request.json
        if not data or "url" not in data:
            return jsonify({"success": False, "message": "Missing camera URL"})
        
        base_url = data["url"]
        
        # Clean the URL to get just the base
        if '://' in base_url:
            # Split by ://
            protocol, rest = base_url.split('://', 1)
            # Remove any paths
            if '/' in rest:
                rest = rest.split('/', 1)[0]
            base_url = f"{protocol}://{rest}"
        
        # Test all formats
        results = self.camera_manager.test_all_camera_formats(base_url)
        
        return jsonify({
            "success": True,
            "test_results": results
        })
    
    def reset_cameras(self):
        """API endpoint to reset all cameras"""
        try:
            if not self.camera_manager:
                return jsonify({"success": False, "message": "Camera manager not available"})
            
            # Reset cameras
            self.camera_manager.reset_cameras()
            
            # Clear crowd data
            with self.data_lock:
                self.crowd_data = {}
                if hasattr(self, 'performance_data'):
                    self.performance_data = {'latency': [], 'fps': []}
                if hasattr(self, 'streaming_frames'):
                    self.streaming_frames = {}
            
            # Clear alerts
            if self.alert_manager:
                self.alert_manager.reset_alert_history()
            
            return jsonify({"success": True, "message": "All cameras and related data have been reset successfully"})
            
        except Exception as e:
            print(f"Error resetting cameras: {str(e)}")
            return jsonify({"success": False, "message": f"Error resetting cameras: {str(e)}"})
    
    def clear_camera_data(self, camera_id):
        """
        Clear crowd data for a camera when it disconnects
        
        Parameters:
            camera_id: ID of the camera that disconnected
        """
        with self.data_lock:
            if camera_id in self.crowd_data:
                # Add a zero count data point to show that the camera is offline
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.crowd_data[camera_id]["current_count"] = 0
                self.crowd_data[camera_id]["history"].append({
                    "timestamp": timestamp,
                    "count": 0,
                    "status": "offline"  # Add a status indicator
                })
                if len(self.crowd_data[camera_id]["history"]) > 100:
                    self.crowd_data[camera_id]["history"] = self.crowd_data[camera_id]["history"][-100:]
    
    def remove_camera_api(self):
        """API endpoint to remove a camera"""
        if not self.camera_manager:
            return jsonify({"success": False, "message": "Camera manager not available"})
        
        data = request.json
        if not data or "id" not in data:
            return jsonify({"success": False, "message": "Missing camera ID"})
        
        camera_id = data["id"]
        
        try:
            # First, remove the camera from the camera manager
            if not self.camera_manager.remove_camera(camera_id):
                return jsonify({"success": False, "message": "Failed to remove camera from camera manager"})
            
            # Then clean up dashboard data
            with self.data_lock:
                # Remove streaming frame
                if camera_id in self.streaming_frames:
                    del self.streaming_frames[camera_id]
                
                # Remove crowd data
                if camera_id in self.crowd_data:
                    del self.crowd_data[camera_id]
                
                # Clean up performance data related to this camera
                if hasattr(self, 'performance_data'):
                    if 'latency' in self.performance_data:
                        self.performance_data['latency'] = [lat for lat in self.performance_data['latency'] 
                                                          if lat.get('camera_id') != camera_id]
                    if 'fps' in self.performance_data:
                        self.performance_data['fps'] = [fps for fps in self.performance_data['fps'] 
                                                      if fps.get('camera_id') != camera_id]
                
                # Clean up analytics data
                if hasattr(self, 'analytics_data'):
                    if 'crowd_trends' in self.analytics_data:
                        self.analytics_data['crowd_trends'] = [trend for trend in self.analytics_data['crowd_trends'] 
                                                             if trend.get('camera_id') != camera_id]
                    if 'performance_metrics' in self.analytics_data:
                        self.analytics_data['performance_metrics'] = [metric for metric in self.analytics_data['performance_metrics'] 
                                                                    if metric.get('camera_id') != camera_id]
            
            # Notify alert manager to remove camera alerts
            if self.alert_manager:
                try:
                    print(f"Attempting to remove alerts for camera {camera_id} from alert manager")
                    alert_removal_success = self.alert_manager.remove_camera_alerts(camera_id)
                    if not alert_removal_success:
                        print(f"Warning: Failed to remove alerts for camera {camera_id}")
                except Exception as alert_error:
                    print(f"Error removing alerts for camera {camera_id}: {str(alert_error)}")
                    import traceback
                    traceback.print_exc()
                    # Continue with the rest of the method even if alert removal fails
            
            # Call the callback if set
            if hasattr(self, 'on_camera_removed') and self.on_camera_removed:
                self.on_camera_removed(camera_id)
            
            return jsonify({"success": True, "message": "Camera removed successfully"})
            
        except Exception as e:
            print(f"Error in remove_camera_api: {str(e)}")
            import traceback
            traceback.print_exc()
            return jsonify({"success": False, "message": f"Error removing camera: {str(e)}"})
    
    def clear_alerts(self):
        """API endpoint to clear all alerts"""
        try:
            if not self.alert_manager:
                return jsonify({"success": False, "message": "Alert manager not available"})
            
            success = self.alert_manager.reset_alert_history()
            return jsonify({"success": success, "message": "Alert history cleared" if success else "Failed to clear alert history"})
        except Exception as e:
            print(f"Error in clear_alerts: {str(e)}")
            return jsonify({"success": False, "message": f"Error clearing alerts: {str(e)}"})
    
    def update_performance_metrics(self, fps=None, latency=None, camera_id=None):
        """
        Update performance metrics for display
        
        Args:
            fps: Frames per second
            latency: Processing latency in seconds
            camera_id: ID of the camera (optional)
        """
        # Get current timestamp
        timestamp = time.time()
        time_str = datetime.now().strftime("%H:%M:%S")
        
        with self.data_lock:
            # Initialize performance data if not exists
            if not hasattr(self, 'performance_data'):
                self.performance_data = {
                    'fps': [],
                    'latency': []
                }
            
            # Update FPS data
            if fps is not None:
                # Add new data point
                data_point = {
                    'timestamp': timestamp,
                    'datetime': time_str,
                    'value': fps if isinstance(fps, (int, float)) else fps.get('actual', 0)
                }
                
                # Add camera_id if provided
                if camera_id:
                    data_point['camera_id'] = camera_id
                    
                self.performance_data['fps'].append(data_point)
                
                # Keep only the last 50 data points
                if len(self.performance_data['fps']) > 50:
                    self.performance_data['fps'].pop(0)
            
            # Update latency data
            if latency is not None:
                # Add new data point
                data_point = {
                    'timestamp': timestamp,
                    'datetime': time_str,
                    'value': latency
                }
                
                # Add camera_id if provided
                if camera_id:
                    data_point['camera_id'] = camera_id
                    
                self.performance_data['latency'].append(data_point)
                
                # Keep only the last 50 data points
                if len(self.performance_data['latency']) > 50:
                    self.performance_data['latency'].pop(0)
    
    def check_feed(self):
        """API endpoint to check camera feed status"""
        try:
            print("Check feed request received")
            data = request.get_json()
            
            if not data:
                print("No data received in request")
                return jsonify({
                    "success": False,
                    "message": "No data provided",
                    "status": "error"
                }), 400
            
            camera_id = data.get('id')
            if not camera_id:
                print("No camera ID provided")
                return jsonify({
                    "success": False,
                    "message": "Camera ID is required",
                    "status": "error"
                }), 400
            
            print(f"Checking feed for camera: {camera_id}")
            
            # Get camera info
            if not self.camera_manager:
                print("Camera manager not available")
                return jsonify({
                    "success": False,
                    "message": "Camera manager not available",
                    "status": "error"
                }), 500
            
            camera_info = self.camera_manager.get_camera(camera_id)
            if not camera_info:
                print(f"Camera {camera_id} not found")
                return jsonify({
                    "success": False,
                    "message": f"Camera {camera_id} not found",
                    "status": "error"
                }), 404
            
            # Test camera connection
            camera_url = camera_info.get('url')
            print(f"Testing connection to camera URL: {camera_url}")
            
            try:
                cap = cv2.VideoCapture(camera_url)
                if not cap.isOpened():
                    print(f"Failed to open camera {camera_id}")
                    return jsonify({
                        "success": False,
                        "message": "Failed to open camera",
                        "status": "error"
                    }), 500
                
                # Try to read a frame
                ret, frame = cap.read()
                cap.release()
                
                if not ret or frame is None:
                    print(f"Failed to read frame from camera {camera_id}")
                    return jsonify({
                        "success": False,
                        "message": "Failed to read frame",
                        "status": "error"
                    }), 500
                
                # Success response
                print(f"Successfully checked feed for camera {camera_id}")
                return jsonify({
                    "success": True,
                    "message": "Feed is working",
                    "status": "ok",
                    "frame_size": {
                        "width": frame.shape[1],
                        "height": frame.shape[0]
                    }
                })
                
            except Exception as e:
                print(f"Error testing camera {camera_id}: {str(e)}")
                return jsonify({
                    "success": False,
                    "message": f"Error testing camera: {str(e)}",
                    "status": "error"
                }), 500
                
        except Exception as e:
            print(f"Unexpected error in check_feed: {str(e)}")
            return jsonify({
                "success": False,
                "message": f"Unexpected error: {str(e)}",
                "status": "error"
            }), 500
    
    def get_resource_usage(self):
        """API endpoint to get system resource usage (CPU, RAM, Disk, GPU)"""
        usage = {
            'cpu_percent': psutil.cpu_percent(interval=0.5),
            'ram_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent,
            'gpu_percent': None
        }
        
        # Check if CUDA is available
        if torch.cuda.is_available():
            try:
                # Get GPU memory usage instead of utilization
                gpu_memory_allocated = torch.cuda.memory_allocated(0)
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory
                gpu_memory_percent = (gpu_memory_allocated / gpu_memory_total) * 100
                usage['gpu_percent'] = round(gpu_memory_percent, 1)
                
                # Try to get GPU utilization using nvidia-smi if available
                try:
                    import subprocess
                    nvidia_smi = "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits"
                    gpu_util = subprocess.check_output(nvidia_smi.split()).decode('ascii').strip()
                    usage['gpu_percent'] = float(gpu_util)
                except:
                    # If nvidia-smi fails, use memory usage as fallback
                    pass
                    
            except Exception as e:
                print(f"Error getting GPU usage: {e}")
                usage['gpu_percent'] = None
            
        return jsonify(usage)
    
    def run(self, host='0.0.0.0', port=5000, debug=False, threaded=True):
        """
        Run the dashboard web server
        
        Parameters:
            host: Host to bind the server to
            port: Port to bind the server to
            debug: Whether to run in debug mode
            threaded: Whether to run in threaded mode
        """
        self.app.run(host=host, port=port, debug=debug, threaded=threaded)
