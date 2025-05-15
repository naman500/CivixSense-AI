# Main entry point for the Real-Time Crowd Detection System
import threading
import time
import cv2
import numpy as np
import os
import sys
from flask import Flask, render_template
from datetime import datetime

# Get the absolute path to the project root and models directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, 'models'))

# Add RTCDM directory to path if running as main script
if __name__ == "__main__":
    # If we're running this file directly
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    from cameras.camera_manager import CameraManager
    from detection.yolo_model import YOLOModel
    from detection.detection_utils import combine_detections, draw_roi, draw_crowd_info, calculate_performance_metrics
    from alerts.alert_manager import AlertManager
    from dashboard.dashboard import Dashboard
    from detection.hybrid_detector import HybridDetector
else:
    # If imported from another module
    from RTCDM.cameras.camera_manager import CameraManager
    from RTCDM.detection.yolo_model import YOLOModel
    from RTCDM.detection.detection_utils import combine_detections, draw_roi, draw_crowd_info, calculate_performance_metrics
    from RTCDM.alerts.alert_manager import AlertManager
    from RTCDM.dashboard.dashboard import Dashboard
    from RTCDM.detection.hybrid_detector import HybridDetector

import torch

# Create Flask app
dashboard_dir = os.path.join(PROJECT_ROOT, 'dashboard')
template_dir = os.path.join(dashboard_dir, 'templates')

# Ensure template directory exists
os.makedirs(template_dir, exist_ok=True)

# Template file path
template_file = "dashboard_template.html"
source_template = os.path.join(dashboard_dir, template_file)
dest_template = os.path.join(template_dir, template_file)

# Copy template if needed
if os.path.exists(source_template) and not os.path.exists(dest_template):
    try:
        import shutil
        shutil.copyfile(source_template, dest_template)
        print(f"Copied template file to {dest_template}")
    except Exception as e:
        print(f"Warning: Could not copy template file: {str(e)}")

# Create a single Flask app
app = Flask("rtcdm", 
            static_folder=os.path.join(dashboard_dir, 'static'),
            template_folder=template_dir)

def check_models(yolo_size='n'):
    """
    Check if required models exist and download them if needed
    
    Args:
        yolo_size: Size of the YOLO model ('n', 's', 'm', 'l', 'x')
        
    Returns:
        Tuple of model paths (YOLO_MODEL_PATH, YOLO_M_MODEL_PATH)
    """
    # Ensure models directory exists
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        print(f"Created models directory at {MODELS_DIR}")
    
    # Define model paths - use ONNX format for YOLO
    YOLO_MODEL_PATH = os.path.join(MODELS_DIR, f'yolov8{yolo_size}.onnx')
    YOLO_M_MODEL_PATH = os.path.join(MODELS_DIR, 'yolov8m.onnx')
    
    # Check if models exist
    models_missing = []
    if not os.path.exists(YOLO_MODEL_PATH):
        models_missing.append(f'yolov8{yolo_size}.onnx')
        # Download YOLO model
        print(f"Downloading YOLOv8{yolo_size} model...")
        try:
            from ultralytics import YOLO
            model_name = f'yolov8{yolo_size}'
            # Create model instance with explicit task parameter
            model = YOLO(model_name, task='detect')
            
            # Export to ONNX format
            print(f"Exporting {model_name} to ONNX format...")
            model.export(format='onnx', imgsz=640)
            
            # Move file to models directory
            temp_path = f'{model_name}.onnx'
            if os.path.exists(temp_path):
                os.rename(temp_path, YOLO_MODEL_PATH)
                print(f"Successfully exported YOLOv8{yolo_size} to ONNX format at {YOLO_MODEL_PATH}")
            else:
                raise FileNotFoundError(f"Expected ONNX file {temp_path} was not created")
        except Exception as e:
            print(f"Error exporting YOLO model to ONNX: {str(e)}")
            print("Trying to install onnxruntime...")
            try:
                import subprocess
                subprocess.check_call([sys.executable, "-m", "pip", "install", "onnxruntime"])
                print("onnxruntime installed, retrying export...")
                # Try again with onnxruntime installed
                model = YOLO(model_name, task='detect')
                model.export(format='onnx', imgsz=640)
                temp_path = f'{model_name}.onnx'
                if os.path.exists(temp_path):
                    os.rename(temp_path, YOLO_MODEL_PATH)
                    print(f"Successfully exported YOLOv8{yolo_size} to ONNX format at {YOLO_MODEL_PATH}")
                else:
                    raise FileNotFoundError(f"Expected ONNX file {temp_path} was not created")
            except Exception as install_error:
                print(f"Error installing onnxruntime or exporting model: {str(install_error)}")
                sys.exit(1)
    
    if not os.path.exists(YOLO_M_MODEL_PATH):
        models_missing.append('yolov8m.onnx')
        # Download YOLOv8m model
        print("Downloading YOLOv8m model...")
        try:
            from ultralytics import YOLO
            model = YOLO('yolov8m', task='detect')
            
            # Export to ONNX format
            print("Exporting YOLOv8m to ONNX format...")
            model.export(format='onnx', imgsz=640)
            
            # Move file to models directory
            temp_path = 'yolov8m.onnx'
            if os.path.exists(temp_path):
                os.rename(temp_path, YOLO_M_MODEL_PATH)
                print(f"Successfully exported YOLOv8m to ONNX format at {YOLO_M_MODEL_PATH}")
            else:
                raise FileNotFoundError(f"Expected ONNX file {temp_path} was not created")
        except Exception as e:
            print(f"Error exporting YOLOv8m model to ONNX: {str(e)}")
            sys.exit(1)
    
    return YOLO_MODEL_PATH, YOLO_M_MODEL_PATH

# Global flag to control processing
running = True
# Store global model references
global_yolo_model = None
global_alert_manager = None
global_dashboard = None
global_hybrid_detector = None

# Store active camera threads
camera_threads = {}

def process_camera(camera_id, camera_info, hybrid_detector, alert_manager, dashboard):
    """
    Process frames from a camera using hybrid detection
    
    Parameters:
        camera_id: ID of the camera
        camera_info: Camera configuration
        hybrid_detector: HybridDetector instance
        alert_manager: AlertManager instance
        dashboard: Dashboard instance
    """
    camera_url = camera_info['url']
    # Initialize settings cache
    settings_cache = {
        'threshold': camera_info.get('threshold', 50),
        'density_threshold': camera_info.get('density_threshold', 20),
        'priority': camera_info.get('priority', 3),
        'target_fps': int(camera_info.get('fps', 30))
    }
    last_settings_check = time.time()
    settings_check_interval = 1.0  # Check settings every second
    
    # Connection retry parameters
    max_retries = 10
    retry_count = 0
    retry_delay = 5  # seconds
    connected = False
    
    # Set a flag on the thread to track if it should be running
    thread = threading.current_thread()
    if not hasattr(thread, "stop_processing"):
        thread.stop_processing = False
    
    print(f"Starting camera thread for camera {camera_id} with URL: {camera_url}")
    print(f"Target FPS: {settings_cache['target_fps']}")
    
    # Create a blank frame for displaying when camera is unavailable
    blank_frame = create_blank_frame(camera_id, camera_url)
    dashboard.update_frame(camera_id, blank_frame)
    
    # Reference to VideoCapture object
    cap = None
    
    # Performance tracking variables
    frame_times = []
    last_fps_update = time.time()
    fps_update_interval = 1.0  # Update FPS every second
    frame_count = 0
    frames_processed = 0
    frames_skipped = 0
    current_fps = 0
    
    # Buffer management
    buffer_size = 10
    frame_buffer = []
    last_frame_time = time.time()
    frame_interval = 1.0 / settings_cache['target_fps'] if settings_cache['target_fps'] > 0 else float('inf')
    
    # Processing loop
    while running:
        try:
            # Check if this specific camera thread should stop
            if thread.stop_processing:
                print(f"Camera {camera_id} thread received stop signal, exiting...")
                break

            # Check if this camera still exists in the camera manager
            if dashboard.camera_manager and camera_id not in dashboard.camera_manager.get_cameras():
                print(f"Camera {camera_id} no longer exists in camera manager, stopping thread...")
                break
            
            # Check for settings updates periodically
            current_time = time.time()
            if current_time - last_settings_check >= settings_check_interval:
                camera_info = dashboard.camera_manager.get_camera(camera_id)
                if camera_info is not None:
                    # Only update if settings have changed
                    new_threshold = camera_info.get('threshold', 50)
                    new_density_threshold = camera_info.get('density_threshold', 20)
                    new_priority = camera_info.get('priority', 3)
                    new_target_fps = int(camera_info.get('fps', 30))
                    
                    if (new_threshold != settings_cache['threshold'] or
                        new_density_threshold != settings_cache['density_threshold'] or
                        new_priority != settings_cache['priority'] or
                        new_target_fps != settings_cache['target_fps']):
                        
                        # Update cache
                        settings_cache.update({
                            'threshold': new_threshold,
                            'density_threshold': new_density_threshold,
                            'priority': new_priority,
                            'target_fps': new_target_fps
                        })
                        
                        # Update frame interval if FPS changed
                        frame_interval = 1.0 / new_target_fps if new_target_fps > 0 else float('inf')
                        
                        # Update hybrid detector threshold if changed
                        if hasattr(hybrid_detector, 'set_density_threshold'):
                            hybrid_detector.set_density_threshold(new_density_threshold)
                        
                        print(f"Updated settings for camera {camera_id}: threshold={new_threshold}, density={new_density_threshold}, priority={new_priority}, fps={new_target_fps}")
                
                last_settings_check = current_time
            
            # Try to connect if not connected
            if not connected:
                try:
                    # Handle webcam index
                    if camera_url.isdigit():
                        camera_index = int(camera_url)
                        cap = cv2.VideoCapture(camera_index)
                    else:
                        cap = cv2.VideoCapture(camera_url)
                        
                    if cap.isOpened():
                        # Set buffer size
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
                        # Set resolution
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                        # Set FPS
                        cap.set(cv2.CAP_PROP_FPS, settings_cache['target_fps'])
                        # Set codec
                        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
                        connected = True
                        print(f"Successfully connected to camera {camera_id}")
                        retry_count = 0
                    else:
                        raise Exception("Failed to open camera stream")
                except Exception as e:
                    print(f"Error connecting to camera {camera_id}: {str(e)}")
                    retry_count += 1
                    if retry_count >= max_retries:
                        print(f"Max retries reached for camera {camera_id}, stopping thread...")
                        break
                    time.sleep(retry_delay)
                    continue
            
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print(f"Error reading frame from camera {camera_id}")
                connected = False
                continue
            
            # Buffer management
            current_time = time.time()
            time_since_last_frame = current_time - last_frame_time
            
            # Strict FPS control - sleep if we're processing too fast
            if time_since_last_frame < frame_interval:
                time.sleep(frame_interval - time_since_last_frame)
                continue
            
            # Add frame to buffer
            frame_buffer.append((frame, current_time))
            if len(frame_buffer) > buffer_size:
                frame_buffer.pop(0)  # Remove oldest frame
            
            # Process frame
            try:
                # Start timing the processing
                processing_start = time.time()

                # Fetch ROI for this camera
                roi_points = dashboard.camera_manager.get_roi(camera_id)

                # Call detection with ROI
                count, density_map, detection_mode = hybrid_detector.detect(frame, roi_points=roi_points)

                # Debug: print the count returned by detector
                print(f"[DEBUG] Camera {camera_id} detected count: {count}")

                # Calculate processing latency
                processing_time = time.time() - processing_start
                
                # Update dashboard with the correct count
                dashboard.update_crowd_data(camera_id, count, detection_mode=detection_mode, timestamp=time.time())
                dashboard.update_frame(camera_id, frame)
                
                # Check for alerts
                if count > settings_cache['threshold']:
                    alert_manager.add_alert(
                        camera_id=camera_id,
                        camera_name=camera_info.get('name', f"Camera {camera_id}"),
                        crowd_count=count,
                        threshold=settings_cache['threshold'],
                        severity='high' if count > settings_cache['threshold'] * 2 else 'medium'
                    )
                
                frames_processed += 1
                last_frame_time = current_time
                
            except Exception as e:
                print(f"Error processing frame from camera {camera_id}: {str(e)}")
                frames_skipped += 1
                continue
            
            # Update performance metrics
            frame_count += 1
            if current_time - last_fps_update >= fps_update_interval:
                current_fps = frame_count / (current_time - last_fps_update)
                # Update dashboard with both FPS and latency
                dashboard.update_performance_metrics(
                    fps={
                        'target': settings_cache['target_fps'],
                        'actual': round(current_fps, 1)
                    },
                    latency=processing_time,
                    camera_id=camera_id
                )
                # Update camera data with current metrics
                dashboard.update_crowd_data(
                    camera_id, 
                    count, 
                    detection_mode=detection_mode,
                    fps={
                        'target': settings_cache['target_fps'],
                        'actual': round(current_fps, 1)
                    },
                    latency=processing_time,
                    timestamp=time.time()
                )
                frame_count = 0
                last_fps_update = current_time
            
            # Small sleep to prevent CPU overload
            time.sleep(0.01)
            
        except Exception as e:
            print(f"Unexpected error in camera {camera_id} thread: {str(e)}")
            connected = False
            time.sleep(retry_delay)
    
    # Cleanup
    if cap is not None:
        cap.release()
    print(f"Camera {camera_id} thread stopped")

def create_blank_frame(camera_id, camera_url=None, message=None, error=None):
    """Create a blank frame with camera info for display when camera is unavailable"""
    height, width = 480, 640
    blank = np.ones((height, width, 3), dtype=np.uint8) * 240  # Light gray
    
    # Add camera info
    cv2.putText(blank, f"Camera ID: {camera_id}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Add URL info if available
    if camera_url:
        # Truncate URL if too long
        display_url = camera_url
        if len(camera_url) > 40:
            display_url = camera_url[:37] + "..."
        
        cv2.putText(blank, f"URL: {display_url}", (20, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    else:
        cv2.putText(blank, "URL: Not configured", (20, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # Add error message if provided
    if error:
        cv2.putText(blank, error, (20, height//2 - 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # Add custom message if provided
    if message:
        cv2.putText(blank, message, (20, height//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    else:
        cv2.putText(blank, "Camera not available", (20, height//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
    cv2.putText(blank, "Please check the connection", (20, height//2 + 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return blank
    
def start_camera_threads(camera_manager, hybrid_detector, alert_manager, dashboard):
    """
    Start threads for processing each camera
    
    Parameters:
        camera_manager: Camera manager instance
        hybrid_detector: HybridDetector instance
        alert_manager: Alert manager instance
        dashboard: Dashboard instance
    """
    global camera_threads
    
    # Get all cameras
    cameras = camera_manager.get_cameras()
    
    # Sort cameras by priority (highest to lowest)
    sorted_cameras = []
    for camera_id, camera_info in cameras.items():
        sorted_cameras.append((camera_id, camera_info))
    
    # Sort by priority (highest=5 first, lowest=1 last)
    sorted_cameras.sort(key=lambda x: x[1].get('priority', 3), reverse=True)
    
    print(f"Starting camera threads in priority order:")
    for camera_id, camera_info in sorted_cameras:
        priority = camera_info.get('priority', 3)
        print(f"Starting thread for camera {camera_id} (Priority: {priority})")
        
        # Create and start a thread for this camera
        start_camera_thread(camera_id, camera_info, hybrid_detector, alert_manager, dashboard)
    
    return list(camera_threads.values())

def start_camera_thread(camera_id, camera_info, hybrid_detector, alert_manager, dashboard):
    """
    Start a single camera processing thread
    
    Parameters:
        camera_id: ID of the camera
        camera_info: Camera information dictionary
        hybrid_detector: HybridDetector instance
        alert_manager: Alert manager instance
        dashboard: Dashboard instance
    """
    global camera_threads
    
    # Check if a thread is already running for this camera
    if camera_id in camera_threads and camera_threads[camera_id].is_alive():
        print(f"Thread for camera {camera_id} is already running")
        return
    
    # Create and start a new thread
    thread = threading.Thread(
        target=process_camera,
        args=(camera_id, camera_info, hybrid_detector, alert_manager, dashboard),
        daemon=True
    )
    thread.start()
    camera_threads[camera_id] = thread
    
    print(f"Started new thread for camera {camera_id}")
    return thread

def on_camera_added(camera_id, camera_info):
    """
    Callback when a new camera is added through the dashboard
    
    Parameters:
        camera_id: ID of the camera
        camera_info: Camera information dictionary
    """
    global global_hybrid_detector, global_alert_manager, global_dashboard
    
    print(f"Starting thread for newly added camera {camera_id}")
    start_camera_thread(
        camera_id, 
        camera_info, 
        global_hybrid_detector, 
        global_alert_manager, 
        global_dashboard
    )

def on_camera_removed(camera_id):
    """
    Callback when a camera is removed through the dashboard
    
    Parameters:
        camera_id: ID of the camera that was removed
    """
    global camera_threads, running, global_dashboard
    
    print(f"Camera {camera_id} was removed, terminating its processing thread...")
    
    # Check if there's a thread for this camera
    if camera_id in camera_threads:
        thread = camera_threads[camera_id]
        
        # Set a flag to stop processing that specific camera
        if thread.is_alive():
            setattr(thread, "stop_processing", True)
            
            # Give the thread a moment to clean up
            thread.join(timeout=1.0)
            
            # If the thread is still alive after timeout, it might be stuck
            if thread.is_alive():
                print(f"Warning: Thread for camera {camera_id} did not exit cleanly")
        
        # Remove it from our tracking dictionary
        camera_threads.pop(camera_id, None)
        
        # Clear all data related to this camera in the dashboard
        if global_dashboard:
            # Clear crowd data
            with global_dashboard.data_lock:
                if camera_id in global_dashboard.crowd_data:
                    del global_dashboard.crowd_data[camera_id]
                
                # Clear performance data
                if hasattr(global_dashboard, 'performance_data'):
                    if 'latency' in global_dashboard.performance_data:
                        global_dashboard.performance_data['latency'] = [
                            lat for lat in global_dashboard.performance_data['latency'] 
                            if lat.get('camera_id') != camera_id
                        ]
                    if 'fps' in global_dashboard.performance_data:
                        global_dashboard.performance_data['fps'] = [
                            fps for fps in global_dashboard.performance_data['fps'] 
                            if fps.get('camera_id') != camera_id
                        ]
                
                # Clear streaming frames
                if camera_id in global_dashboard.streaming_frames:
                    del global_dashboard.streaming_frames[camera_id]
        
        print(f"Cleanup for camera {camera_id} complete, processing thread terminated")

def validate_camera_config(camera_manager):
    """
    Validate camera configuration and remove invalid entries
    
    Parameters:
        camera_manager: Camera manager instance
        
    Returns:
        fixed_count: Number of fixed entries
    """
    cameras = camera_manager.get_cameras()
    invalid_cameras = []
    
    # Find invalid camera entries
    for camera_id, camera_info in cameras.items():
        if not isinstance(camera_info, dict):
            print(f"Warning: Camera {camera_id} has invalid configuration type: {type(camera_info)}")
            invalid_cameras.append(camera_id)
            continue
        
        if 'url' not in camera_info:
            print(f"Warning: Camera {camera_id} missing URL field")
            invalid_cameras.append(camera_id)
    
    # Remove invalid cameras
    for camera_id in invalid_cameras:
        cameras.pop(camera_id, None)
        print(f"Removed invalid camera configuration: {camera_id}")
    
    # Save cleaned configuration
    if invalid_cameras:
        camera_manager.save_cameras()
        print(f"Fixed {len(invalid_cameras)} invalid camera entries")
    
    return len(invalid_cameras)

def initialize_application(yolo_size='n'):
    """
    Initialize application components for external use
    
    Args:
        yolo_size: Size of the YOLO model ('n', 's', 'm', 'l', 'x')
    """
    global global_alert_manager, global_dashboard, global_hybrid_detector, running, camera_threads, app
    
    # Check for required models with specified size
    YOLO_MODEL_PATH, YOLO_M_MODEL_PATH = check_models(yolo_size)
    
    # Initialize components
    camera_manager = CameraManager()
    
    # Reset cameras on startup - comment this line out if you want to keep existing cameras
    # camera_manager.reset_cameras()  # Commented to preserve camera settings
    
    # Validate camera configuration
    if 'validate_camera_config' in globals():
        fixed_entries = validate_camera_config(camera_manager)
        if fixed_entries > 0:
            print(f"Camera configuration file was updated to fix {fixed_entries} invalid entries")
    
    # Initialize alert manager and dashboard
    print("Initializing alert manager...")
    global_alert_manager = AlertManager()
    
    print("Initializing dashboard...")
    global_dashboard = Dashboard(camera_manager, global_alert_manager)
    
    # Register dashboard routes on our Flask app
    global_dashboard._register_routes(external_app=app)
    
    # Create hybrid detector
    print(f"Creating hybrid detector with YOLOv8{yolo_size}...")
    global_hybrid_detector = HybridDetector(
        yolo_model_path=YOLO_MODEL_PATH,
        density_threshold=20,
        yolo_model_size=yolo_size
    )
    
    # Load YOLOv8m model for high-density scenarios
    print("Loading YOLOv8m model for high-density scenarios...")
    global_hybrid_detector.load_yolo_m_model(YOLO_M_MODEL_PATH)
    
    # Set up the callbacks for when cameras are added or removed
    global_dashboard.on_camera_added = lambda camera_id, camera_info: start_camera_thread(
        camera_id, 
        camera_info, 
        global_hybrid_detector, 
        global_alert_manager, 
        global_dashboard
    )
    global_dashboard.on_camera_removed = on_camera_removed
    
    # Start camera processing threads
    print("Starting camera processing threads...")
    threads = start_camera_threads(
        camera_manager, 
        global_hybrid_detector, 
        global_alert_manager, 
        global_dashboard
    )
    
    # Return necessary components for external use
    return global_dashboard, global_alert_manager, global_hybrid_detector, threads

def main():
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Real-Time Crowd Detection System')
    parser.add_argument('--yolo-size', type=str, choices=['n', 's', 'm', 'l', 'x'], default='n',
                        help='YOLOv8 model size: n (nano), s (small), m (medium), l (large), x (extra large)')
    parser.add_argument('--reset-cameras', action='store_true',
                        help='Reset all camera configurations on startup')
    args = parser.parse_args()
    
    # Check for required models
    YOLO_MODEL_PATH, YOLO_M_MODEL_PATH = check_models(args.yolo_size)
    
    # Initialize components
    camera_manager = CameraManager()
    
    # Reset cameras on startup if requested
    if args.reset_cameras:
        print("Resetting camera configurations...")
        camera_manager.reset_cameras()
    
    # Validate camera configuration
    if 'validate_camera_config' in globals():
        fixed_entries = validate_camera_config(camera_manager)
        if fixed_entries > 0:
            print(f"Camera configuration file was updated to fix {fixed_entries} invalid entries")
    
    # Initialize alert manager and dashboard
    print("Initializing alert manager...")
    global_alert_manager = AlertManager()
    print("Initializing dashboard...")
    global_dashboard = Dashboard(camera_manager, global_alert_manager)
    
    # Create hybrid detector
    print(f"Creating hybrid detector with YOLOv8{args.yolo_size}...")
    global_hybrid_detector = HybridDetector(
        yolo_model_path=YOLO_MODEL_PATH,
        density_threshold=20,
        yolo_model_size=args.yolo_size
    )
    
    # Load YOLOv8m model for high-density scenarios
    print("Loading YOLOv8m model for high-density scenarios...")
    global_hybrid_detector.load_yolo_m_model(YOLO_M_MODEL_PATH)

    # Set up the callbacks for when cameras are added or removed
    global_dashboard.on_camera_added = lambda camera_id, camera_info: start_camera_thread(
        camera_id, 
        camera_info, 
        global_hybrid_detector, 
        global_alert_manager, 
        global_dashboard
    )
    global_dashboard.on_camera_removed = on_camera_removed
    
    # Start camera processing threads
    print("Starting camera processing threads...")
    threads = start_camera_threads(
        camera_manager, 
        global_hybrid_detector, 
        global_alert_manager, 
        global_dashboard
    )
    
    try:
        # Run the dashboard web server
        print("Starting dashboard web server...")
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        # Handle application shutdown
        print("Shutting down...")
        running = False
        
        # Wait for threads to finish
        for thread in threads:
            thread.join(timeout=1.0)
            
        print("Application shutdown complete")

if __name__ == "__main__":
    main()
