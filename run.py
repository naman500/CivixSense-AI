"""
Run script for the Real-Time Crowd Detection and Monitoring System
"""
import os
import sys
import argparse

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import main from RTCDM
from RTCDM.main import app, initialize_application

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Real-Time Crowd Detection and Monitoring System')
    parser.add_argument('--yolo-size', type=str, choices=['n', 's', 'm', 'l', 'x'], default='n',
                      help='YOLOv8 model size: n (nano), s (small), m (medium), l (large), x (extra large)')
    parser.add_argument('--reset-cameras', action='store_true',
                      help='Reset all camera configurations on startup')
    args = parser.parse_args()
    
    print(f"Initializing Crowd Detection System with YOLOv8{args.yolo_size} model...")
    
    # Initialize the application with the specified model size
    dashboard, alert_manager, hybrid_detector, threads = initialize_application(yolo_size=args.yolo_size)
    
    try:
        # Run the Flask app
        print("Starting dashboard web server...")
        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        # Handle application shutdown
        print("\nShutting down...")
        
        # Set running flag to False in the main module
        import RTCDM.main
        RTCDM.main.running = False
        
        # Wait for threads to finish
        for thread in threads:
            if thread and thread.is_alive():
                thread.join(timeout=1.0)
            
        print("Application shutdown complete") 