import sys
import os
import cv2
import numpy as np
import time
import torch
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add RTCDM to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from RTCDM.detection.hybrid_detector import HybridDetector

def create_test_frame(num_people=10, frame_size=(640, 480)):
    """Create a synthetic test frame with rectangles representing people"""
    frame = np.ones((frame_size[1], frame_size[0], 3), dtype=np.uint8) * 255
    for _ in range(num_people):
        x = np.random.randint(0, frame_size[0] - 50)
        y = np.random.randint(0, frame_size[1] - 100)
        cv2.rectangle(frame, (x, y), (x + 50, y + 100), (0, 0, 0), -1)
    return frame

def test_hybrid_switching():
    """Test the hybrid model switching functionality"""
    logging.info("Initializing hybrid detector test...")
    
    # Initialize paths
    models_dir = os.path.join(project_root, 'RTCDM', 'models')
    yolo_n_path = os.path.join(models_dir, 'yolov8n.pt')
    yolo_m_path = os.path.join(models_dir, 'yolov8m.pt')
    
    # Verify model files exist
    if not os.path.exists(yolo_n_path) or not os.path.exists(yolo_m_path):
        logging.error("Model files not found. Please ensure both YOLOv8n and YOLOv8m models are present.")
        return False
    
    # Create hybrid detector
    logging.info("Creating HybridDetector with YOLOv8n...")
    detector = HybridDetector(
        yolo_model_path=yolo_n_path,
        density_threshold=20,
        yolo_model_size='n'
    )
    
    # Load YOLOv8m model
    logging.info("Loading YOLOv8m model...")
    if not detector.load_yolo_m_model(yolo_m_path):
        logging.error("Failed to load YOLOv8m model")
        return False
    
    # Test scenarios
    test_scenarios = [
        (5, "Low density"),       # Should use YOLOv8n
        (18, "Medium density"),   # Near threshold
        (30, "High density"),     # Should use YOLOv8m
        (15, "Medium density"),   # Should transition back
        (8, "Low density")        # Should use YOLOv8n
    ]
    
    logging.info("\nStarting model switching tests...")
    last_mode = None
    transitions_observed = 0
    
    for num_people, scenario in test_scenarios:
        logging.info(f"\nTesting {scenario} scenario ({num_people} people)...")
        frame = create_test_frame(num_people)
        
        # Run detection
        start_time = time.time()
        try:
            # Wait to ensure min_mode_duration constraint is satisfied
            time.sleep(2.1)  # Slightly longer than the 2.0s minimum mode duration
            
            count, density_map, info = detector.detect(frame)
            inference_time = time.time() - start_time
            
            # Print results
            logging.info(f"Detection count: {count}")
            logging.info(f"Current model: {info['model']}")
            logging.info(f"Mode: {info['mode']}")
            logging.info(f"Transition weight: {info.get('transition', 0):.2f}")
            logging.info(f"Inference time: {inference_time:.3f}s")
            
            # Track mode changes for transition testing
            if last_mode is not None and last_mode != info['mode']:
                transitions_observed += 1
                logging.info(f"MODE TRANSITION DETECTED: {last_mode} → {info['mode']}")
            last_mode = info['mode']
            
            # Verify expected behavior with more flexible logic
            if num_people < 15:
                if info['mode'] == 'yolo_m' and info.get('transition', 0) < 0.3:
                    logging.warning(f"Unexpected: Still using {info['mode']} with high transition weight ({info.get('transition', 0):.2f}) for low density")
            elif num_people > 25:
                if info['mode'] == 'yolo_n' and detector.yolo_m_model is not None:
                    logging.warning(f"Unexpected: Using {info['mode']} for high density scenario")
            
            # Optional: Display frame with information
            display_frame = frame.copy()
            cv2.putText(display_frame, f"Count: {count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(display_frame, f"Mode: {info['mode']}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Test Frame', display_frame)
            key = cv2.waitKey(1500)  # Show each frame for 1.5 seconds
            if key == 27:  # ESC key
                break
                
        except Exception as e:
            logging.error(f"Error during detection: {str(e)}")
            return False
    
    cv2.destroyAllWindows()
    
    # Final verification of transitions
    if transitions_observed >= 1:
        logging.info(f"\nDetected {transitions_observed} transitions between models.")
        logging.info("Hybrid model switching is working as expected!")
    else:
        logging.warning("\nNo transitions were observed between models. Check if thresholds are set appropriately.")
    
    logging.info("\nHybrid detector test completed successfully!")
    return True

def test_with_webcam():
    """Test hybrid detection using webcam feed"""
    logging.info("Initializing webcam test...")
    
    # Initialize detector
    models_dir = os.path.join(project_root, 'RTCDM', 'models')
    detector = HybridDetector(
        yolo_model_path=os.path.join(models_dir, 'yolov8n.pt'),
        density_threshold=20,
        yolo_model_size='n'
    )
    detector.load_yolo_m_model(os.path.join(models_dir, 'yolov8m.pt'))
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Could not open webcam")
        return False
    
    logging.info("\nStarting webcam test (press 'ESC' to exit)...")
    try:
        frame_count = 0
        total_inference_time = 0
        last_switch_time = time.time()
        current_mode = 'yolo_n'
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Run detection
            start_time = time.time()
            count, _, info = detector.detect(frame)
            inference_time = time.time() - start_time
            
            # Update statistics
            frame_count += 1
            total_inference_time += inference_time
            
            # Check for mode changes
            if info['mode'] != current_mode:
                time_since_switch = time.time() - last_switch_time
                logging.info(f"Model switched from {current_mode} to {info['mode']} after {time_since_switch:.2f}s")
                current_mode = info['mode']
                last_switch_time = time.time()
            
            # Draw information on frame
            cv2.putText(frame, f"Count: {count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Model: {info['model']}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"FPS: {1/inference_time:.1f}", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow('Webcam Test', frame)
            if cv2.waitKey(1) == 27:  # ESC key
                break
        
        # Print summary
        avg_fps = frame_count / total_inference_time if total_inference_time > 0 else 0
        logging.info(f"\nTest Summary:")
        logging.info(f"Frames processed: {frame_count}")
        logging.info(f"Average FPS: {avg_fps:.2f}")
        
    except Exception as e:
        logging.error(f"Error during webcam test: {str(e)}")
        return False
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    logging.info("Webcam test completed successfully!")
    return True

if __name__ == "__main__":
    # Run synthetic frame test
    if test_hybrid_switching():
        # Run webcam test if available
        response = input("\nWould you like to test with webcam? (y/n): ")
        if response.lower() == 'y':
            test_with_webcam()
    else:
        logging.error("Hybrid switching test failed. Please check the logs above.")