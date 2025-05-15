import os
import requests
import sys
import argparse
from tqdm import tqdm
import torch
from ultralytics import YOLO

def download_file(url, destination):
    """Download a file with progress bar"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes
        total_size = int(response.headers.get('content-length', 0))
        
        with open(destination, 'wb') as f, tqdm(
            desc=os.path.basename(destination),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                bar.update(size)
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {str(e)}")
        return False
    return True

def download_yolo_model(destination, model_size='n'):
    """
    Download YOLOv8 model using ultralytics
    
    Args:
        destination: Path to save the model
        model_size: Size of the model ('n', 's', 'm', 'l', 'x')
        
    Returns:
        bool: Success status
    """
    try:
        # Map of model sizes
        size_map = {
            'n': 'yolov8n',  # Nano (smallest, fastest)
            's': 'yolov8s',  # Small
            'm': 'yolov8m',  # Medium
            'l': 'yolov8l',  # Large
            'x': 'yolov8x'   # Extra large (largest, most accurate)
        }
        
        # Make sure the size is valid
        if model_size not in size_map:
            print(f"Invalid model size: {model_size}. Using 'n' (nano) instead.")
            model_size = 'n'
        
        model_name = size_map[model_size]
        print(f"Downloading {model_name} model...")
        
        # Create the model instance with explicit task=detect to avoid warning
        model = YOLO(model_name, task='detect')
        
        # Export to ONNX format
        print(f"Exporting {model_name} to ONNX format...")
        
        # Export with optimizations for inference
        model.export(format='onnx', imgsz=640, optimize=True)
        
        # Move file to destination
        temp_path = f'{model_name}.onnx'
        if os.path.exists(temp_path):
            os.rename(temp_path, destination)
            print(f"Successfully exported {model_name} to ONNX format at {destination}")
            return True
        else:
            print(f"Error: Expected ONNX file {temp_path} was not created")
            return False
            
    except Exception as e:
        print(f"Error exporting YOLO model to ONNX: {str(e)}")
        print("Trying to install onnxruntime...")
        try:
            import subprocess, sys
            subprocess.check_call([sys.executable, "-m", "pip", "install", "onnxruntime"])
            print("onnxruntime installed, retrying export...")
            
            # Try again with onnxruntime installed
            model = YOLO(model_name, task='detect')
            model.export(format='onnx', imgsz=640, optimize=True)
            temp_path = f'{model_name}.onnx'
            
            if os.path.exists(temp_path):
                os.rename(temp_path, destination)
                print(f"Successfully exported {model_name} to ONNX format at {destination}")
                return True
            else:
                print(f"Error: Expected ONNX file {temp_path} was not created")
                return False
                
        except Exception as install_error:
            print(f"Error installing onnxruntime or exporting model: {str(install_error)}")
        return False

def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Download models for Real-Time Crowd Detection System')
    parser.add_argument('--skip-yolo', action='store_true', help='Skip downloading YOLO models')
    parser.add_argument('--force', action='store_true', help='Force download even if models already exist')
    
    args = parser.parse_args()
    
    # Get the absolute path to the models directory
    if __file__.endswith('download_models.py'):
        # If called directly
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if os.path.basename(script_dir) == 'RTCDM':
            # If script is in RTCDM root
            models_dir = os.path.abspath(os.path.join(script_dir, 'models'))
        else:
            # If script is in utils or another subdirectory
            models_dir = os.path.abspath(os.path.join(os.path.dirname(script_dir), 'models'))
    else:
        # Fallback to absolute path
        rtcdm_dir = os.path.dirname(os.path.abspath(__file__))
        while os.path.basename(rtcdm_dir) != 'RTCDM' and rtcdm_dir != os.path.dirname(rtcdm_dir):
            rtcdm_dir = os.path.dirname(rtcdm_dir)
        models_dir = os.path.abspath(os.path.join(rtcdm_dir, 'models'))
    
    print(f"Using models directory: {models_dir}")
    
    # Create models directory if it doesn't exist
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f"Created models directory at {models_dir}")
    else:
        print(f"Models directory already exists at {models_dir}")
    
    # Model paths for both nano and medium models
    yolo_nano_path = os.path.join(models_dir, "yolov8n.onnx")
    yolo_medium_path = os.path.join(models_dir, "yolov8m.onnx")
    
    # Download models
    success = True
    
    # Download YOLO models if needed
    if not args.skip_yolo:
        # Download nano model
        if args.force or not os.path.exists(yolo_nano_path):
            print("Downloading YOLOv8n model...")
            if not download_yolo_model(yolo_nano_path, 'n'):
                success = False
                print("Failed to download YOLOv8n model")
        else:
            print(f"YOLOv8n model already exists at {yolo_nano_path}")
    
        # Download medium model
        if args.force or not os.path.exists(yolo_medium_path):
            print("Downloading YOLOv8m model...")
            if not download_yolo_model(yolo_medium_path, 'm'):
                success = False
                print("Failed to download YOLOv8m model")
            else:
                print("Successfully downloaded YOLOv8m model")
        else:
            print(f"YOLOv8m model already exists at {yolo_medium_path}")
    else:
        print("Skipping YOLO model downloads as requested")
    
    if not success:
        print("\nSome models failed to download. Please try again or download them manually.")
        sys.exit(1)
    
    print("\nAll models downloaded successfully!")
    
    # Print usage instructions
    print("\nUsage Instructions:")
    print(f"1. YOLOv8n ONNX model saved to: {yolo_nano_path}")
    print(f"2. YOLOv8m ONNX model saved to: {yolo_medium_path}")
    print("\nTo use the system with these models:")
    print("   python RTCDM/main.py")
    print("\nTo force re-download of models:")
    print("   python -m RTCDM.download_models --force")

if __name__ == "__main__":
    main() 