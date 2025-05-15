import os
import torch
import time
import psutil
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import subprocess
import sys

class ModelManager:
    def __init__(self, models_dir=None):
        # Get the absolute path to the RTCDM directory
        if models_dir is None:
            # Get the absolute path to the RTCDM directory
            if __file__.endswith('model_utils.py'):
                script_dir = os.path.dirname(os.path.abspath(__file__))
                if os.path.basename(script_dir) == 'detection':
                    rtcdm_dir = os.path.dirname(script_dir)
                else:
                    rtcdm_dir = script_dir
            else:
                # If not running as a script, use the current file's location
                rtcdm_dir = os.path.dirname(os.path.abspath(__file__))
                while os.path.basename(rtcdm_dir) != 'RTCDM' and rtcdm_dir != os.path.dirname(rtcdm_dir):
                    rtcdm_dir = os.path.dirname(rtcdm_dir)
            
            models_dir = os.path.join(rtcdm_dir, 'models')
        
        self.models_dir = os.path.abspath(models_dir)
        os.makedirs(self.models_dir, exist_ok=True)
        self.performance_log = {}
        
        # Try to install pynvml if not present
        try:
            import pynvml
            self.pynvml = pynvml
            pynvml.nvmlInit()
        except ImportError:
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pynvml'])
                import pynvml
                self.pynvml = pynvml
                pynvml.nvmlInit()
            except Exception as e:
                print(f"Warning: Could not initialize NVIDIA Management Library: {e}")
                self.pynvml = None
        
    def setup_onnx_runtime(self):
        """Setup CUDA-enabled ONNX Runtime"""
        try:
            # First try to import onnxruntime-gpu
            try:
                import onnxruntime
                if 'CUDAExecutionProvider' in onnxruntime.get_available_providers():
                    print("CUDA-enabled ONNX Runtime already installed")
                    return True
            except ImportError:
                pass
            
            # Uninstall existing ONNX Runtime installations
            subprocess.check_call([sys.executable, '-m', 'pip', 'uninstall', '-y', 'onnxruntime', 'onnxruntime-gpu'])
            print("Removed existing ONNX Runtime installations")
            
            # Install CUDA-enabled ONNX Runtime
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'onnxruntime-gpu==1.16.3'])
            
            # Verify installation
            import onnxruntime
            if 'CUDAExecutionProvider' in onnxruntime.get_available_providers():
                print("Successfully installed CUDA-enabled ONNX Runtime")
                return True
            else:
                print("Warning: CUDA provider not available after installation")
                return False
        except Exception as e:
            print(f"Error setting up ONNX Runtime: {str(e)}")
            return False
    
    def convert_to_pytorch(self, model_name='yolov8n'):
        """Convert or download PyTorch model"""
        try:
            # Ensure model name has proper format
            if not model_name.startswith('yolov8'):
                model_name = f'yolov8{model_name}'
            
            # Download and save PyTorch model
            model = YOLO(f'{model_name}.pt')
            pt_path = os.path.join(self.models_dir, f'{model_name}.pt')
            model.save(pt_path)
            print(f"PyTorch model saved to {pt_path}")
            
            # Export to ONNX as backup
            onnx_path = os.path.join(self.models_dir, f'{model_name}.onnx')
            model.export(format='onnx', opset=12)
            print(f"ONNX model exported to {onnx_path}")
            
            return pt_path
        except Exception as e:
            print(f"Error converting model: {str(e)}")
            # Try direct download
            try:
                pt_path = os.path.join(self.models_dir, f'{model_name}.pt')
                if not os.path.exists(pt_path):
                    model = YOLO(f'yolov8{model_name}.pt')
                    model.save(pt_path)
                return pt_path
            except Exception as e2:
                print(f"Error in fallback download: {str(e2)}")
                return None
    
    def get_gpu_utilization(self):
        """Get GPU utilization safely"""
        if self.pynvml and torch.cuda.is_available():
            try:
                handle = self.pynvml.nvmlDeviceGetHandleByIndex(0)
                info = self.pynvml.nvmlDeviceGetUtilizationRates(handle)
                return info.gpu
            except Exception as e:
                print(f"Warning: Could not get GPU utilization: {e}")
        return 0
    
    def monitor_performance(self, model_type, device_type, batch_size=1):
        """Monitor model performance metrics"""
        metrics = {
            'timestamp': time.time(),
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'batch_size': batch_size
        }
        
        if device_type == 'cuda' and torch.cuda.is_available():
            metrics.update({
                'gpu_memory_allocated': torch.cuda.memory_allocated() / 1024**2,  # MB
                'gpu_memory_cached': torch.cuda.memory_reserved() / 1024**2,  # MB
                'gpu_utilization': self.get_gpu_utilization()
            })
        
        # Store metrics
        if model_type not in self.performance_log:
            self.performance_log[model_type] = []
        self.performance_log[model_type].append(metrics)
        
        return metrics

    # ... rest of the existing code ...