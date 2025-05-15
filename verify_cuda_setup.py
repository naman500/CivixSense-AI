import torch
import onnxruntime as ort
import numpy as np
import os

def verify_setup():
    print("=== CUDA Environment ===")
    print(f"CUDA_HOME: {os.environ.get('CUDA_HOME')}")
    print(f"CUDA_PATH: {os.environ.get('CUDA_PATH')}")
    
    print("\n=== PyTorch CUDA ===")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    print("\n=== ONNX Runtime ===")
    print(f"Version: {ort.__version__}")
    print(f"Available Providers: {ort.get_available_providers()}")
    
    # Test ONNX Runtime with CUDA
    try:
        # Create a simple model
        import onnx
        from onnx import helper, numpy_helper, shape_inference
        
        # Create input and output tensors
        X = helper.make_tensor_value_info('X', onnx.TensorProto.FLOAT, [1, 3, 224, 224])
        Y = helper.make_tensor_value_info('Y', onnx.TensorProto.FLOAT, [1, 3, 224, 224])
        Z = helper.make_tensor_value_info('Z', onnx.TensorProto.FLOAT, [1, 3, 224, 224])
        
        # Create a simple Add node
        add_node = helper.make_node(
            'Add',
            inputs=['X', 'Y'],
            outputs=['Z'],
            name='add_node'
        )
        
        # Create the graph
        graph = helper.make_graph(
            [add_node],
            'test-model',
            [X, Y],
            [Z]
        )
        
        # Create the model
        model = helper.make_model(graph)
        model = shape_inference.infer_shapes(model)
        onnx.save(model, 'test_model.onnx')
        
        # Test with CUDA
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        session = ort.InferenceSession('test_model.onnx', providers=providers)
        
        # Create test data
        x = np.random.randn(1, 3, 224, 224).astype(np.float32)
        y = np.random.randn(1, 3, 224, 224).astype(np.float32)
        
        # Run inference
        outputs = session.run(['Z'], {'X': x, 'Y': y})
        print("\n=== ONNX Runtime CUDA Test ===")
        print("Success! Model ran on GPU")
        print(f"Output shape: {outputs[0].shape}")
        
        # Clean up
        os.remove('test_model.onnx')
        
    except Exception as e:
        print("\n=== ONNX Runtime CUDA Test ===")
        print(f"Failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_setup()