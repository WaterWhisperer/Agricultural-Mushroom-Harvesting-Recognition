from ultralytics import YOLO
import sys
import argparse
import os

def convert_model(pt_path, onnx_output_path):
    """
    Converts a YOLO .pt model to .onnx format.

    Args:
        pt_path (str): The file path of the .pt model.
        onnx_output_path (str): The output path for the .onnx model.
    """
    if not pt_path.endswith('.pt'):
        print("Error: Model path must end with .pt")
        return

    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(onnx_output_path), exist_ok=True)
        
        # Load the YOLO model from the .pt file
        model = YOLO(pt_path)
        
        # Export the model to ONNX format
        # The 'opset' can be adjusted if needed, 12 is a common choice for compatibility.
        # imgsz sets the input image size.
        model.export(format='onnx', imgsz=640, opset=12)
        
        # Get the default exported ONNX path
        default_onnx_path = pt_path.replace('.pt', '.onnx')
        
        # Move the exported file to the desired location
        if os.path.exists(default_onnx_path):
            os.rename(default_onnx_path, onnx_output_path)
            print(f"Successfully converted {pt_path} to {onnx_output_path}")
        else:
            print(f"Error: Could not find exported model at {default_onnx_path}")
    
    except Exception as e:
        print(f"An error occurred during conversion: {e}")

if __name__ == '__main__':
    # --- Instructions ---
    # Run this script from your terminal with the model path as an argument.
    # Example 1: python export_onnx.py --pt_path weights/yolov8n.pt --onnx_output_path weights/yolov8n.onnx
    # Example 2: python export_onnx.py --pt_path weights/yolo11n.pt --onnx_output_path weights/yolo11n.onnx
    
    parser = argparse.ArgumentParser(description="Export YOLO model to ONNX format")
    parser.add_argument('--pt_path', required=True, help='Path to the .pt model')
    parser.add_argument('--onnx_output_path', required=True, help='Path to the output .onnx model')
    args = parser.parse_args()
    
    # Convert the model to ONNX format
    convert_model(args.pt_path, args.onnx_output_path)