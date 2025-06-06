from ultralytics import YOLO
import sys

def convert_model(pt_path):
    """
    Converts a YOLO .pt model to .onnx format.

    Args:
        pt_path (str): The file path of the .pt model.
    """
    if not pt_path.endswith('.pt'):
        print("Error: Model path must end with .pt")
        return

    try:
        # Load the YOLO model from the .pt file
        model = YOLO(pt_path)
        
        # Export the model to ONNX format
        # The 'opset' can be adjusted if needed, 12 is a common choice for compatibility.
        # imgsz sets the input image size.
        model.export(format='onnx', imgsz=640, opset=12)
        
        print(f"Successfully converted {pt_path} to {pt_path.replace('.pt', '.onnx')}")
    
    except Exception as e:
        print(f"An error occurred during conversion: {e}")

if __name__ == '__main__':
    # --- Instructions ---
    # Run this script from your terminal with the model path as an argument.
    # Example 1: python export_onnx.py weights/yolov8n.pt
    # Example 2: python export_onnx.py weights/yolo11n.pt
    
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        convert_model(model_path)
    else:
        print("Usage: python export_onnx.py <path_to_your_model.pt>")
        # Default conversion for demonstration
        print("Running default conversion for 'weights/mushroom_v8n.pt'")
        convert_model('weights/mushroom_v8n.pt')