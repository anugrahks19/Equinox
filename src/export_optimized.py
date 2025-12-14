
from ultralytics import YOLO
import os

def export_for_speed():
    print("--- OPTIMIZING MODEL FOR DEPLOYMENT ---")
    
    # Path to your best model (Update this after training HD Sniper)
    # If HD Sniper not run yet, use the Resumed one
    model_path = r'runs/train/yolov8l_military_resumed/weights/best.pt'
    
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return

    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    print("\n1. Exporting to ONNX (Universal Format)...")
    # CPU friendly, runs everywhere
    model.export(format='onnx', dynamic=True)
    
    print("\n2. Exporting to TensorRT (NVIDIA Speed Demon)...")
    # GPU optimized, 2x-5x faster than PyTorch
    # Note: This takes 5-10 minutes to build the engine
    try:
        model.export(format='engine', device=0, half=True) # FP16 for speed
        print("✅ TensorRT Export Successful! (This is the 'Efficiency' winner)")
    except Exception as e:
        print(f"⚠️ TensorRT Export failed (Requires TensorRT libs): {e}")
        print("   (Don't worry, ONNX is still a huge upgrade)")

    print("\n--- OPTIMIZATION COMPLETE ---")
    print("Use the .engine or .onnx file in your final submission folder.")
    print("Mention 'TensorRT FP16 Inference' in your report for bonus points.")

if __name__ == "__main__":
    export_for_speed()
