
from ultralytics import YOLO
import time
import os
import glob
import numpy as np

def check_efficiency():
    print("--- Efficiency Verification ---")
    
    # 1. Model Size
    # Updated to point to the Resumed 80-Epoch Model
    model_path = r'runs/train/yolov8l_military_resumed/weights/best.pt'
    
    if not os.path.exists(model_path):
        print(f"Custom model not found at {model_path}, using base 'yolov8l.pt' for benchmark example...")
        model_path = 'yolov8l.pt'
    else:
        print(f"Benchmarking custom model: {model_path}")

    # Load model
    model = YOLO(model_path)
    
    # Check file size
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"Model Size: {size_mb:.2f} MB")
    else:
        print("Model size: ~85.0 MB (Standard YOLOv8l)")

    # 2. Inference Speed
    test_weights = r'D:\military_object_dataset\military_object_dataset\test\images'
    images = glob.glob(os.path.join(test_weights, '*.jpg'))[:100] # Test on 100 images
    
    if not images:
        print("No test images found to benchmark!")
        return

    print("Running warmup on 10 images...")
    # Warmup
    model.predict(source=images[:10], verbose=False)
    
    print(f"Benchmarking inference on {len(images)} images...")
    start_time = time.time()
    
    # Run inference
    results = model.predict(source=images, verbose=False, device=0) 
    
    end_time = time.time()
    total_time = end_time - start_time
    
    avg_per_image = (total_time / len(images)) * 1000 # in ms
    fps = len(images) / total_time
    
    print("\n--- Efficiency Results ---")
    print(f"Total Time: {total_time:.2f} seconds")
    print(f"Inference Speed: {avg_per_image:.2f} ms per image")
    print(f"FPS (Frame Rate): {fps:.2f} FPS")
    print("--------------------------")
    
    # Judging Criteria Check
    if avg_per_image < 20: 
        print("Status: EXCELLENT efficiency (Real-time compatible)")
    elif avg_per_image < 100:
        print("Status: GOOD (Suitable for most applications)")
    else:
        print("Status: LOW (May need optimization for real-time)")

if __name__ == "__main__":
    check_efficiency()
