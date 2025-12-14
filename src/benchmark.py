
from ultralytics import YOLO
import time
import torch
import numpy as np

def benchmark():
    print("--- ⚡ EQUINOX EFFICIENCY BENCHMARK ⚡ ---")
    
    # Load Model
    model_path = r'D:\military_object_dataset\military_object_dataset\final_submission\best.pt'
    print(f"Loading: {model_path}")
    model = YOLO(model_path)
    
    # Dummy Input (1080p equivalent)
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    
    # Warmup
    print("Warming up GPU/CPU...")
    for _ in range(10):
        model(img, verbose=False)
        
    # Benchmark
    print("Benchmarking 50 frames...")
    start = time.time()
    for _ in range(50):
        model(img, verbose=False)
    end = time.time()
    
    duration = end - start
    fps = 50 / duration
    
    print(f"\nRESULT: {fps:.2f} FPS")
    if fps > 10:
        print("✅ Status: REAL-TIME CAPABLE")
    else:
        print("⚠️ Status: BATCH PROCESSING ONLY")

if __name__ == "__main__":
    benchmark()
