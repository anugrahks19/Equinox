
from ultralytics import YOLO
import os

# Fix for CUDA memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def train_ultimate():
    print("Initializing ULTIMATE TRAINING for MAX ACCURACY (>95%)...")
    
    data_yaml = r'D:\military_object_dataset\military_object_dataset\military_dataset.yaml'
    
    # STRATEGY CHANGE: Switch to YOLOv8x (Extra Large)
    # This is the most accurate model available.
    # Your 4090 can handle it.
    model_name = 'yolov8x.pt' 
    
    print(f"Loading {model_name} (The Beast)...")
    model = YOLO(model_name)

    # Training Configuration for WINNING
    # 300 Epochs is standard for competition-grade accuracy.
    # Patience 50 allows it to learn through plateaus.
    results = model.train(
        data=data_yaml,
        epochs=300,          # The detailed study session
        imgsz=640,
        batch=4,             # Safe batch size for 4090 Laptop (16GB VRAM)
        save=True,
        patience=50,         # Don't give up easily
        project='runs/train',
        name='yolov8x_ultimate',
        exist_ok=True,
        pretrained=True,
        
        # Aggressive Augmentation to force generalization
        mosaic=1.0,
        mixup=0.5,           # Mix images harder
        copy_paste=0.5,      # Use our synthetic strategy more
        hsv_h=0.015,         # Color jitter to handle lighting
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,        # Rotation
        translate=0.1,
        scale=0.5,           # Scale variation (Robustness)
        fliplr=0.5,
        
        # Efficiency
        device=0,
        workers=4
    )
    print("ULTIMATE TRAINING COMPLETE.")
    print(f"Best model saved at: {results.save_dir}")

if __name__ == "__main__":
    train_ultimate()
