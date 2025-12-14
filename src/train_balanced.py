
from ultralytics import YOLO
import os

# Fix for CUDA memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def train_balanced():
    print("Initializing BALANCED TRAINING (Medium Model, High Speed)...")
    
    data_yaml = r'D:\military_object_dataset\military_object_dataset\military_dataset.yaml'
    
    # STRATEGY CHANGE: Switch to YOLOv8m (Medium)
    # Why?
    # 1. It is 2x faster than 'Large'.
    # 2. It requires less memory, so we can increase Batch Size (Speed up).
    # 3. Accuracy difference is minimal (often < 1%).
    model_name = 'yolov8m.pt' 
    
    print(f"Loading {model_name} (The Fast & Accurate One)...")
    model = YOLO(model_name)

    # Training Configuration
    # Goal: >95% in ~8-10 hours
    results = model.train(
        data=data_yaml,
        epochs=100,          # 100 is the sweet spot for high accuracy
        imgsz=640,
        batch=16,            # Medium model fits 16 items easily in 8GB VRAM
        save=True,
        patience=20,
        project='runs/train',
        name='yolov8m_balanced',
        exist_ok=True,
        pretrained=True,
        
        # Augmentation (Standard Strong)
        mosaic=1.0,
        mixup=0.2,           
        copy_paste=0.2,      
        
        # Hardware
        device=0,
        workers=4            # Ensure generic workers for data loading speed
    )
    print("BALANCED TRAINING COMPLETE.")
    print(f"Best model saved at: {results.save_dir}")

if __name__ == "__main__":
    train_balanced()
