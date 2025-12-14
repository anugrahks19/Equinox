
from ultralytics import YOLO
import os

# Fix for CUDA memory
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def train_nuclear():
    print("--- OPERATION: NUCLEAR OPTION (YOLOv8-ExtraLarge) ---")
    print("Strategy: Max Parameters (68M). Max Capacity. Max Accuracy.")
    
    # 1. Load Model (Extension Logic)
    last_ckpt = r'runs/train/yolov8x_nuclear/weights/last.pt'
    
    if os.path.exists(last_ckpt):
        print(f"🔄 EXTENDING Nuclear Training from: {last_ckpt}")
        print("Training for another 30 epochs (Total will be 60)...")
        model = YOLO(last_ckpt)
        # We start a NEW run, but initialized with the trained weights.
        # This acts as Epochs 31-60.
        resume_mode = False 
    else:
        print("🚀 STARTING Fresh Nuclear Training")
        model = YOLO('yolov8x.pt')
        resume_mode = False

    # 2. Train
    if resume_mode:
        results = model.train(resume=True)
    else:
        results = model.train(
            data=r'D:\military_object_dataset\military_object_dataset\military_dataset.yaml',
        
        epochs=30,           # Short but powerful run
        patience=0,
        
        imgsz=640,           # Standard res to fit in memory (X is huge)
        
        # Hyperparameters for Convergence
        optimizer='AdamW', 
        lr0=0.001,           # Standard start
        batch=2,             # X needs VRAM. Keep batch low.
        workers=0,           # Safe mode
        
        # Robustness
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.15,     # Use our synthetic data
        
        project='runs/train',
        name='yolov8x_nuclear',
        exist_ok=True
    )
    
    print("NUCLEAR TRAINING COMPLETE.")
    print(f"Best model saved at: {results.save_dir}")

if __name__ == "__main__":
    train_nuclear()
