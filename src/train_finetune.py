
from ultralytics import YOLO
import os

# Fix for CUDA memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def train_emergency():
    print("--- EMERGENCY HYPER-TUNING (The 'Final Push') ---")
    print("Strategy: High Res (800p) + Low LR + Aggressive Augmentation")
    
    # 1. Load the Last Checkpoint (Crash Recovery)
    model_path = r'runs/train/yolov8l_emergency_finetune/weights/last.pt'
    
    if not os.path.exists(model_path):
        print(f"❌ Error: Checkpoint not found at {model_path}")
        # Fallback to best.pt if last.pt is missing, but start new run
        model_path = r'runs/train/yolov8l_military_resumed/weights/best.pt'
        print(f"Fallback: Starting fresh from {model_path}")
        model = YOLO(model_path)
        resume_mode = False
    else:
        print(f"Resuming from crash: {model_path}")
        model = YOLO(model_path)
        resume_mode = True

    # 2. Resume Training
    if resume_mode:
        results = model.train(resume=True)
    else:
        results = model.train(
            data=r'D:\military_object_dataset\military_object_dataset\military_dataset.yaml',
            epochs=50,
            patience=15,
            imgsz=800,
            lr0=0.0001,
            lrf=0.01,
            optimizer='AdamW',
            batch=2,
            workers=0,           # CRITICAL FIX: 0 workers prevents RAM explosion (OpenCV crash).
            cache=False,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            scale=0.5,
            mosaic=1.0,
            mixup=0.15,
            copy_paste=0.15,
            project='runs/train',
            name='yolov8l_emergency_finetune',
            exist_ok=True
        )
    
    print("EMERGENCY TRAINING COMPLETE.")
    print(f"Best model saved at: {results.save_dir}")

if __name__ == "__main__":
    train_emergency()
