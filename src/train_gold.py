
from ultralytics import YOLO
import os

# Fix for CUDA memory
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def train_gold():
    print("--- OPERATION: GOLD STANDARD (The Accuracy Recovery) ---")
    print("Strategy: Disable ALL noise/augmentation. Train on PURE data.")
    
    # 1. Load the Epoch 130 Model (The Battle-Hardened One)
    # We use the one we just put in final_submission
    model_path = r'D:\military_object_dataset\military_object_dataset\final_submission\best.pt'
    
    if not os.path.exists(model_path):
        print(f"❌ Error: Model not found at {model_path}")
        return

    print(f"Loading weights from: {model_path}")
    model = YOLO(model_path)

    # 2. The "Cool Down" Run
    # We turn off all the "hard" stuff. No mosaic, no mixup, no weather.
    # Just pure, pixel-perfect image training.
    results = model.train(
        data=r'D:\military_object_dataset\military_object_dataset\military_dataset.yaml',
        
        epochs=20,           # Short sprint
        patience=0,          # Don't stop
        
        # KEY CHANGE 1: Back to Native Resolution
        imgsz=640,           # The model was confused by 800. Let's ground it.
        
        # KEY CHANGE 2: Zero Noise (The "Polishing" phase)
        mosaic=0.0,          # OFF
        mixup=0.0,           # OFF
        copy_paste=0.0,      # OFF
        hsv_h=0.0,           # OFF
        hsv_s=0.0,           # OFF
        hsv_v=0.0,           # OFF
        scale=0.0,           # OFF (No zooming, just standard view)
        fliplr=0.5,          # keep simple flip
        
        # KEY CHANGE 3: Micro Learning Rate
        lr0=0.00001,         # 1e-5 (Tiny steps to find the global minimum)
        lrf=0.01,
        
        optimizer='AdamW',
        batch=4,             # Increase batch size (since 640p uses less RAM)
        workers=0,           # Keep safe
        
        project='runs/train',
        name='yolov8l_gold',
        exist_ok=True
    )
    
    print("GOLD TRAINING COMPLETE.")
    print(f"Best model saved at: {results.save_dir}")

if __name__ == "__main__":
    train_gold()
