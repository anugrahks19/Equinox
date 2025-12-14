
from ultralytics import YOLO
import os

# Fix for CUDA memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def train_hd_sniper():
    print("Initializing HD SNIPER MODE (1280p Resolution)...")
    print("Strategy: Smaller Model (Medium) + Sharper Eyes (HD)")
    
    data_yaml = r'D:\military_object_dataset\military_object_dataset\military_dataset.yaml'
    
    # We switch to YOLOv8m (Medium)
    # Why? 'Large' model @ 1280p requires ~16GB VRAM.
    # 'Medium' model @ 1280p fits in your 8GB.
    model = YOLO('yolov8m.pt') 

    results = model.train(
        data=data_yaml,
        epochs=50,           # 50 Epochs of HD training is worth 200 of SD
        imgsz=1280,          # THE KEY TO 95%: HD Resolution
        
        # Memory Optimization for 8GB VRAM
        batch=2,             # Very small batch for massive images
        workers=2,
        cache=False,         # Save RAM
        
        save=True,
        project='runs/train',
        name='yolov8m_hd_sniper',
        exist_ok=True,
        pretrained=True,
        
        # Stronger Augmentation for Robustness
        mosaic=1.0,
        mixup=0.2,
        copy_paste=0.2,
        degrees=15.0,        # More rotation
        scale=0.6,           # More scale variation
    )
    print("HD SNIPER TRAINING COMPLETE.")
    print(f"Best model saved at: {results.save_dir}")

if __name__ == "__main__":
    train_hd_sniper()
