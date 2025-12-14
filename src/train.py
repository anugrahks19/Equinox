
from ultralytics import YOLO
import os

# Fix for CUDA memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def train_model():
    # Define absolute path to dataset config
    data_yaml = r'D:\military_object_dataset\military_object_dataset\military_dataset.yaml'
    
    # Check if we can resume from a checkpoint (to save progress)
    checkpoint = r'D:\military_object_dataset\military_object_dataset\runs\train\yolov8l_military\weights\last.pt'
    if os.path.exists(checkpoint):
        print(f"Resuming from checkpoint: {checkpoint}")
        model = YOLO(checkpoint)
    else:
        # Initialize YOLOv8 Large model
        # Using 'l' (large) for better accuracy as primary goal is >95% mAP
        model = YOLO('yolov8l.pt') 

    print("Starting training...")
    
    # Training arguments
    results = model.train(
        data=data_yaml,
        epochs=30,           # Sufficient for fine-tuning
        imgsz=640,
        batch=4,             # Reduced to 4 to fix Fragmentation OOM
        save=True,
        # device=0,  <-- Removed to allow CPU fallback
        patience=10,         # Early stopping
        project='runs/train',
        name='yolov8l_military',
        exist_ok=True,
        pretrained=True,
        optimizer='auto',
        verbose=True,
        seed=42,
        cos_lr=True,         # Cosine learning rate scheduler
        
        # Augmentation hyperparameters (mild mixup/mosaic as we already did synthetic)
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,      # Additional built-in copy-paste
    )
    
    print("Training complete.")
    print(f"Best model saved at: {results.save_dir}")

if __name__ == "__main__":
    train_model()
