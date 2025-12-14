
from ultralytics import YOLO
import os

# Anti-Fragmentation Fix (Important for your 8GB GPU)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def resume_training():
    print("Resuming Training for 50 MORE Epochs...")
    
    # Path to your existing 30-epoch model
    checkpoint = r'D:\military_object_dataset\military_object_dataset\runs\train\yolov8l_military\weights\best.pt'
    
    if not os.path.exists(checkpoint):
        print("Error: Could not find previous model to resume!")
        return

    model = YOLO(checkpoint)

    # We want +50 epochs.
    # If previous ended at 30, we set total to 80.
    total_epochs = 80 
    
    results = model.train(
        data=r'D:\military_object_dataset\military_object_dataset\military_dataset.yaml',
        epochs=total_epochs, 
        imgsz=640,
        
        # Optimization for Speed on your Laptop
        batch=4,           # Safe limit for 8GB VRAM (Large Model)
        workers=2,         # Reduced from 8 to 2 to prevent RAM OOM (cv2 error)
        cache=False,       # Disabled caching to save RAM
        
        save=True,
        project='runs/train',
        name='yolov8l_military_resumed',
        exist_ok=True,
        
        # Augmented for robustness
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1
    )
    print("Resumed Training Complete.")

if __name__ == "__main__":
    resume_training()
