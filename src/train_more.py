
from ultralytics import YOLO
import os

# Fix for CUDA memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def train_more():
    data_yaml = r'D:\military_object_dataset\military_object_dataset\military_dataset.yaml'
    
    # Load the BEST model we have so far
    checkpoint = r'D:\military_object_dataset\military_object_dataset\runs\train\yolov8l_military\weights\best.pt'
    
    if os.path.exists(checkpoint):
        print(f"Loading best model to continue training: {checkpoint}")
        model = YOLO(checkpoint)
    else:
        print("No previous model found! Converting to fresh training...")
        model = YOLO('yolov8l.pt')

    print("Starting Extended Training (Target: 100 Epochs)...")
    
    # Train for another 70 epochs (Total 100)
    results = model.train(
        data=data_yaml,
        epochs=100,          # Increased target
        imgsz=640,
        batch=4,             # Keep safe batch size
        save=True,
        patience=20,         # More patience
        project='runs/train',
        name='yolov8l_military_extended',
        exist_ok=True,
        lr0=0.001,           # Lower learning rate for fine-tuning
        cos_lr=True
    )
    print("Extended training complete.")

if __name__ == "__main__":
    train_more()
