
from ultralytics import YOLO
import sys

def dry_run():
    print("Running dry run (sanity check) of the training pipeline...")
    # Use Nano model for speed test
    model = YOLO('yolov8n.pt')
    
    # Train for 1 epoch on very small image size just to check data loading and memory
    try:
        model.train(
            data=r'D:\military_object_dataset\military_object_dataset\military_dataset.yaml',
            epochs=1,
            imgsz=320,
            batch=4,
            project='runs/test',
            name='dry_run',
            verbose=True
        )
        print("\nSUCCESS: Data pipeline and Training setup are valid!")
        print("You can now securely run the full training.")
    except Exception as e:
        print(f"\nFAILURE: Found error in setup: {e}")

if __name__ == "__main__":
    dry_run()
