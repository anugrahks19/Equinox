
from ultralytics import YOLO
import sys

def validate_model(model_path):
    print(f"Validating model: {model_path}")
    model = YOLO(model_path)
    
    # Run validation
    metrics = model.val(
        data=r'D:\military_object_dataset\military_object_dataset\military_dataset.yaml',
        split='val',
        batch=16,
        device=0,
        plots=True, # Saves confusion matrix and other plots
        verbose=True
    )
    
    print("\nValidation Results breakdown:")
    print(f"mAP@50: {metrics.box.map50}")
    print(f"mAP@50-95: {metrics.box.map}")
    
    # Use internal mapping to show per-class map if available
    # Ultralytics prints this by default
    
    print(f"Results saved to: {metrics.save_dir}")
    print("Check the 'confusion_matrix.png' in the save directory for report.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        model_p = sys.argv[1]
    else:
        model_p = 'runs/train/yolov8l_military/weights/best.pt'
        
    validate_model(model_p)
