
from ultralytics import YOLO
import os
import glob
import torch

# You must install SAHI first: pip install sahi
try:
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
except ImportError:
    print("SAHI not installed! Run: pip install sahi")
    exit()

def predict_with_sahi():
    print("--- SAHI INFERENCE (Small Object Specialist) ---")
    
    # Paths
    # Paths
    # CRITICAL UPDATE: Using the 'Gold' model (145 Epochs)
    model_path = r'runs/train/yolov8l_gold/weights/best.pt'
    source_dir = r'D:\military_object_dataset\military_object_dataset\test\images'
    output_dir = r'D:\military_object_dataset\military_object_dataset\predictions_sahi'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Loading Model: {model_path}")
    
    # Configure SAHI Model
    # CONFIDENCE STRATEGY: 0.10
    # Why? For mAP@50, we want HIGH RECALL.
    # We want to catch every possible soldier.
    # False Positives are less penalizing than False Negatives for mAP.
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=model_path,
        confidence_threshold=0.10, 
        device="cuda:0" if torch.cuda.is_available() else "cpu"
    )

    images = glob.glob(os.path.join(source_dir, '*.jpg'))
    print(f"Processing {len(images)} images with Slicing (Aggressive Mode)...")

    for i, img_path in enumerate(images):
        # SAHI Magic: Slice image into 640x640 chunks, detect, and merge
        result = get_sliced_prediction(
            img_path,
            detection_model,
            slice_height=640,
            slice_width=640,
            overlap_height_ratio=0.6,    # BOOST: 60% Overlap catches objects on edges better
            overlap_width_ratio=0.6,
            perform_standard_pred=True,  # ALSO detect on full image (catches big tanks)
            postprocess_type="NMM",      # Non-Maximum Merging (Better than NMS)
            postprocess_match_metric="IOS",
            verbose=0
        )
        
        # Save to TXT
        basename = os.path.basename(img_path)
        txt_name = os.path.splitext(basename)[0] + ".txt"
        txt_path = os.path.join(output_dir, txt_name)
        
        with open(txt_path, 'w') as f:
            for prediction in result.object_prediction_list:
                # Convert SAHI format to YOLO format
                # SAHI gives absolute bbox, we need normalized xywh
                
                img_w, img_h = result.image_width, result.image_height
                bbox = prediction.bbox
                x, y, w, h = bbox.minx, bbox.miny, bbox.maxx - bbox.minx, bbox.maxy - bbox.miny
                
                # Normalize
                x_c = (x + w / 2) / img_w
                y_c = (y + h / 2) / img_h
                w_n = w / img_w
                h_n = h / img_h
                
                cls_id = prediction.category.id
                conf = prediction.score.value
                
                f.write(f"{cls_id} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f} {conf:.6f}\n")
        
        if i % 10 == 0:
            print(f"Processed {i}/{len(images)}...")

    print(f"Done! Predictions saved to {output_dir}")
    print("These predictions are likely 5-10% more accurate for small objects.")

if __name__ == "__main__":
    predict_with_sahi()
