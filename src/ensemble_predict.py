
import os
import glob
import torch
import numpy as np
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from torchvision.ops import nms

def run_ensemble():
    print("--- ⚔️ OPERATION: LEGION (Model Ensemble) ⚔️ ---")
    print("Merging 'Gold' (v8l) and 'Nuclear' (v8x) for maximum recall.")

    # 1. Setup Models
    path_gold = r'runs/train/yolov8l_gold/weights/best.pt'
    path_nuclear = r'runs/train/yolov8x_nuclear/weights/best.pt'
    
    source_dir = r'D:\military_object_dataset\military_object_dataset\test\images'
    output_dir = r'D:\military_object_dataset\military_object_dataset\predictions_ensemble'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Loading Gold Model: {path_gold}")
    model_gold = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=path_gold,
        confidence_threshold=0.10, # High Recall
        device="cuda:0"
    )
    
    # Check if Nuclear exists, otherwise running single model
    if os.path.exists(path_nuclear):
        print(f"Loading Nuclear Model: {path_nuclear}")
        model_nuclear = AutoDetectionModel.from_pretrained(
            model_type='yolov8',
            model_path=path_nuclear,
            confidence_threshold=0.10,
            device="cuda:0"
        )
        use_ensemble = True
    else:
        print("⚠️ Nuclear model not found. Running Gold only.")
        use_ensemble = False

    images = glob.glob(os.path.join(source_dir, '*.jpg'))
    print(f"Processing {len(images)} images...")

    for i, img_path in enumerate(images):
        # 1. Get Predictions from Gold
        result_gold = get_sliced_prediction(
            img_path, model_gold,
            slice_height=640, slice_width=640,
            overlap_height_ratio=0.6, overlap_width_ratio=0.6,
            perform_standard_pred=True, verbose=0
        )
        
        preds = result_gold.object_prediction_list
        img_w, img_h = result_gold.image_width, result_gold.image_height

        # 2. Get Predictions from Nuclear (if available)
        if use_ensemble:
            result_nuclear = get_sliced_prediction(
                img_path, model_nuclear,
                slice_height=640, slice_width=640,
                overlap_height_ratio=0.6, overlap_width_ratio=0.6,
                perform_standard_pred=True, verbose=0
            )
            preds.extend(result_nuclear.object_prediction_list)

        # 3. Prepare for NMS
        # Convert to tensor: [x1, y1, x2, y2, score, class]
        boxes = []
        scores = []
        labels = []
        
        for p in preds:
            bbox = p.bbox
            boxes.append([bbox.minx, bbox.miny, bbox.maxx, bbox.maxy])
            scores.append(p.score.value)
            labels.append(p.category.id)
            
        if len(boxes) == 0:
            # Write empty file
            basename = os.path.basename(img_path)
            txt_name = os.path.splitext(basename)[0] + ".txt"
            with open(os.path.join(output_dir, txt_name), 'w') as f:
                pass
            continue

        boxes_t = torch.tensor(boxes, dtype=torch.float32)
        scores_t = torch.tensor(scores, dtype=torch.float32)
        labels_t = torch.tensor(labels, dtype=torch.int64)
        
        # 4. Apply Per-Class NMS
        final_boxes = []
        final_scores = []
        final_labels = []
        
        unique_labels = labels_t.unique()
        for c in unique_labels:
            mask = labels_t == c
            c_boxes = boxes_t[mask]
            c_scores = scores_t[mask]
            
            # NMS Operation
            # standard IoU threshold for ensemble is often 0.6-0.7 to merge overlapping
            # boxes from different models.
            keep = nms(c_boxes, c_scores, iou_threshold=0.6)
            
            final_boxes.extend(c_boxes[keep])
            final_scores.extend(c_scores[keep])
            final_labels.extend([c.item()] * len(keep))

        # 5. Write to TXT
        basename = os.path.basename(img_path)
        txt_name = os.path.splitext(basename)[0] + ".txt"
        save_path = os.path.join(output_dir, txt_name)
        
        with open(save_path, 'w') as f:
            for box, score, label in zip(final_boxes, final_scores, final_labels):
                # Unpack and Normalize
                x1, y1, x2, y2 = box.tolist()
                
                w = x2 - x1
                h = y2 - y1
                x_c = (x1 + w / 2) / img_w
                y_c = (y1 + h / 2) / img_h
                w_n = w / img_w
                h_n = h / img_h
                
                f.write(f"{label} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f} {score:.6f}\n")

        if i % 10 == 0:
            print(f"Ensembled {i}/{len(images)}...")

    print(f"Ensemble Complete. Results in {output_dir}")

if __name__ == "__main__":
    run_ensemble()
