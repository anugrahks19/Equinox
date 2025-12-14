
from ultralytics import YOLO
import os
import glob
import zipfile

def run_inference(model_path, source_dir, output_dir):
    model = YOLO(model_path)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Loading model from {model_path}...")
    print(f"Processing images from {source_dir}...")
    
    # Run inference
    # Passing the directory directly is more memory efficient than a list of files
    # augment=True enables Test Time Augmentation (TTA) for higher accuracy
    results = model.predict(source=source_dir, save=False, conf=0.20, iou=0.45, stream=True, verbose=False, device=0, augment=True)
    
    file_count = 0
    
    print("Generating predictions...")
    for i, result in enumerate(results):
        if i % 100 == 0:
            print(f"Processed {i} images...")
        path = result.path
        basename = os.path.basename(path)
        name_no_ext = os.path.splitext(basename)[0]
        
        txt_path = os.path.join(output_dir, name_no_ext + '.txt')
        
        with open(txt_path, 'w') as f:
            for box in result.boxes:
                # box.xywhn -> x_center, y_center, width, height (normalized)
                # box.cls -> class id
                # box.conf -> confidence
                
                cls_id = int(box.cls[0].item())
                conf = box.conf[0].item()
                x_c, y_c, w, h = box.xywhn[0].tolist()
                
                # Format: class_id x_center y_center width height confidence
                line = f"{cls_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f} {conf:.6f}\n"
                f.write(line)
        file_count += 1
        if file_count % 100 == 0:
            print(f"Processed {file_count} images")

    print(f"Finished. Generated {file_count} prediction files in {output_dir}")
    
    # Create ZIP
    zip_name = "submission_predictions.zip"
    print(f"Zipping results to {zip_name}...")
    with zipfile.ZipFile(zip_name, 'w') as zipf:
        for txt_file in glob.glob(os.path.join(output_dir, '*.txt')):
            zipf.write(txt_file, os.path.basename(txt_file))
            
    print("Done.")

if __name__ == "__main__":
    # Adjust path to your best trained model
    # For initial testing, we might use a pretrained one or the one we just trained
    model_path = 'runs/train/yolov8l_military/weights/best.pt' 
    test_dir = r'D:\military_object_dataset\military_object_dataset\test\images'
    out_dir = 'predictions'
    
    # Check if model exists, if not warn user
    if not os.path.exists(model_path):
        print(f"WARNING: Model not found at {model_path}. Please train first.")
    else:
        run_inference(model_path, test_dir, out_dir)
