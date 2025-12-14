
import os
import random
import cv2
import numpy as np
import glob

def load_yolo_label(label_path, img_w, img_h):
    objects = []
    if not os.path.exists(label_path):
        return objects
    
    with open(label_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        parts = line.strip().split()
        cls_id = int(parts[0])
        if len(parts) > 5:
            # Handle segmentation/polygon: class x1 y1 x2 y2 ...
            # Extract points
            coords = list(map(float, parts[1:]))
            xs = coords[0::2]
            ys = coords[1::2]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            
            w = max_x - min_x
            h = max_y - min_y
            x_c = min_x + w/2
            y_c = min_y + h/2
        else:
            # Standard YOLO: class x_c y_c w h
            x_c, y_c, w, h = map(float, parts[1:])
        
        # Convert to pixel coords
        x1 = int((x_c - w/2) * img_w)
        y1 = int((y_c - h/2) * img_h)
        x2 = int((x_c + w/2) * img_w)
        y2 = int((y_c + h/2) * img_h)
        
        objects.append({
            'class_id': cls_id,
            'bbox': [x1, y1, x2, y2],
            'normalized': [x_c, y_c, w, h]
        })
    return objects

def save_start_yolo_label(objects, file_path, img_w, img_h):
    lines = []
    for obj in objects:
        cls_id = obj['class_id']
        x1, y1, x2, y2 = obj['bbox']
        
        # Convert back to normalized
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h
        x_c = (x1 + x2) / 2 / img_w
        y_c = (y1 + y2) / 2 / img_h
        
        # Clip to ensure valid range
        w = min(max(w, 0.001), 1.0)
        h = min(max(h, 0.001), 1.0)
        x_c = min(max(x_c, 0.0), 1.0)
        y_c = min(max(y_c, 0.0), 1.0)

        lines.append(f"{cls_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")
        
    with open(file_path, 'w') as f:
        f.write('\n'.join(lines))

def augment_rare_classes(root_dir):
    train_img_dir = os.path.join(root_dir, 'train', 'images')
    train_lbl_dir = os.path.join(root_dir, 'train', 'labels')
    
    # Sources
    rare_files = {
        9: ['011224', '011232'], # Trench
        5: ['003961', '003986', '003992', '004002', '004170', '004340', '004643', '004834', '005012', '005139', '005201'] # Civilian (sample)
    }
    
    # Store extracted patches
    patches = {9: [], 5: []}
    
    print("Extracting rare object patches...")
    for cls_id, basenames in rare_files.items():
        for base in basenames:
            img_path = os.path.join(train_img_dir, base + '.jpg')
            lbl_path = os.path.join(train_lbl_dir, base + '.txt')
            
            if not os.path.exists(img_path):
                continue
                
            img = cv2.imread(img_path)
            if img is None: continue
            h_img, w_img = img.shape[:2]
            
            objs = load_yolo_label(lbl_path, w_img, h_img)
            
            for obj in objs:
                if obj['class_id'] == cls_id:
                    x1, y1, x2, y2 = obj['bbox']
                    # Ensure coords within bounds
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w_img, x2), min(h_img, y2)
                    
                    if x2 > x1 and y2 > y1:
                        patch = img[y1:y2, x1:x2].copy()
                        patches[cls_id].append(patch)
    
    print(f"Extracted {len(patches[9])} trench patches and {len(patches[5])} civilian patches.")
    
    # Background candidates (images WITHOUT rare classes)
    all_imgs = glob.glob(os.path.join(train_img_dir, '*.jpg'))
    bg_candidates = []
    # Just take a random sample of 2000 images to serve as backgrounds
    random.shuffle(all_imgs)
    sampled_bgs = all_imgs[:2000]
    
    target_count = 500 # Generate 500 new images per class
    
    for cls_id in [9, 5]:
        if not patches[cls_id]:
            print(f"No patches for class {cls_id}!")
            continue
            
        print(f"Generating {target_count} synthetic images for class {cls_id}...")
        
        for i in range(target_count):
            if i % 50 == 0:
                print(f"  Generated {i}/{target_count}")
            # Select random background
            bg_path = random.choice(sampled_bgs)
            bg_img = cv2.imread(bg_path)
            if bg_img is None: continue
            
            h_bg, w_bg = bg_img.shape[:2]
            bg_base = os.path.splitext(os.path.basename(bg_path))[0]
            
            # Load existing labels of background
            bg_lbl_path = os.path.join(train_lbl_dir, bg_base + '.txt')
            current_objects = load_yolo_label(bg_lbl_path, w_bg, h_bg)
            
            # Select random patch
            patch = random.choice(patches[cls_id])
            h_p, w_p = patch.shape[:2]
            
            # Random scale (0.5 to 1.5)
            scale = random.uniform(0.5, 1.5)
            new_w = int(w_p * scale)
            new_h = int(h_p * scale)
            
            # Resize patch
            try:
                patch_resized = cv2.resize(patch, (new_w, new_h))
            except:
                continue

            # Find valid location (try 50 times to find non-overlapping spot, or just paste anyway)
            # For simplicity and "occlusion" robustness, we allow some overlap, but prefer empty space.
            # We'll just pick a random spot that keeps the object fully inside the image.
            
            if new_w >= w_bg or new_h >= h_bg:
                # Patch too big, resize to 20% of bg
                 scale_factor = min(w_bg*0.2/w_p, h_bg*0.2/h_p)
                 new_w = int(w_p * scale_factor)
                 new_h = int(h_p * scale_factor)
                 patch_resized = cv2.resize(patch, (new_w, new_h))

            # Random position
            start_x = random.randint(0, w_bg - new_w)
            start_y = random.randint(0, h_bg - new_h)
            
            # Paste (using simple replacement, could use seamless clone but simple is mostly enough for YOLO)
            # Optional: Gaussian blur edges
            
            bg_img[start_y:start_y+new_h, start_x:start_x+new_w] = patch_resized
            
            # Add new object
            current_objects.append({
                'class_id': cls_id,
                'bbox': [start_x, start_y, start_x+new_w, start_y+new_h],
                'normalized': [] # Recomputed in save
            })
            
            # Save new file
            new_base = f"syn_{cls_id}_{i}_{bg_base}"
            new_img_path = os.path.join(train_img_dir, new_base + '.jpg')
            new_lbl_path = os.path.join(train_lbl_dir, new_base + '.txt')
            
            cv2.imwrite(new_img_path, bg_img)
            save_start_yolo_label(current_objects, new_lbl_path, w_bg, h_bg)
            
    print("Augmentation complete.")

if __name__ == "__main__":
    augment_rare_classes(r'D:\military_object_dataset\military_object_dataset')
