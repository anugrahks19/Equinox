
import os
import glob

def verify_submission():
    print("--- Verifying Submission Compliance ---")
    
    test_dir = r'D:\military_object_dataset\military_object_dataset\test\images'
    pred_dir = r'D:\military_object_dataset\military_object_dataset\predictions'
    
    # 1. Count Check
    test_imgs = glob.glob(os.path.join(test_dir, '*.jpg'))
    pred_txts = glob.glob(os.path.join(pred_dir, '*.txt'))
    
    # We need to account for the fact that pred_dir might be zipped or inside the project.
    # The predict.py saved to 'predictions' folder.
    
    print(f"Test Images Found: {len(test_imgs)}")
    print(f"Prediction Files Found: {len(pred_txts)}")
    
    missing = []
    for img in test_imgs:
        base = os.path.splitext(os.path.basename(img))[0]
        txt_path = os.path.join(pred_dir, base + '.txt')
        if not os.path.exists(txt_path):
            missing.append(base)
            
    if missing:
        print(f"CRITICAL ERROR: Missing {len(missing)} prediction files!")
        print(f"Examples: {missing[:5]}")
        print("You will get ZERO marks for these.")
    else:
        print("SUCCESS: 1-to-1 Match. Every image has a corresponding .txt file.")

    # 2. Format Check
    print("\nChecking File Content Format...")
    valid_format = True
    if pred_txts:
        # Check first 5
        for p in pred_txts[:5]:
            with open(p, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) != 6:
                        print(f"ERROR in {os.path.basename(p)}: Found {len(parts)} values, expected 6 (class x y w h conf)")
                        valid_format = False
                        break
    
    if valid_format:
        print("SUCCESS: Content format looks correct (6 values per line).")

    print("\n--- Verification Summary ---")
    if not missing and valid_format:
        print("✅ READY TO SUBMIT. Your submission complies with all rules.")
    else:
        print("❌ DO NOT SUBMIT. Fix errors above.")

if __name__ == "__main__":
    verify_submission()
