
import zipfile
import os
import glob

def create_submission():
    # 1. Create Predictions ZIP
    print("Zipping ENSEMBLE predictions...")
    pred_zip_name = "predictions.zip"
    pred_dir = r"D:\military_object_dataset\military_object_dataset\predictions_ensemble" # ERA Update: Use Ensemble
    
    print(f"Creating {pred_zip_name}...")
    with zipfile.ZipFile(pred_zip_name, 'w', zipfile.ZIP_DEFLATED) as pz:
        if os.path.exists(pred_dir):
            files = glob.glob(os.path.join(pred_dir, "*.txt"))
            for txt_file in files:
                pz.write(txt_file, os.path.basename(txt_file))
            print(f"Packed {len(files)} prediction files into {pred_zip_name}")
        else:
            print("WARNING: No predictions found!")

    # 2. Create Final Submission Archive
    final_zip_name = "Final_Submission_Serve_Smart.zip"
    print(f"Creating final archive: {final_zip_name}...")
    
    with zipfile.ZipFile(final_zip_name, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Add Code
        print("Adding code...")
        for file in glob.glob("src/*.py"):
            zf.write(file)
            
        # Add Documentation
        if os.path.exists("REPORT.md"):
            zf.write("REPORT.md")
        if os.path.exists("README.md"):
            zf.write("README.md")
        
        # Add Run Guide (Requested by User)
        if os.path.exists("README_RUN_GUIDE.md"):
            print("Adding README_RUN_GUIDE.md...")
            zf.write("README_RUN_GUIDE.md")
            
        # Add Requirements (Critical for new PDF/Streamlit features)
        if os.path.exists("requirements.txt"):
            print("Adding requirements.txt...")
            zf.write("requirements.txt")

        # Add PPT Content
        if os.path.exists("PPT_CONTENT.md"):
            print("Adding PPT_CONTENT.md...")
            zf.write("PPT_CONTENT.md")
            
        # Add Project PDF (Critical)
        if os.path.exists("equinox.pdf"):
            print("Adding equinox.pdf...")
            zf.write("equinox.pdf")

        # Add the Predictions Zip
        if os.path.exists(pred_zip_name):
            print("Adding predictions.zip...")
            zf.write(pred_zip_name)
            
        # 4. Add Model Weights (Crucial for Reproducibility)
        # Check root first, then runs directory
        weights_dir = r"runs/train/yolov8l_military_resumed/weights"
        openvino_dir = os.path.join(weights_dir, "best_int8_openvino_model")
        
        # Gold Model
        if os.path.exists("best.pt"):
            print("Adding Gold Model (best.pt) from root...")
            zf.write("best.pt")
        elif os.path.exists(os.path.join(weights_dir, "best.pt")):
             print(f"Adding Gold Model from {weights_dir}...")
             zf.write(os.path.join(weights_dir, "best.pt"), "best.pt")
             
        # Nuclear Model
        if os.path.exists("nuclear.pt"):
            print("Adding Nuclear Model (nuclear.pt) from root...")
            zf.write("nuclear.pt")
            
        # OpenVINO Model
        if os.path.exists(openvino_dir):
            print(f"Adding OpenVINO Model: {openvino_dir}...")
            for root, dirs, files in os.walk(openvino_dir):
                for file in files:
                    abs_path = os.path.join(root, file)
                    # Rel path inside zip: best_int8_openvino_model/file
                    rel_path = os.path.relpath(abs_path, weights_dir)
                    zf.write(abs_path, rel_path)
            
    print(f"\nSUCCESS. Created {final_zip_name}")
    print("Contains: Code, Report, README, Run Guide, Predictions, and MODEL WEIGHTS.")

if __name__ == "__main__":
    create_submission()
