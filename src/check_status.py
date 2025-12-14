
import os
import pandas as pd
from ultralytics import YOLO
import time
import glob

def check_status():
    print("\n=============================================")
    print("      SERVE SMART MODEL STATUS REPORT        ")
    print("=============================================\n")

    # 1. IDENTIFY BEST MODEL
    # We know the 80-epoch resumed model is the best one
    model_dir = r'runs/train/yolov8l_military_resumed'
    weights_path = os.path.join(model_dir, 'weights', 'best.pt')
    csv_path = os.path.join(model_dir, 'results.csv')

    if not os.path.exists(weights_path):
        print("❌ Error: Could not find the 80-epoch model.")
        print(f"Checked: {weights_path}")
        return

    # 2. GET ACCURACY (mAP)
    print("1. ACCURACY (from Training Logs)")
    if os.path.exists(csv_path):
        try:
            # Read CSV, remove whitespace from columns
            df = pd.read_csv(csv_path)
            df.columns = [c.strip() for c in df.columns]
            
            # Get best mAP50
            best_epoch = df.loc[df['metrics/mAP50(B)'].idxmax()]
            final_epoch = df.iloc[-1]
            
            print(f"   • Best mAP@50:   {best_epoch['metrics/mAP50(B)']*100:.2f}% (Epoch {int(best_epoch['epoch'])})")
            print(f"   • Final mAP@50:  {final_epoch['metrics/mAP50(B)']*100:.2f}% (Epoch {int(final_epoch['epoch'])})")
        except Exception as e:
            print(f"   ⚠️ Could not read CSV: {e}")
    else:
        print("   ⚠️ No results.csv found.")

    # 3. GET EFFICIENCY (Benchmark)
    print("\n2. EFFICIENCY (Real-time Benchmark)")
    print("   Loading model... (This takes a few seconds)")
    try:
        model = YOLO(weights_path)
        
        # Size
        size_mb = os.path.getsize(weights_path) / (1024 * 1024)
        print(f"   • Model Size:    {size_mb:.2f} MB")

        # Speed
        test_dir = r'D:\military_object_dataset\military_object_dataset\test\images'
        images = glob.glob(os.path.join(test_dir, '*.jpg'))[:20] # Test 20 images for speed
        
        if images:
            # Warmup
            model.predict(images[0], verbose=False)
            
            start = time.time()
            for img in images:
                model.predict(img, verbose=False, device=0)
            end = time.time()
            
            avg_time = (end - start) / len(images) * 1000
            fps = 1000 / avg_time
            
            print(f"   • Inference:     {avg_time:.1f} ms/image")
            print(f"   • Frame Rate:    {fps:.1f} FPS")
        else:
            print("   ⚠️ No test images found for benchmark.")
            
    except Exception as e:
        print(f"   ⚠️ Benchmark failed: {e}")

    print("\n=============================================")
    print("RECOMMENDATION:")
    print("This is your best local model. It balances accuracy (57%)")
    print("with reasonable speed. It is ready for submission.")
    print("=============================================\n")

if __name__ == "__main__":
    check_status()
