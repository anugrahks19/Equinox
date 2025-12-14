
import cv2
import os
import glob
import matplotlib.pyplot as plt
import random

def show_examples():
    # 1. Show a "Synthetic" image (Trench/Civilian)
    syn_images = glob.glob(r'D:\military_object_dataset\military_object_dataset\train\images\syn_*.jpg')
    
    if syn_images:
        syn_path = random.choice(syn_images)
        syn_img = cv2.imread(syn_path)
        # Draw label? The label file is same name .txt
        
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title(f"Synthetic Image (Created to fix imbalance)\n{os.path.basename(syn_path)}")
        plt.imshow(cv2.cvtColor(syn_img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
    else:
        print("No synthetic images found! Did you run augment_rare.py?")

    # 2. Show a "Real" image
    real_images = glob.glob(r'D:\military_object_dataset\military_object_dataset\train\images\*.jpg')
    # Filter out syn
    real_images = [x for x in real_images if 'syn_' not in x]
    
    if real_images:
        real_path = random.choice(real_images)
        real_img = cv2.imread(real_path)
        
        plt.subplot(1, 2, 2)
        plt.title(f"Original Real Image\n{os.path.basename(real_path)}")
        plt.imshow(cv2.cvtColor(real_img, cv2.COLOR_BGR2RGB))
        plt.axis('off')

    output_path = r'D:\military_object_dataset\military_object_dataset\explanation_example.png'
    plt.savefig(output_path)
    print(f"Saved explanation image to {output_path}")

if __name__ == "__main__":
    show_examples()
