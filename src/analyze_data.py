
import os
from collections import Counter
import glob
import yaml
import matplotlib.pyplot as plt

def analyze_dataset(root_dir):
    # Read yaml
    with open(os.path.join(root_dir, 'military_dataset.yaml'), 'r') as f:
        data_config = yaml.safe_load(f)
    
    classes = data_config['names']
    print(f"Classes found: {classes}")
    
    splits = ['train', 'val']
    stats = {}
    
    for split in splits:
        label_dir = os.path.join(root_dir, split, 'labels')
        label_files = glob.glob(os.path.join(label_dir, '*.txt'))
        
        cnt = Counter()
        for lf in label_files:
            with open(lf, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    try:
                        cls_id = int(line.strip().split()[0])
                        cnt[cls_id] += 1
                    except:
                        pass
        stats[split] = cnt
        print(f"\nStats for {split} ({len(label_files)} files):")
        for cls_id, count in cnt.items():
            name = classes.get(cls_id, str(cls_id))
            print(f"  {name}: {count}")
            
    return stats

if __name__ == "__main__":
    analyze_dataset(r'D:\military_object_dataset\military_object_dataset')
