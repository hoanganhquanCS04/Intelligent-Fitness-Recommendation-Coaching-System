import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

def analyze_dataset(data_root):
    """
    Phân tích dataset: count, size, quality
    """
    
    data_root = Path(data_root)
    all_images = list(data_root.glob("**/*.jpg"))
    
    print(f"🔍 Found {len(all_images)} images")
    
    # Group by video folder
    videos = defaultdict(list)
    for img in all_images:
        video_id = img.parent.name
        videos[video_id].append(img)
    
    print(f"📁 {len(videos)} video folders")
    
    # Analyze image properties
    stats = {
        'widths': [],
        'heights': [],
        'sizes_kb': [],
        'mean_brightness': []
    }
    
    corrupted = []
    
    for img_path in tqdm(all_images[:500], desc="Analyzing samples"):  # Sample 500 images
        try:
            # File size
            size_kb = img_path.stat().st_size / 1024
            stats['sizes_kb'].append(size_kb)
            
            # Image properties
            img = cv2.imread(str(img_path))
            if img is None:
                corrupted.append(str(img_path))
                continue
            
            h, w = img.shape[:2]
            stats['widths'].append(w)
            stats['heights'].append(h)
            
            # Brightness
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            brightness = gray.mean()
            stats['mean_brightness'].append(brightness)
            
        except Exception as e:
            corrupted.append(str(img_path))
    
    # Print statistics
    print("\n" + "="*50)
    print("📊 DATASET STATISTICS")
    print("="*50)
    
    print(f"\nTotal images: {len(all_images)}")
    print(f"Video folders: {len(videos)}")
    print(f"Corrupted images: {len(corrupted)}")
    
    if stats['widths']:
        print(f"\n🖼️  Image Dimensions:")
        print(f"  Width:  min={min(stats['widths'])}, max={max(stats['widths'])}, avg={np.mean(stats['widths']):.0f}")
        print(f"  Height: min={min(stats['heights'])}, max={max(stats['heights'])}, avg={np.mean(stats['heights']):.0f}")
    
    if stats['sizes_kb']:
        print(f"\n💾 File Sizes:")
        print(f"  min={min(stats['sizes_kb']):.1f} KB, max={max(stats['sizes_kb']):.1f} KB, avg={np.mean(stats['sizes_kb']):.1f} KB")
    
    if stats['mean_brightness']:
        print(f"\n💡 Brightness:")
        print(f"  min={min(stats['mean_brightness']):.1f}, max={max(stats['mean_brightness']):.1f}, avg={np.mean(stats['mean_brightness']):.1f}")
    
    # Print top 10 videos by frame count
    print(f"\n🎥 Top 10 videos by frame count:")
    top_videos = sorted(videos.items(), key=lambda x: len(x[1]), reverse=True)[:10]
    for video_id, frames in top_videos:
        print(f"  {video_id}: {len(frames)} frames")
    
    return stats, videos, corrupted


def visualize_samples(data_root, n_samples=9):
    """
    Hiển thị random samples từ dataset
    """
    
    data_root = Path(data_root)
    all_images = list(data_root.glob("**/*.jpg"))
    
    # Random sample
    samples = np.random.choice(all_images, min(n_samples, len(all_images)), replace=False)
    
    # Plot
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.ravel()
    
    for i, img_path in enumerate(samples):
        img = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        axes[i].imshow(img_rgb)
        axes[i].set_title(f"{img_path.parent.name}\n{img_path.name}", fontsize=8)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig("GYM data/processed/dataset_samples.png", dpi=150)
    print(f"\n✅ Samples saved → GYM data/processed/dataset_samples.png")
    plt.close()


def main():
    data_root = "GYM data/squat-20260402T134558Z-3-001/squat"
    
    # Create output dir
    os.makedirs("GYM data/processed", exist_ok=True)
    
    # Analyze
    stats, videos, corrupted = analyze_dataset(data_root)
    
    # Visualize
    visualize_samples(data_root, n_samples=9)
    
    # Save corrupted list
    if corrupted:
        with open("GYM data/processed/corrupted_images.txt", 'w') as f:
            f.write("\n".join(corrupted))
        print(f"\n⚠️  Corrupted images list saved → GYM data/processed/corrupted_images.txt")
    
    print("\n✅ Exploration complete!")


if __name__ == "__main__":
    main()