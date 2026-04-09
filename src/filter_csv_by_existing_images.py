"""
filter_csv_by_existing_images.py

Filter CSV để chỉ giữ các dòng có ảnh tồn tại trong thư mục local.
Tạo file CSV mới với chỉ những ảnh đã tải về.

Usage:
python src/filter_csv_by_existing_images.py --csv "GYM data/squat.../yolo_keypoints_cleaned_final.csv" --images_root "GYM data/squat.../squat" --output "GYM data/squat_local_filtered.csv"
"""

import argparse
import csv
from pathlib import Path
from tqdm import tqdm


def filter_csv_by_images(csv_path, images_root, output_csv):
    """
    Filter CSV to keep only rows where image exists in images_root
    """
    
    csv_path = Path(csv_path)
    images_root = Path(images_root)
    output_csv = Path(output_csv)
    
    # Read all images in local folder
    local_images = set()
    for img_path in images_root.glob("**/*.jpg"):
        local_images.add(img_path.name)
    
    print(f"Found {len(local_images)} local images")
    
    # Read CSV
    with open(csv_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames
    
    print(f"CSV has {len(rows)} total rows")
    
    # Filter rows
    filtered_rows = []
    skipped = 0
    
    for row in tqdm(rows, desc="Filtering"):
        img_name = row.get('image') or row.get('filename')
        if img_name in local_images:
            filtered_rows.append(row)
        else:
            skipped += 1
    
    print(f"\nFiltered: {len(filtered_rows)} rows kept, {skipped} rows skipped")
    
    # Save filtered CSV
    with open(output_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(filtered_rows)
    
    print(f"Saved filtered CSV: {output_csv}")
    print(f"\nSummary:")
    print(f"  Total images in local: {len(local_images)}")
    print(f"  Matched in CSV: {len(filtered_rows)}")
    print(f"  Coverage: {len(filtered_rows)/len(local_images)*100:.1f}%")
    
    return output_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True, help='Input CSV file')
    parser.add_argument('--images_root', required=True, help='Root folder containing local images')
    parser.add_argument('--output', required=True, help='Output filtered CSV file')
    
    args = parser.parse_args()
    
    filter_csv_by_images(args.csv, args.images_root, args.output)


if __name__ == '__main__':
    main()
