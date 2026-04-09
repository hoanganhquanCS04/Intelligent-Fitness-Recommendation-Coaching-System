"""
convert_csv_to_yolo_pose.py

Convert a CSV containing keypoint columns (image,label,x0,y0,x1,y1,...,x17,y17)
into YOLOv8-pose .txt label files with normalized bbox + keypoints.

Usage (PowerShell):
python src\convert_csv_to_yolo_pose.py --csv "GYM data\squat-annotations.csv" --images_root "GYM data\squat-20260402T134558Z-3-001\squat" --out_labels "GYM data\cv_dataset\labels" --class_map "squat:0"

Notes:
- Assumes one row per person/image. If multiple persons per image, CSV must contain multiple rows with same image name.
- If visibility is not provided, visible=2 for present keypoints, 0 for missing.
"""

import argparse
import csv
import json
from pathlib import Path
import cv2


def parse_class_map(s: str):
    # Format like 'squat:0,deadlift:1'
    m = {}
    for part in s.split(','):
        if not part.strip():
            continue
        k, v = part.split(':')
        m[k.strip()] = int(v.strip())
    return m


def read_csv_rows(csv_path):
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def convert_row_to_label(row, images_root: Path, out_labels_dir: Path, class_map: dict):
    # Expected columns: image,label,x0,y0,x1,y1,...,x17,y17
    img_name = row.get('image') or row.get('filename') or row.get('file')
    if img_name is None:
        raise ValueError('CSV must contain an "image" column')
    img_path = images_root.joinpath(img_name)
    if not img_path.exists():
        # try without subfolders
        possible = list(images_root.glob(f"**/{img_name}"))
        if possible:
            img_path = possible[0]
        else:
            print(f"[WARN] Image not found: {img_name}")
            return

    # Read image to get size
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[WARN] Failed to open image: {img_path}")
        return
    h, w = img.shape[:2]

    label_name = row.get('label', '').strip()
    class_id = 0
    if label_name:
        class_id = class_map.get(label_name, 0)

    # Collect keypoints
    kps = []  # list of tuples (x,y,vis)
    for i in range(17):
        xk = row.get(f'x{i}')
        yk = row.get(f'y{i}')
        # Some CSVs may use x0,y0.. x17,y17 or x1..x17; handle missing
        if xk is None and yk is None:
            # try prefixed with kp
            xk = row.get(f'kp{i}_x') or row.get(f'kp{i}_X')
            yk = row.get(f'kp{i}_y') or row.get(f'kp{i}_Y')
        try:
            if xk is None or xk == '' or yk is None or yk == '':
                kps.append((None, None, 0))
            else:
                xf = float(xk)
                yf = float(yk)
                # assume coordinates are pixel values; convert to normalized below
                kps.append((xf, yf, 2))
        except Exception as e:
            kps.append((None, None, 0))

    # If all kps missing -> skip
    if all(k[0] is None for k in kps):
        print(f"[WARN] No keypoints for image: {img_name}")
        return

    # Compute bounding box from visible keypoints
    xs = [kp[0] for kp in kps if kp[0] is not None]
    ys = [kp[1] for kp in kps if kp[1] is not None]
    if not xs or not ys:
        print(f"[WARN] No valid kp coords for bbox: {img_name}")
        return

    xmin = min(xs)
    xmax = max(xs)
    ymin = min(ys)
    ymax = max(ys)

    # add small padding 5% of width/height
    pad_x = (xmax - xmin) * 0.05
    pad_y = (ymax - ymin) * 0.05
    xmin = max(0, xmin - pad_x)
    ymin = max(0, ymin - pad_y)
    xmax = min(w, xmax + pad_x)
    ymax = min(h, ymax + pad_y)

    box_w = xmax - xmin
    box_h = ymax - ymin
    x_center = xmin + box_w / 2
    y_center = ymin + box_h / 2

    # Normalize bboxes and keypoints to [0,1]
    x_center_n = x_center / w
    y_center_n = y_center / h
    box_w_n = box_w / w
    box_h_n = box_h / h

    kp_parts = []
    for (xk, yk, vis) in kps:
        if xk is None or yk is None:
            kp_parts.extend(["0", "0", "0"])  # normalized 0,0 and vis=0
        else:
            kp_parts.extend([f"{(xk / w):.6f}", f"{(yk / h):.6f}", str(int(vis))])

    # Compose label line
    line_parts = [str(class_id), f"{x_center_n:.6f}", f"{y_center_n:.6f}", f"{box_w_n:.6f}", f"{box_h_n:.6f}"] + kp_parts
    line = ' '.join(line_parts)

    # Save to labels dir with same stem
    out_label_path = out_labels_dir.joinpath(img_path.stem + '.txt')
    with open(out_label_path, 'w', encoding='utf-8') as f:
        f.write(line + '\n')

    print(f"Saved label: {out_label_path.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True, help='Path to CSV file')
    parser.add_argument('--images_root', required=True, help='Root folder containing images')
    parser.add_argument('--out_labels', required=True, help='Output folder to save .txt labels')
    parser.add_argument('--class_map', default='squat:0', help='Comma-separated class map, e.g. squat:0')

    args = parser.parse_args()

    csv_path = Path(args.csv)
    images_root = Path(args.images_root)
    out_labels_dir = Path(args.out_labels)
    ensure_dir(out_labels_dir)

    class_map = parse_class_map(args.class_map)

    rows = read_csv_rows(csv_path)
    print(f"Loaded {len(rows)} rows from CSV")

    for row in rows:
        convert_row_to_label(row, images_root, out_labels_dir, class_map)

    print('\nConversion complete!')


if __name__ == '__main__':
    main()
