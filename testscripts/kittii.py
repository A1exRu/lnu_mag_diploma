import os
import shutil
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

KITTI_BASE_URL = "http://www.cvlibs.net/datasets/kitti/raw_data_downloader.php?file="
KITTI_DATASET_DIR = "../kitti_dataset"
KITTI_FILES = [
    "data_object_image_2.zip",
    "data_object_label_2.zip",
    "data_object_calib.zip",
]


def download_file(url, dest_path):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    with open(dest_path, 'wb') as f, tqdm(
            desc=dest_path,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            f.write(data)
            bar.update(len(data))


def unzip_file(zip_path, extract_dir):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print(f"Розпаковано: {zip_path} до {extract_dir}")


def download_and_organize_kitti(dataset_dir):
    os.makedirs(dataset_dir, exist_ok=True)
    temp_dir = os.path.join(dataset_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)

    for file_name in KITTI_FILES:
        url = KITTI_BASE_URL + file_name
        dest_path = os.path.join(temp_dir, file_name)
        print(f"Завантаження {file_name}...")
        download_file(url, dest_path)
        unzip_file(dest_path, temp_dir)

    final_dir = os.path.join(dataset_dir, "training")
    os.makedirs(final_dir, exist_ok=True)
    for folder in ["image_2", "label_2", "calib"]:
        src_folder = os.path.join(temp_dir, f"training/{folder}")
        dest_folder = os.path.join(final_dir, folder)
        if os.path.exists(src_folder):
            shutil.move(src_folder, dest_folder)
            print(f"Переміщено {src_folder} до {dest_folder}")

    shutil.rmtree(temp_dir)
    print(f"Датасет KITTI організовано в {dataset_dir}")


def convert_kitti_to_yolo(label_dir, output_dir, class_mapping):
    os.makedirs(output_dir, exist_ok=True)
    for label_file in Path(label_dir).glob("*.txt"):
        with open(label_file, 'r') as f:
            lines = f.readlines()
        yolo_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 15:
                continue
            class_name = parts[0]
            if class_name not in class_mapping:
                continue
            class_id = class_mapping[class_name]

            left, top, right, bottom = map(float, parts[4:8])

            img_width, img_height = 1242, 375
            x_center = (left + right) / 2 / img_width
            y_center = (top + bottom) / 2 / img_height
            width = (right - left) / img_width
            height = (bottom - top) / img_height
            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

        output_file = os.path.join(output_dir, label_file.name)
        with open(output_file, 'w') as f:
            f.writelines(yolo_lines)
    print(f"Конвертовано мітки до YOLO формату в {output_dir}")


if __name__ == "__main__":
    download_and_organize_kitti(KITTI_DATASET_DIR)

    class_mapping = {
        "Pedestrian": 0,
        "Car": 1,
        "Cyclist": 2,

    }
    label_dir = os.path.join(KITTI_DATASET_DIR, "training", "label_2")
    yolo_label_dir = os.path.join(KITTI_DATASET_DIR, "training", "yolo_labels")
    convert_kitti_to_yolo(label_dir, yolo_label_dir, class_mapping)
