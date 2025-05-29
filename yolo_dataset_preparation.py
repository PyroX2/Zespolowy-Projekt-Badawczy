import pandas as pd
import os
from pathlib import Path
from argparse import ArgumentParser
import utils


# Parse command line arguments
parser = ArgumentParser()
parser.add_argument('--csv-path', type=str, required=True, help='Path to the CSV file with metadata')
parser.add_argument('--embed-root', '-e', type=str, required=True, help='Root directory for the embedded dataset')
parser.add_argument('--root-dir', '-r', type=str, required=True, help='Root directory for saving labels and images')
parser.add_argument('--split', '-s', type=str, choices=['train', 'val', 'test'], default='train', help='Dataset split to process')
parser.add_argument('--cohort', type=int, default=1, help='Cohort number to filter the dataset')
args = parser.parse_args()


csv_path = args.csv_path
root_dir = Path(args.root_dir)
embed_dataset_root = Path(args.embed_root)
selected_cohort = args.cohort

# Raise an error if the cohort is not valid
if selected_cohort not in [1, 2, 3]:
    raise ValueError("Cohort must be either 1 or 2.")

split = args.split
images_output_dir = os.path.join(root_dir, 'images', split)
labels_output_dir = os.path.join(root_dir, 'labels', split)

# Create directories if they do not exist
if not root_dir.exists():
    os.makedirs(root_dir)
os.makedirs(images_output_dir, exist_ok=True)
os.makedirs(labels_output_dir, exist_ok=True)

def main():
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip() # Delete spaces in column names

    # Get only rows with valid ROIs and 2D images for the selected cohort
    df = df[
        (df['num_roi'] > 0) &
        (df['FinalImageType'] == '2D') &
        (df['cohort_num'] == selected_cohort)  
    ]

    for idx, row in df.iterrows():
        roi_str = row['ROI_coords']
        if not isinstance(roi_str, str) or roi_str.strip() == "":
            print(f"ROI for row {idx} is not a valid string, skipping.")
            continue

        rois = utils.parse_roi(roi_str)

        # Get image dimensions
        try:
            image_width = int(row['Columns'])
            image_height = int(row['Rows'])
        except:
            print(f"No image width and hight in row {idx}, skipping.")
            continue

        image_src_path = row['png_path'].strip()
        relative_path = image_src_path.split('/extracted-images/')[-1]
        image_src_path = os.path.join(embed_dataset_root, 'png_images/cohort_1/', relative_path)
        image_filename = row['png_filename'].strip() 

        if image_filename.endswith('.jpg'):
            txt_name = image_filename.replace('.jpg', '.txt')
        elif image_filename.endswith('.png'):
            txt_name = image_filename.replace('.png', '.txt')
        else:
            raise ValueError(f"Unsupported image format in filename: {image_filename}")

        output_label_path = os.path.join(labels_output_dir, txt_name)
        output_image_path = os.path.join(images_output_dir, image_filename)

        lines = [] # Lines for the output label file

        # obliczanie parametrów pod model YOLO
        for roi in rois:
            x_center, y_center, width, height = utils.roi_rescale(roi, image_height, image_width)
        
            # Add bounding box to list of all bounding boxes in YOLO format
            line = f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
            lines.append(line)

        with open(output_label_path, 'w') as f:
            f.write(''.join(lines))

        os.symlink(image_src_path, output_image_path)

    print(f"File saved to dataset root: {root_dir}")

if __name__ == "__main__":
    main()