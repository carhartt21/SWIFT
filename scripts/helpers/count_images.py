import os
import csv
import argparse

# parent_folder = "path/to/parent/folder"
# output_csv = "file_counts.csv"
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Open CSV file in append mode
def count_files_in_subdirectories(parent_folder, output_csv):
    with open(output_csv, mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write the parent folder name as a separate row
        writer.writerow([f"Parent Folder: {parent_folder}"])
        
        # Walk through the directory tree
        for root, dirs, files in os.walk(parent_folder):
            if os.path.normpath(root) == os.path.normpath(parent_folder):
                continue  # skip the parent (root) directory
            file_count = sum(
                1 for f in files
                if os.path.splitext(f)[1].lower() in {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp', '.heic', '.heif'}
            )
            writer.writerow([os.path.basename(root), file_count])
            logger.info(f"Counted {file_count} files in {root}")
    return

def main():
    argument_parser = argparse.ArgumentParser(description="Count files in subdirectories and save to CSV.")
    argument_parser.add_argument("parent_folder", type=str, help="Path to the parent folder.")
    argument_parser.add_argument("output_csv", type=str, help="Path to the output CSV file.")
    args = argument_parser.parse_args()
    count_files_in_subdirectories(args.parent_folder, args.output_csv)
    print(f"File counts have been written to {args.output_csv}")
if __name__ == "__main__":
    main()
