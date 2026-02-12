import os
import glob

# Define the folders relative to the script's directory
folders = ['indicators', 'trades', 'data']

# Get the directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))

for folder in folders:
    folder_path = os.path.join(script_dir, folder)
    if os.path.exists(folder_path):
        # Find all CSV files in the folder
        csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
        for csv_file in csv_files:
            try:
                os.remove(csv_file)
                print(f"Deleted: {csv_file}")
            except OSError as e:
                print(f"Error deleting {csv_file}: {e}")
    else:
        print(f"Folder not found: {folder_path}")