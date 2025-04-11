import os
import shutil

# get the path of the folder
folder_path = "dataset/test/manu"

# get a list of all files and folders in the folder
files_and_folders = os.listdir(folder_path)

# loop through the list and delete all files and folders except for "line_raw" and "ref_raw"
for file_or_folder in files_and_folders:
    full_path = os.path.join(folder_path, file_or_folder)
    if file_or_folder not in ("line_raw", "ref_raw"):
        if os.path.isfile(full_path):
            os.remove(full_path)
            print("Deleted file:", file_or_folder)
        elif os.path.isdir(full_path):
            shutil.rmtree(full_path)
            print("Deleted folder:", file_or_folder)
    else:
        print("Skipping:", file_or_folder)
