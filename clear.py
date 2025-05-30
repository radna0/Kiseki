import argparse
import os
import shutil


def main():
    parser = argparse.ArgumentParser(
        description="Delete everything in a folder except 'line_raw' and 'ref_raw'."
    )
    parser.add_argument(
        "--folder-path",
        required=True,
        help="Path to the folder you want to clean",
    )
    args = parser.parse_args()
    folder_path = args.folder_path

    if not os.path.isdir(folder_path):
        print(f"Error: '{folder_path}' is not a valid directory.")
        return

    # get a list of all files and folders in the folder
    items = os.listdir(folder_path)

    # loop through the list and delete all files and folders except for "line_raw" and "ref_raw"
    for name in items:
        full_path = os.path.join(folder_path, name)
        if name not in ("line_raw", "ref_raw"):
            if os.path.isfile(full_path):
                os.remove(full_path)
                print("Deleted file:", name)
            elif os.path.isdir(full_path):
                shutil.rmtree(full_path)
                print("Deleted folder:", name)
        else:
            print("Skipping:", name)


if __name__ == "__main__":
    main()
