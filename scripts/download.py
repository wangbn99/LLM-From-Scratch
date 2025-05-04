"""
This script downloads and saves datasets from a specified URL.
The script provides functionality to download training, validation, and test datasets
from a remote server. It uses the `requests` library to fetch the files and `tqdm` 
to display a progress bar during the download process. The downloaded files are saved 
to user-specified directories.
You can download the training data by manual from the following URLs:
https://huggingface.co/datasets/monology/pile-uncopyrighted/tree/main/train

Functions:
    download_file(url, target_dir):
        Downloads a file from the given URL and saves it to the specified directory.
        Skips the download if the file already exists.
Main Functionality:
    - Parses command-line arguments to specify directories for training and validation data.
    - Creates the necessary directories if they do not exist.
    - Downloads the validation and test datasets.
    - Iterates through a list of training dataset URLs and downloads each file.
Command-line Arguments:
    --train_dir: Directory to save the training data. Default is "data/train".
    --val_dir: Directory to save the validation data. Default is "data/val".
Usage:
    Run the script from the command line:
        python download.py --train_dir <path_to_train_dir> --val_dir <path_to_val_dir>
"""
import argparse
import os
import requests
import tqdm

BASE_URL = "https://huggingface.co/datasets/monology/pile-uncopyrighted/resolve/main"
VAL_URL = f"{BASE_URL}/val.jsonl.zst"
TEST_URL = f"{BASE_URL}/test.jsonl.zst"
TRAIN_URLS = [f"{BASE_URL}/train/{i:02d}.jsonl.zst" for i in range(29)]

def download_file(url, target_dir):
    """
    Downloads a file from the specified URL and saves it to the target directory.

    If the file already exists in the target directory, the download is skipped.

    Args:
        url (str): The URL of the file to download.
        target_dir (str): The directory where the downloaded file will be saved.

    Raises:
        requests.exceptions.RequestException: If the HTTP request for the file fails.
        OSError: If there is an issue writing the file to the target directory.

    Notes:
        - The function uses a progress bar from the `tqdm` library to display download progress.
        - The file is downloaded in chunks of 1024 bytes to handle large files efficiently.
    """
    filename = os.path.join(target_dir, os.path.basename(url))
    if os.path.exists(filename):
        print(f"File {filename} already exists. Skipping download.")
        return
    print(f"Downloading {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))
    with open(filename, "wb") as f:
        for chunk in tqdm.tqdm(response.iter_content(chunk_size=1024), total=total_size // 1024, unit='KB'):
            f.write(chunk)
    print(f"Dataset downloaded and saved as {filename}.")   


def main():
    """
    Main function to handle the downloading and preprocessing of datasets.
    This function parses command-line arguments to specify directories for saving
    training and validation data. It ensures the directories exist, downloads the
    necessary files, and organizes them accordingly.
    Command-line Arguments:
        --train_dir (str): Directory to save the training data. Defaults to "data/train".
        --val_dir (str): Directory to save the validation data. Defaults to "data/val".
    Functionality:
        - Creates the specified directories if they do not already exist.
        - Downloads validation and test datasets to the validation directory.
        - Downloads training datasets to the training directory.
    Raises:
        Any exceptions raised during file downloading or directory creation.
    """
    parser = argparse.ArgumentParser(description="Download and preprocess the dataset.")
    parser.add_argument("--train_dir", type=str, default="data/train", help="Directory to save the training data.")
    parser.add_argument("--val_dir", type=str, default="data/val", help="Directory to save the validation data.")   
    # parser.add_argument("--test_dir", type=str, default="data/test", help="Directory to save the test data.")   
    
    args = parser.parse_args()
    os.makedirs(args.train_dir, exist_ok=True)
    os.makedirs(args.val_dir, exist_ok=True) 
    # os.makedirs(args.test_dir, exist_ok=True)
    
    download_file(VAL_URL, args.val_dir)
    download_file(TEST_URL, args.val_dir)
    for url in TRAIN_URLS:
        download_file(url, args.train_dir)
        break
    print("All files downloaded successfully.")


if __name__ == "__main__":
    main()
