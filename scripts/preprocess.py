

import argparse
import json
import os
import h5py
import numpy as np
import tiktoken
from tqdm import tqdm
import zstandard as zstd


def main() -> None:
    """
    Main function for preprocessing the dataset.
    This script processes training and validation datasets by reading data from specified directories,
    performing necessary preprocessing, and saving the processed data to output files. The script
    also ensures that the required directories exist and creates them if they do not.
    Command-line Arguments:
        --train_dir (str): Directory containing the training data. Default is "data/train".
        --val_dir (str): Directory containing the validation data. Default is "data/val".
        --output_train_file (str): Path to save the processed training data. Default is "data/train/pile_train.h5".
        --output_val_file (str): Path to save the processed validation data. Default is "data/val/pile_val.h5".
    Returns:
        None
    Notes:
        - The script checks if the specified directories for training and validation data exist.
          If they do not, it prints an error message and exits.
        - The output directories are created automatically if they do not exist.
        - The `data_processing` function is called to handle the actual preprocessing of the data.
    """
    print("Preprocessing data...")
    parser = argparse.ArgumentParser(description="Preprocess the dataset.")
    parser.add_argument("--train_dir", type=str, default="data/train", help="Directory to save the training data.")
    parser.add_argument("--val_dir", type=str, default="data/val", help="Directory to save the validation data.")
    parser.add_argument("--test_dir", type=str, default="data/val", help="Directory to save the test data.")
    parser.add_argument("--output_train_file", type=str, default="data/train/pile_train.h5", help="Output file for training data.")
    parser.add_argument("--output_val_file", type=str, default="data/val/pile_val.h5", help="Output file for validation data.") 
    parser.add_argument("--output_test_file", type=str, default="data/val/pile_test.h5", help="Output file for test data.")
    args = parser.parse_args()

    if not os.path.isdir(args.train_dir):
        print(f"Training directory {args.train_dir} does not exist. Please check the path.")
        return
    if not os.path.isdir(args.val_dir):
        print(f"Validation directory {args.val_dir} does not exist. Please check the path.")
        return

    # Create directories if they do not exist
    os.makedirs(os.path.dirname(args.output_train_file), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_val_file), exist_ok=True)
    # os.makedirs(os.path.dirname(args.output_test_file), exist_ok=True)

    data_processing(args.val_dir, args.output_val_file)
    data_processing(args.train_dir, args.output_train_file)
    data_processing(args.test_dir, args.output_test_file)
    
def data_processing(input_dir, output_file):
    """
    Processes text data from compressed JSONL files in the input directory, tokenizes the text, 
    and stores the tokenized data in an HDF5 file.
    Args:
        input_dir (str): Path to the directory containing input files. The files should be in 
                         `.jsonl.zst` format.
        output_file (str): Path to the output HDF5 file where the tokenized data will be stored.
    Workflow:
        1. Iterates through all `.jsonl.zst` files in the input directory.
        2. Decompresses and reads each file line by line.
        3. Extracts the "text" field from each JSON object.
        4. Tokenizes the text using the GPT-2 tokenizer and appends the tokenized data to an 
           HDF5 dataset.
        5. Handles empty text entries, JSON decoding errors, and other unexpected errors gracefully.
    Notes:
        - The tokenizer used is specific to GPT-2 and expects the `tiktoken` library.
        - The HDF5 dataset is dynamically resized to accommodate the tokenized data.
    Raises:
        json.JSONDecodeError: If a line in the JSONL file cannot be parsed as valid JSON.
        Exception: For any other unexpected errors during processing.
    Warnings:
        - Prints a warning message if an empty text entry is encountered.
        - Prints error messages for JSON decoding issues or other unexpected errors.
    Example:
        data_processing("data/input", "data/output/tokens.h5")
    """
    tokenizer = tiktoken.get_encoding("gpt2")

    with h5py.File(output_file, "w") as f:
        dataset = f.create_dataset("tokens", shape=(0,), maxshape=(None,), dtype='i')
        for filename in os.listdir(input_dir):
            if filename.endswith(".jsonl.zst"):
                file_path = os.path.join(input_dir, filename)
                with zstd.open(file_path, "r", encoding="utf-8") as file:
                    for line in tqdm(file, desc=f"Processing {filename}", unit="line"):
                        # Read each line and decode it
                        try:
                            data = json.loads(line)
                            text = data.get("text", "")
                            if text:
                                # Tokenize the text and append to the dataset
                                encoded = tokenizer.encode(text + "<|endoftext|>", allowed_special={"<|endoftext|>"})
                                encoded_len = len(encoded)
                                dataset.resize(dataset.shape[0] + encoded_len, axis=0)
                                dataset[-encoded_len:] = encoded
                            else:
                                print(f"warning: empty text found in {filename}. Skipping this entry.")
                        except json.JSONDecodeError as e:
                            print(f"Error decoding JSON in file {filename}: {e}")
                        except Exception as e:
                            print(f"Unexpected error in file {filename}: {e}")
                            continue


if __name__ == "__main__":
    main()
    print("Data preprocessing complete.")
