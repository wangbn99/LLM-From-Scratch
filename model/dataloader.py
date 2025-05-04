import h5py
import numpy as np
from typing import Tuple
import torch
from torch.utils.data import Dataset, DataLoader

class HDF5TokenDataset(Dataset):
    """
    A PyTorch Dataset for loading token sequences from an HDF5 file.
    """
    def __init__(self, data_path: str, context_length: int):
        super().__init__()
        if context_length <= 0:
            raise ValueError(f"Context length {context_length} must be greater than 0.")

        self.data_path = data_path
        self.context_length = context_length

        with h5py.File(self.data_path, "r") as f:
            self.dataset = f["tokens"]
            self.dataset_size = f["tokens"].shape[0]
            print(self.dataset[0:10])

        self.n_samples = (self.dataset_size - 1) // self.context_length
        if self.n_samples <= 0:
             raise ValueError(f"Dataset size ({self.dataset_size}) is too small for context length ({self.context_length}).")
        print(f"Dataset size: {self.dataset_size}, Number of samples: {self.n_samples}")
        print(self.dataset[0:10])

    def __len__(self) -> int:
        # Return the total number of possible sequences
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # idx represents the starting position of the input sequence

        idx = idx * self.context_length
        input_ids = self.dataset[idx:idx + self.context_length]
        target_ids = self.dataset[idx + 1:idx + self.context_length + 1]
        return torch.tensor(input_ids, device=self.device), torch.tensor(target_ids, device=self.device)


def get_dataloader(data_path: str, batch_size: int = 32, context_length: int = 1024, shuffle: bool = True, drop_last: bool = True) -> DataLoader:
    """
    Create a DataLoader for the HDF5 token dataset.
    """
    if batch_size <= 0:
        raise ValueError(f"Batch size {batch_size} must be greater than 0.")
    if context_length <= 0:
        raise ValueError(f"Context length {context_length} must be greater than 0.")

    dataset = HDF5TokenDataset(data_path, context_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    return dataloader


if __name__ == "__main__":
    # Example usage
    data_path = "data/val/pile_val.h5"
    batch_size = 32
    context_length = 1024
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    dataloader = get_dataloader(data_path, batch_size, context_length)
    for input_ids, target_ids in dataloader:
        print(f"Input IDs: {input_ids.shape}, Target IDs: {target_ids.shape}")
        break  # Remove this break to iterate through the entire dataset