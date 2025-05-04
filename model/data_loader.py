from typing import Iterator, Tuple
import h5py
import numpy as np
import torch


def get_batch_iterator(data_path: str, batch_size: int = 32, context_length: int = 1024, device: str='cpu') -> Iterator[Tuple[torch.Tensor, torch.Tensor, int, int, int]]:
    if batch_size <= 0:
        raise ValueError(f"Batch size {batch_size} must be greater than 0.")
    if context_length <= 0:
        raise ValueError(f"Context length {context_length} must be greater than 0.")
    if device not in ['cpu', 'cuda']:
        raise ValueError(f"Device {device} must be 'cpu' or 'cuda'.")  

    with h5py.File(data_path, "r") as f:
        dataset = f["tokens"]
        dataset_size = dataset.shape[0]
        n_samples =  (dataset_size - 1) // context_length
        sample_orders = np.arange(n_samples)
        if batch_size > n_samples:
            raise ValueError(f"Batch size {batch_size} is larger than number of samples {n_samples}.")
        
        epoch = 0
        counter = 0
        while True:
            if counter + batch_size > n_samples:
                # Shuffle the indices for the next epoch
                np.random.shuffle(sample_orders)
                counter = 0
                epoch += 1

            indices = sample_orders[counter:counter + batch_size] * context_length
            samples = np.array([dataset[i:i + context_length + 1] for i in indices])
            input_ids = samples[:, :-1]
            target_ids = samples[:, 1:]
            counter += batch_size

            yield torch.tensor(input_ids, dtype=torch.long, device=device), torch.tensor(target_ids, dtype=torch.long, device=device)


if __name__ == "__main__":
    # Example usage
    data_path = "data/val/pile_val.h5"
    batch_size = 32
    context_length = 1024
    device = 'cpu'
    
    for input_ids, target_ids in get_batch_iterator(data_path, batch_size, context_length, device):
        print(f"Input IDs: {input_ids.shape}, Target IDs: {target_ids.shape}")
        break  # Remove this break to iterate through the entire dataset