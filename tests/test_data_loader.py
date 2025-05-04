import pytest
import torch
import h5py
import numpy as np
import tempfile
import os
from unittest.mock import patch

from model.data_loader import get_batch_iterator

# Fixture to create a dummy HDF5 file for testing
@pytest.fixture
def dummy_h5_file():
    # Create a temporary file
    temp_f = tempfile.NamedTemporaryFile(delete=False, suffix=".h5")
    file_path = temp_f.name
    temp_f.close() # Close it so h5py can open it

    # Define dataset parameters
    context_length = 10
    total_tokens = 105 # Enough for 10 samples of context_length 10
    data = np.arange(total_tokens, dtype=np.int64)

    # Write data to the HDF5 file
    with h5py.File(file_path, 'w') as f:
        f.create_dataset('tokens', data=data)

    # Yield the file path and context length
    yield file_path, context_length

    # Cleanup: remove the temporary file
    os.remove(file_path)

# --- Test Initialization Errors ---

def test_get_batch_iterator_invalid_batch_size(dummy_h5_file):
    file_path, context_length = dummy_h5_file
    with pytest.raises(ValueError, match="Batch size 0 must be greater than 0."):
        iterator = get_batch_iterator(file_path, batch_size=0, context_length=context_length)
        next(iterator) # Need to call next to trigger potential errors inside the generator

def test_get_batch_iterator_invalid_context_length(dummy_h5_file):
    file_path, _ = dummy_h5_file
    with pytest.raises(ValueError, match="Context length -1 must be greater than 0."):
        iterator = get_batch_iterator(file_path, batch_size=2, context_length=-1)
        next(iterator)

def test_get_batch_iterator_invalid_device(dummy_h5_file):
    file_path, context_length = dummy_h5_file
    with pytest.raises(ValueError, match="Device invalid_device must be 'cpu' or 'cuda'."):
        iterator = get_batch_iterator(file_path, batch_size=2, context_length=context_length, device="invalid_device")
        next(iterator)

def test_get_batch_iterator_batch_size_too_large(dummy_h5_file):
    file_path, context_length = dummy_h5_file
    # n_samples = (105 - 1) // 10 = 10
    with pytest.raises(ValueError, match="Batch size 11 is larger than number of samples 10."):
        iterator = get_batch_iterator(file_path, batch_size=11, context_length=context_length)
        next(iterator)

# --- Test Functionality ---

def test_get_batch_iterator_output_shape_type_device(dummy_h5_file):
    file_path, context_length = dummy_h5_file
    batch_size = 4
    device = 'cpu' # Assuming CPU for testing ease
    iterator = get_batch_iterator(file_path, batch_size=batch_size, context_length=context_length, device=device)

    input_ids, target_ids = next(iterator)

    assert input_ids.shape == (batch_size, context_length)
    assert target_ids.shape == (batch_size, context_length)
    assert input_ids.dtype == torch.long
    assert target_ids.dtype == torch.long
    assert str(input_ids.device) == device
    assert str(target_ids.device) == device

def test_get_batch_iterator_data_content(dummy_h5_file):
    file_path, context_length = dummy_h5_file
    batch_size = 2
    iterator = get_batch_iterator(file_path, batch_size=batch_size, context_length=context_length, device='cpu')

    input_ids, target_ids = next(iterator)

    # Check the first batch (indices 0 and 1, multiplied by context_length)
    # Sample 0: data[0:10] -> input, data[1:11] -> target
    # Sample 1: data[10:20] -> input, data[11:21] -> target
    expected_input_0 = torch.arange(0, 10, dtype=torch.long)
    expected_target_0 = torch.arange(1, 11, dtype=torch.long)
    expected_input_1 = torch.arange(10, 20, dtype=torch.long)
    expected_target_1 = torch.arange(11, 21, dtype=torch.long)

    assert torch.equal(input_ids[0], expected_input_0)
    assert torch.equal(target_ids[0], expected_target_0)
    assert torch.equal(input_ids[1], expected_input_1)
    assert torch.equal(target_ids[1], expected_target_1)

    # Check relationship within the batch
    assert torch.equal(input_ids[:, 1:], target_ids[:, :-1])

@patch('numpy.random.shuffle') # Mock shuffle to check if it's called
def test_get_batch_iterator_epochs_and_shuffling(mock_shuffle, dummy_h5_file):
    file_path, context_length = dummy_h5_file
    batch_size = 4
    # n_samples = (105 - 1) // 10 = 10
    # Batches per epoch = 10 // 4 = 2 batches (with remainder)
    batches_per_epoch = 10 // batch_size
    iterator = get_batch_iterator(file_path, batch_size=batch_size, context_length=context_length, device='cpu')

    # First epoch
    first_batch_input_1, _ = next(iterator)
    for _ in range(batches_per_epoch - 1):
        next(iterator)

    # At this point, counter should be batch_size * batches_per_epoch = 4 * 2 = 8
    # The next call should trigger shuffle and reset counter
    assert mock_shuffle.call_count == 0
    second_batch_input_1, _ = next(iterator) # Start of second epoch
    assert mock_shuffle.call_count == 1

    # Check that the data is potentially different after shuffle (though mock doesn't actually reorder)
    # In a real scenario, the order would likely change. Here we just check shuffle was called.
    # We can also check that we get the expected number of batches again.
    for _ in range(batches_per_epoch - 1):
        next(iterator)

    # Check third epoch start
    assert mock_shuffle.call_count == 1 # Shuffle only happens at epoch boundary
    third_batch_input_1, _ = next(iterator)
    assert mock_shuffle.call_count == 2 # Shuffle called again


def test_get_batch_iterator_exact_batch_size(dummy_h5_file):
    file_path, context_length = dummy_h5_file
    # n_samples = (105 - 1) // 10 = 10
    batch_size = 10
    iterator = get_batch_iterator(file_path, batch_size=batch_size, context_length=context_length, device='cpu')
    input_ids, target_ids = next(iterator)
    assert input_ids.shape == (batch_size, context_length)
    # The next call should immediately trigger shuffle for the next epoch
    with patch('numpy.random.shuffle') as mock_shuffle:
        next(iterator)
        mock_shuffle.assert_called_once()