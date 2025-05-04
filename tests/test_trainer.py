import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from unittest.mock import MagicMock, patch, call, ANY
import tempfile
import os

from model.trainer import Trainer
from model.modeling import GPTModel
from model.configuration import GPTConfig

# --- Fixtures ---

@pytest.fixture
def mock_config():
    # Minimal config for testing
    return GPTConfig(
        vocab_size=50,
        context_length=16,
        n_embd=8,
        n_heads=2,
        n_layers=1,
        dropout=0.0,
        qkv_bias=False,
        activation='gelu'
    )

@pytest.fixture
def mock_model(mock_config):
    # Use MagicMock to simulate the model behavior
    model = MagicMock(spec=GPTModel)
    model.parameters.return_value = [nn.Parameter(torch.randn(1))] # Need parameters for optimizer and clipping
    model.apply = MagicMock() # Mock apply for weight init
    model.to = MagicMock(return_value=model) # Mock device moving
    model.train = MagicMock()
    model.eval = MagicMock()
    model.state_dict = MagicMock(return_value={'param': torch.tensor([1.0])}) # Mock state dict
    # Mock the forward call
    model.return_value = torch.randn(4, mock_config.context_length, mock_config.vocab_size) # (B, T, V)
    return model

@pytest.fixture
def mock_optimizer(mock_model):
    # Use MagicMock for the optimizer
    optimizer = MagicMock(spec=optim.AdamW)
    optimizer.state_dict = MagicMock(return_value={'state': 'optim_state'}) # Mock state dict
    optimizer.zero_grad = MagicMock()
    optimizer.step = MagicMock()
    return optimizer

@pytest.fixture
def mock_data_iterator(mock_config):
    # Create a simple generator that yields mock batches
    def _iterator():
        while True:
            input_batch = torch.randint(0, mock_config.vocab_size, (4, mock_config.context_length), dtype=torch.long)
            target_batch = torch.randint(0, mock_config.vocab_size, (4, mock_config.context_length), dtype=torch.long)
            yield input_batch, target_batch
    return _iterator()

@pytest.fixture
def temp_checkpoint_dir():
    # Create a temporary directory for saving checkpoints
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

# --- Tests ---

@patch('torch.nn.utils.clip_grad_norm_')
def test_trainer_init(mock_clip_grad, mock_model, mock_data_iterator, mock_optimizer):
    device = 'cpu'
    trainer = Trainer(mock_model, mock_data_iterator, mock_data_iterator, mock_optimizer, device)

    assert trainer.model == mock_model
    assert trainer.train_data_iterator == mock_data_iterator
    assert trainer.val_data_iterator == mock_data_iterator
    assert trainer.optimizer == mock_optimizer
    assert trainer.device == device

    # Check if methods were called
    mock_model.apply.assert_called_once() # _initialize_weights called
    mock_clip_grad.assert_called_once_with(mock_model.parameters(), max_norm=1.0)
    mock_model.to.assert_called_once_with(device)

def test_trainer_initialize_weights():
    # Test the actual weight initialization logic on a small layer
    trainer = MagicMock() # Mock trainer instance, only need the method
    linear_layer = nn.Linear(10, 5, bias=True)
    # Bind the method to the mock instance for testing
    bound_init = Trainer._initialize_weights.__get__(trainer, Trainer)
    bound_init(linear_layer)

    # Check if weights are initialized (not zero)
    assert not torch.allclose(linear_layer.weight, torch.zeros_like(linear_layer.weight))
    # Check if bias is initialized to 0.01
    assert torch.allclose(linear_layer.bias, torch.full_like(linear_layer.bias, 0.01))

def test_trainer_calc_loss_batch(mock_model, mock_config):
    trainer = Trainer(mock_model, None, None, None, 'cpu') # Iterators/optimizer not needed
    batch_size = 4
    input_batch = torch.randint(0, mock_config.vocab_size, (batch_size, mock_config.context_length), dtype=torch.long)
    target_batch = torch.randint(0, mock_config.vocab_size, (batch_size, mock_config.context_length), dtype=torch.long)

    # Mock model output
    mock_logits = torch.randn(batch_size, mock_config.context_length, mock_config.vocab_size, requires_grad=True)
    mock_model.return_value = mock_logits

    loss = trainer.calc_loss_batch(input_batch, target_batch)

    # Check if model was called with input_batch
    mock_model.assert_called_once_with(input_batch)
    # Check if loss is calculated (cross_entropy)
    expected_loss = torch.nn.functional.cross_entropy(mock_logits.flatten(0, 1), target_batch.flatten())
    assert torch.isclose(loss, expected_loss)
    assert loss.requires_grad # Ensure loss requires grad

@patch('model.trainer.tqdm', lambda x, **kwargs: x) # Mock tqdm to avoid progress bar output
def test_trainer_evaluate_model(mock_model, mock_data_iterator):
    trainer = Trainer(mock_model, None, mock_data_iterator, None, 'cpu')
    eval_iter = 3
    fixed_loss_value = 1.5

    # Mock calc_loss_batch to return a fixed value
    trainer.calc_loss_batch = MagicMock(return_value=torch.tensor(fixed_loss_value))

    avg_loss = trainer.evaluate_model(eval_iter)

    # Check model mode switching
    assert mock_model.eval.call_count == 1
    assert mock_model.train.call_count == 1 # Should be called after eval
    # Check calc_loss_batch calls
    assert trainer.calc_loss_batch.call_count == eval_iter
    # Check returned average loss
    assert avg_loss == pytest.approx(fixed_loss_value)

@patch('model.trainer.tqdm', lambda x, **kwargs: x) # Mock tqdm
@patch('model.trainer.logger') # Mock logger
def test_trainer_train_model(mock_logger, mock_model, mock_data_iterator, mock_optimizer, temp_checkpoint_dir):
    trainer = Trainer(mock_model, mock_data_iterator, mock_data_iterator, mock_optimizer, 'cpu')
    n_steps = 11
    eval_freq = 5
    eval_iter = 2
    checkpoint_freq = 5
    checkpoint_path = os.path.join(temp_checkpoint_dir, "test_ckpt.pt")

    # Mock loss calculation and evaluation
    mock_train_loss = torch.tensor(2.0, requires_grad=True)
    mock_val_loss = 1.8
    trainer.calc_loss_batch = MagicMock(return_value=mock_train_loss)
    trainer.evaluate_model = MagicMock(return_value=mock_val_loss)
    trainer.save_model = MagicMock() # Mock save_model

    train_losses, val_losses, tokens_seen_list = trainer.train_model(
        n_steps, eval_freq, eval_iter, checkpoint_path, checkpoint_freq
    )

    # --- Assertions ---
    # Check training loop calls
    assert trainer.calc_loss_batch.call_count == n_steps
    assert mock_optimizer.zero_grad.call_count == n_steps
    # Cannot directly check loss.backward() on MagicMock easily, but check optimizer.step
    assert mock_optimizer.step.call_count == n_steps

    # Check evaluation calls (steps 0, 5, 10)
    assert trainer.evaluate_model.call_count == (n_steps // eval_freq) + 1
    trainer.evaluate_model.assert_called_with(eval_iter)

    # Check checkpoint saving calls (steps 0, 5, 10)
    assert trainer.save_model.call_count == (n_steps // checkpoint_freq) + 1
    expected_save_calls = [
        call(checkpoint_path, ANY, ANY, 0),
        call(checkpoint_path, ANY, ANY, 5),
        call(checkpoint_path, ANY, ANY, 10),
    ]
    trainer.save_model.assert_has_calls(expected_save_calls)

    # Check returned lists
    expected_eval_count = (n_steps // eval_freq) + 1
    assert len(train_losses) == expected_eval_count
    assert len(val_losses) == expected_eval_count
    assert len(tokens_seen_list) == expected_eval_count
    assert all(loss == mock_train_loss for loss in train_losses) # Check actual loss object stored
    assert all(loss == mock_val_loss for loss in val_losses)

    # Check logging calls (steps 0, 5, 10)
    assert mock_logger.info.call_count >= expected_eval_count * 2 # Eval log + Checkpoint log

def test_trainer_save_model(mock_model, mock_optimizer, temp_checkpoint_dir):
    trainer = Trainer(mock_model, None, None, mock_optimizer, 'cpu')
    checkpoint_path = os.path.join(temp_checkpoint_dir, "test_save.pt")
    train_losses = [torch.tensor(2.5), torch.tensor(2.1)]
    val_losses = [1.9, 1.7]
    total_steps = 100

    trainer.save_model(checkpoint_path, train_losses, val_losses, total_steps)

    # Check if the file exists
    assert os.path.exists(checkpoint_path)

    # Load the checkpoint and verify contents
    checkpoint = torch.load(checkpoint_path)
    assert 'model_state_dict' in checkpoint
    assert 'optimizer_state_dict' in checkpoint
    assert 'train_losses' in checkpoint
    assert 'val_losses' in checkpoint
    assert 'train_loss' in checkpoint
    assert 'val_loss' in checkpoint
    assert 'steps' in checkpoint

    assert checkpoint['model_state_dict'] == mock_model.state_dict()
    assert checkpoint['optimizer_state_dict'] == mock_optimizer.state_dict()
    assert checkpoint['train_losses'] == train_losses
    assert checkpoint['val_losses'] == val_losses
    assert checkpoint['train_loss'] == train_losses[-1]
    assert checkpoint['val_loss'] == val_losses[-1]
    assert checkpoint['steps'] == total_steps