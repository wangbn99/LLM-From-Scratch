import logging
from typing import Iterator
import numpy as np
import torch 
import torch.nn as nn
import tiktoken as ttk

from tqdm import tqdm
from .modeling import GPTModel

logger = logging.getLogger(__name__)

class Trainer:
    """Trainer class for training the model."""

    def __init__(self, model: GPTModel, train_data_iterator: Iterator[tuple[torch.Tensor, torch.Tensor]], val_data_iterator: Iterator[tuple[torch.Tensor, torch.Tensor]], optimizer: torch.optim.Optimizer, device: str) -> None:
        self.model = model
        self.train_data_iterator = train_data_iterator
        self.val_data_iterator = val_data_iterator
        self.optimizer = optimizer
        self.device = device
        self.model.apply(self._initialize_weights)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.model.to(self.device)


    def _initialize_weights(self, m):
        """Initialize weights of the model."""
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)


    def calc_loss_batch(self, input_batch: torch.Tensor, target_batch: torch.Tensor) -> torch.Tensor:
        # input_batch, target_batch = input_batch.to(self.device), target_batch.to(self.device)
        logits = self.model(input_batch)
        loss = torch.nn.functional.cross_entropy(logits.flatten(0,1), target_batch.flatten())
        return loss
    

    def evaluate_model(self, eval_iter) -> float:
        self.model.eval()
        total_loss = 0.0    
        with torch.no_grad():
            for _ in tqdm(range(eval_iter), desc="Calculating loss"):
                input_batch, target_batch = next(self.val_data_iterator)
                loss = self.calc_loss_batch(input_batch, target_batch)
                total_loss += loss.item()
        self.model.train()
        return total_loss / eval_iter


    def train_model(self, n_steps: int, eval_freq, eval_iter, checkpoint_path: str, checkpoint_freq: int) -> tuple:
        train_losses, val_losses, track_tokens_seen = [], [], []
        tokens_seen = 0

        for step in tqdm(range(n_steps), desc="Training steps"):
            # Get the first batch of data to initialize the model  
            input_batch, target_batch = next(self.train_data_iterator)
            self.model.train()
            self.optimizer.zero_grad()
            loss = self.calc_loss_batch(input_batch, target_batch)
            loss.backward()
            self.optimizer.step()
            tokens_seen += input_batch.numel()

            if step % eval_freq == 0:
                # train_loss, val_loss = self.evaluate_model(eval_iter)
                val_loss = self.evaluate_model(eval_iter)
                train_losses.append(loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)

                logger.info(f"Step {step:03d}: "
                    f"Train Loss: {loss:.3f}, Val Loss: {val_loss:.3f}, Tokens Seen: {tokens_seen}")
            if step % checkpoint_freq == 0:
                self.save_model(checkpoint_path, train_losses, val_losses, step)
                logger.info(f"Checkpoint saved at step {step}.")
                # Save the model state

        return train_losses, val_losses, track_tokens_seen
    
    def save_model(self, checkpoint_path: str, train_losses: list, val_losses: list, total_steps: int) -> None:
        torch.save(
            {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_loss': train_losses[-1],
                'val_loss': val_losses[-1],
                'steps': total_steps,
            },
            checkpoint_path
        )
