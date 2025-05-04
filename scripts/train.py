import logging
import torch
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from model.configuration import GPTConfig
from model.data_loader import get_batch_iterator
from model.modeling import GPTModel
from model.trainer import Trainer


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# --- Configure basic logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# -------------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Training a GPT model...")
    logger.info("create a GPT model...")
    torch.manual_seed(123)
    model = GPTModel(GPTConfig)
    model.to(device)
    logger.info("Model created.")
    
     # --- Print Model Architecture and Parameters ---
    logger.info(f"Model architecture:\n{model}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params}")
    logger.info(f"Trainable parameters: {trainable_params}")
    # ---------------------------------------------

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00009, weight_decay=1e-1)

    logger.info("Loading data...")
    batch_size: int = 32
    # train_data_iterator = get_batch_iterator("data/train/pile_train.h5", batch_size=32, context_length=GPTConfig.context_length, device=device)
    # val_data_iterator = get_batch_iterator("data/val/pile_val.h5", batch_size=batch_size, context_length=GPTConfig.context_length, device=device)
    train_data_iterator = get_batch_iterator("data/val/pile_val.h5", batch_size=batch_size, context_length=GPTConfig.context_length, device=device)
    val_data_iterator = get_batch_iterator("data/val/pile_test.h5", batch_size=batch_size, context_length=GPTConfig.context_length, device=device)
    logger.info("Data loaded.")

    total_steps = 5
    eval_freq=5
    eval_iter=1
    trainer = Trainer(model, train_data_iterator, val_data_iterator, optimizer, device)
    train_losses, val_losses, track_tokens_seen = trainer.train_model(total_steps, eval_freq=eval_freq, eval_iter=eval_iter, checkpoint_path=GPTConfig.model_path, checkpoint_freq=5)
    logger.info(f"Training loss: {train_losses}")
    logger.info(f"Validation loss: {val_losses}")
    logger.info(f"Tokens seen: {track_tokens_seen}")
    logger.info("Training complete.")

if __name__ == "__main__":
    main()
