import torch
from dataclasses import dataclass, asdict


@dataclass
class GPTConfig:
    """Configuration class for the dataset.
    For GPT2, its configuration objects inherit from [`PretrainedConfig`] which is the base class for all 
    configuration classes and provides methods for loading/downloading/saving configurations. Read the
    documentation from transformers.models.gpts.configuration_gpt2 for more information.
    """
    vocab_size: int = 50257 # Vocabulary size
    # The size of the vocabulary used by the tokenizer. This is the number of unique tokens that can be represented.
    context_length: int = 512   
    n_embd: int = 768 # Embedding dimension
    n_heads: int = 12 # Number of attention heads
    n_layers: int = 12   # Number of transformer blocks
    n_positions: int = 1024  # Maximum sequence length
    activation: str = "gelu"  # Activation function to use
    dropout: float = 0.1  # Dropout rate
    layer_norm_epsilon: float = 1e-5  # Epsilon for layer normalization
    initializer_range: float = 0.02  # Standard deviation for weight initialization
    bos_token_id: int = 50256  # Beginning of sentence token ID
    eos_token_id: int = 50256  # End of sentence token ID   
    qkv_bias: bool = False
    model_path: str = "gpt_model.pt"  # Path to the model file
    model_name: str = "gpt_model" # Name of the model
    model_type: str = "gpt2" # Type of the model (e.g., gpt2, gpt3, etc.)

    def to_dict(self) -> dict:
        """Convert the configuration to a dictionary."""
        return asdict(self)
    
    def __repr__(self):
        config_dict = self.to_dict()
        config_str = ",\n    ".join(f"{k}: {repr(v)}" for k, v in config_dict.items())
        return "GPTConfig = {\n    " + config_str + "\n}"
        
GPTConfig = GPTConfig()

