import logging
import numpy as np
import torch 
import torch.nn as nn

from .configuration import GPTConfig


logger = logging.getLogger(__name__)

class MultiHeadAttention(nn.Module):
    """
    MultiHeadAttention is a PyTorch module that implements the multi-head attention mechanism, 
    commonly used in transformer architectures.
    Attributes:
        d_out (int): The output dimension of the attention layer.
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head, calculated as d_out // num_heads.
        query (nn.Linear): Linear layer to project input to query vectors.
        key (nn.Linear): Linear layer to project input to key vectors.
        value (nn.Linear): Linear layer to project input to value vectors.
        proj (nn.Linear): Linear layer to project the concatenated attention output.
        dropout (nn.Dropout): Dropout layer applied to attention scores.
        mask (torch.Tensor): A buffer containing the lower triangular mask for causal attention.
    Methods:
        forward(x: torch.Tensor) -> torch.Tensor:
            Computes the multi-head attention output for the input tensor `x`.
        _transpose(x: torch.Tensor) -> torch.Tensor:
            Reshapes and permutes the input tensor `x` to prepare it for multi-head attention computation.
    Args:
        d_in (int): The input dimension of the attention layer.
        d_out (int): The output dimension of the attention layer.
        context_length (int): The maximum sequence length for the input.
        num_heads (int): The number of attention heads.
        dropout (float): Dropout probability applied to attention scores.
        qkv_bias (bool, optional): Whether to include a bias term in the query, key, and value projections. Defaults to False.
    """

    def __init__(self, d_in: int, d_out: int,  context_length: int, num_heads: int, dropout: float, qkv_bias: bool = False) -> None:
        super(MultiHeadAttention, self).__init__()
        assert d_out % num_heads == 0, f"output dimension {d_out} not divisible by number of heads {num_heads}"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads


        self.query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer("mask", torch.tril(torch.ones(context_length, context_length)).unsqueeze(0).unsqueeze(0))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        q = self._transpose(self.query(x))
        k = self._transpose(self.key(x))
        v = self._transpose(self.value(x))

        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(self.head_dim)
        attn_scores = attn_scores.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(B, T, C)

        return self.proj(attn_output)

        
    def _transpose(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    

class LayerNorm(nn.Module):
    """
    A custom implementation of Layer Normalization.
    Layer Normalization is a technique to normalize the inputs across the features
    for each data sample independently. It helps stabilize and accelerate training
    of deep neural networks.
    Attributes:
        emb_dim (int): The dimensionality of the input embeddings.
        eps (float): A small value added to the denominator for numerical stability.
        scale (torch.nn.Parameter): A learnable parameter to scale the normalized input.
        bias (torch.nn.Parameter): A learnable parameter to shift the normalized input.
    Methods:
        forward(x: torch.Tensor) -> torch.Tensor:
            Applies layer normalization to the input tensor.
    Args:
        emb_dim (int): The size of the last dimension of the input tensor.
        eps (float, optional): A small constant for numerical stability. Default is 1e-5.
    Example:
        >>> layer_norm = LayerNorm(emb_dim=128)
        >>> x = torch.randn(32, 128)  # Batch of 32 samples with 128 features each
        >>> normalized_x = layer_norm(x)
    """

    def __init__(self, emb_dim: int, eps: float = 1e-5) -> None:
        super(LayerNorm, self).__init__()
        self.emb_dim = emb_dim
        self.eps = eps 
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.bias = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False)
        x = (x - mean) / (std + self.eps)
        x = x * self.scale + self.bias
        return x


class FeedForward(nn.Module):
    """
    A feedforward neural network module with configurable activation function and dropout.
    Args:
        n_embd (int): The size of the input and output embeddings.
        activation (str, optional): The activation function to use. 
            Supported values are 'gelu' and 'relu'. Defaults to 'gelu'.
    Raises:
        ValueError: If an unsupported activation function is provided.
    Methods:
        forward(x: torch.Tensor) -> torch.Tensor:
            Applies the feedforward network to the input tensor.
    Example:
        >>> feedforward = FeedForward(n_embd=128, activation='relu', dropout=0.1)
        >>> x = torch.randn(32, 128)  # Batch of 32 with embedding size 128
        >>> output = feedforward(x)
    """

    def __init__(self, n_embd: int, activation: str='gelu') -> None:
        super(FeedForward, self).__init__()
        
        if activation == "gelu":
            activattion = nn.GELU()
        elif activation == "relu":
            activattion = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation function: {activation}")  
        
        self.layer = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            activattion,
            nn.Linear(4 * n_embd, n_embd)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)
    

class TransformerBlock(nn.Module):
    """
    A Transformer block that consists of a multi-head self-attention mechanism 
    followed by a feed-forward neural network, with residual connections, 
    layer normalization, and dropout applied at each stage.
    Attributes:
        ln1 (LayerNorm): Layer normalization applied before the attention mechanism.
        attention (MultiHeadAttention): Multi-head self-attention mechanism.
        resid_dropout1 (nn.Dropout): Dropout applied after the attention mechanism.
        ln2 (LayerNorm): Layer normalization applied before the feed-forward network.
        ff (FeedForward): Feed-forward neural network.
        resid_dropout2 (nn.Dropout): Dropout applied after the feed-forward network.
    Args:
        d_in (int): Dimensionality of the input features.
        d_out (int): Dimensionality of the output features.
        context_length (int): Length of the input sequence.
        num_heads (int): Number of attention heads in the multi-head attention mechanism.
        dropout (float): Dropout probability applied after attention and feed-forward layers.
        qkv_bias (bool, optional): Whether to include bias terms in the query, key, and value projections. Defaults to False.
    Methods:
        forward(x: torch.Tensor) -> torch.Tensor:
            Performs a forward pass through the Transformer block.
            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, context_length, d_in).
            Returns:
                torch.Tensor: Output tensor of shape (batch_size, context_length, d_out).
    """

    def __init__(self, d_in: int, d_out: int, context_length: int, num_heads: int, dropout: float, qkv_bias: bool = False, activation: str='gelu') -> None:
        super(TransformerBlock, self).__init__()
        self.ln1 = LayerNorm(d_in)
        self.attention = MultiHeadAttention(d_in, d_out, context_length, num_heads, dropout, qkv_bias)
        self.resid_dropout1 = nn.Dropout(dropout)
        self.ln2 = LayerNorm(d_in)
        self.ff = FeedForward(d_in, activation=activation)
        self.resid_dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.resid_dropout1(self.attention(self.ln1(x)))
        x = x + self.resid_dropout2(self.ff(self.ln2(x)))
        return x


class GPTModel(nn.Module):
    """
    GPTModel is a PyTorch implementation of a Generative Pre-trained Transformer (GPT) model.
    Attributes:
        config (GPTConfig): Configuration object containing model hyperparameters.
        embedding (nn.Embedding): Token embedding layer.
        position_embedding (nn.Embedding): Positional embedding layer.
        dropout (nn.Dropout): Dropout layer applied after embeddings.
        transformer_blocks (nn.ModuleList): List of Transformer blocks.
        norm_final (LayerNorm): Final layer normalization.
        out_ff (nn.Linear): Output feed-forward layer projecting to vocabulary size.
    Methods:
        forward(x: torch.Tensor) -> torch.Tensor:
            Performs a forward pass through the model.
            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, sequence_length).
            Returns:
                torch.Tensor: Logits of shape (batch_size, sequence_length, vocab_size).
        generate_simple(idx: torch.Tensor, max_new_tokens: int, context_size: int) -> torch.Tensor:
            Generates tokens using greedy decoding.
            Args:
                idx (torch.Tensor): Input tensor of token indices of shape (batch_size, sequence_length).
                max_new_tokens (int): Maximum number of new tokens to generate.
                context_size (int): Context size to consider for generation.
            Returns:
                torch.Tensor: Generated token indices of shape (batch_size, sequence_length + max_new_tokens).
        generate(idx: torch.Tensor, max_new_tokens: int, context_size: int, temperature: float = 0.0, top_k: int = None, eos_id: int = None) -> torch.Tensor:
            Generates tokens using sampling with optional temperature, top-k filtering, and early stopping.
            Args:
                idx (torch.Tensor): Input tensor of token indices of shape (batch_size, sequence_length).
                max_new_tokens (int): Maximum number of new tokens to generate.
                context_size (int): Context size to consider for generation.
                temperature (float, optional): Sampling temperature. Defaults to 0.0.
                top_k (int, optional): Number of top logits to consider for sampling. Defaults to None.
                eos_id (int, optional): End-of-sequence token ID for early stopping. Defaults to None.
            Returns:
                torch.Tensor: Generated token indices of shape (batch_size, sequence_length + max_new_tokens).
    """
    def __init__(self, config: GPTConfig) -> None:
        super(GPTModel, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.context_length, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        # self.transformer_blocks = nn.Sequential(*[TransformerBlock(config.n_embd, config.n_embd, config.context_length, config.n_heads, config.dropout, config.qkv_bias, config.activation) for _ in range(config.n_layers)])
        self.transformer_blocks = nn.ModuleList([TransformerBlock(config.n_embd, config.n_embd, config.context_length, config.n_heads, config.dropout, config.qkv_bias, config.activation) for _ in range(config.n_layers)])
        self.norm_final = LayerNorm(config.n_embd)
        self.out_ff = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.register_buffer('pos_idxs', torch.arange(config.context_length))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.size()
        # pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        # x = self.embedding(x) + self.position_embedding(pos)
        pos_embeding = self.position_embedding(self.pos_idxs[:T])
        x = self.embedding(x) + pos_embeding
        x = self.dropout(x)
        
        for block in self.transformer_blocks:
            x = block(x)
        # x = self.transformer_blocks(x)
        x = self.norm_final(x)
        logits = self.out_ff(x)

        return logits
    
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int = 10, eos_id: int = None) -> torch.Tensor:
        context_size = self.config.context_length
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_size:]
            with torch.no_grad():
                logits = self(idx_cond)
            if temperature == 0.0:
                idx_next = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            else:
                logits = logits[:, -1, :] / temperature
            
                if top_k is not None:
                    top_k_values, top_k_indices = torch.topk(logits, top_k)
                    logits[logits < top_k_values[:, -1, None]] = float("-inf")
                
                if eos_id is not None:
                    logits[:, eos_id] = float("-inf")

                probs = torch.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
        

    def load(self, path: str= "gpt_model_state_dict.pth", device = None) -> None:
        """Load the model from a file."""
        self.load_state_dict(torch.load(path, map_location=device))
        logger.infot(f"Model loaded from {path}")
        self.to(device)
        self.eval() # Set the model to evaluation mode

