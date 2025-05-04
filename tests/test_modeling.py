import pytest
import torch
import torch.nn as nn

from model.modeling import MultiHeadAttention, LayerNorm, FeedForward, TransformerBlock, GPTModel
from model.configuration import GPTConfig

# Default configuration for testing
@pytest.fixture
def test_config():
    return GPTConfig(
        vocab_size=100,
        context_length=64,
        n_embd=32,
        n_heads=4,
        n_layers=2,
        dropout=0.1,
        qkv_bias=False,
        activation='gelu'
    )

# --- Test MultiHeadAttention ---

def test_multiheadattention_init(test_config):
    mha = MultiHeadAttention(
        d_in=test_config.n_embd,
        d_out=test_config.n_embd,
        context_length=test_config.context_length,
        num_heads=test_config.n_heads,
        dropout=test_config.dropout,
        qkv_bias=test_config.qkv_bias
    )
    assert mha.d_out == test_config.n_embd
    assert mha.num_heads == test_config.n_heads
    assert mha.head_dim == test_config.n_embd // test_config.n_heads
    assert isinstance(mha.query, nn.Linear)
    assert isinstance(mha.key, nn.Linear)
    assert isinstance(mha.value, nn.Linear)
    assert isinstance(mha.proj, nn.Linear)
    assert mha.mask.shape == (1, 1, test_config.context_length, test_config.context_length)

def test_multiheadattention_forward(test_config):
    batch_size = 4
    seq_len = test_config.context_length // 2 # Test with shorter sequence
    mha = MultiHeadAttention(
        d_in=test_config.n_embd,
        d_out=test_config.n_embd,
        context_length=test_config.context_length,
        num_heads=test_config.n_heads,
        dropout=test_config.dropout,
        qkv_bias=test_config.qkv_bias
    )
    x = torch.randn(batch_size, seq_len, test_config.n_embd)
    output = mha(x)
    assert output.shape == (batch_size, seq_len, test_config.n_embd)

# --- Test LayerNorm ---

def test_layernorm_init(test_config):
    ln = LayerNorm(emb_dim=test_config.n_embd)
    assert ln.emb_dim == test_config.n_embd
    assert ln.scale.shape == (test_config.n_embd,)
    assert ln.bias.shape == (test_config.n_embd,)

def test_layernorm_forward(test_config):
    batch_size = 4
    seq_len = test_config.context_length
    ln = LayerNorm(emb_dim=test_config.n_embd)
    x = torch.randn(batch_size, seq_len, test_config.n_embd) * 10 + 5 # Add scale and shift
    output = ln(x)

    assert output.shape == x.shape
    # Check if mean is close to 0 and std is close to 1 across the embedding dimension
    assert torch.allclose(output.mean(dim=-1), torch.zeros_like(output.mean(dim=-1)), atol=1e-5)
    assert torch.allclose(output.std(dim=-1, unbiased=False), torch.ones_like(output.std(dim=-1)), atol=1e-5)

# --- Test FeedForward ---

def test_feedforward_init(test_config):
    ff_gelu = FeedForward(n_embd=test_config.n_embd, activation='gelu')
    assert isinstance(ff_gelu.layer[1], nn.GELU)

    ff_relu = FeedForward(n_embd=test_config.n_embd, activation='relu')
    assert isinstance(ff_relu.layer[1], nn.ReLU)

    with pytest.raises(ValueError):
        FeedForward(n_embd=test_config.n_embd, activation='unknown')

def test_feedforward_forward(test_config):
    batch_size = 4
    seq_len = test_config.context_length
    ff = FeedForward(n_embd=test_config.n_embd, activation=test_config.activation)
    x = torch.randn(batch_size, seq_len, test_config.n_embd)
    output = ff(x)
    assert output.shape == x.shape

# --- Test TransformerBlock ---

def test_transformerblock_init(test_config):
    block = TransformerBlock(
        d_in=test_config.n_embd,
        d_out=test_config.n_embd,
        context_length=test_config.context_length,
        num_heads=test_config.n_heads,
        dropout=test_config.dropout,
        qkv_bias=test_config.qkv_bias,
        activation=test_config.activation
    )
    assert isinstance(block.ln1, LayerNorm)
    assert isinstance(block.attention, MultiHeadAttention)
    assert isinstance(block.resid_dropout1, nn.Dropout)
    assert isinstance(block.ln2, LayerNorm)
    assert isinstance(block.ff, FeedForward)
    assert isinstance(block.resid_dropout2, nn.Dropout)

def test_transformerblock_forward(test_config):
    batch_size = 4
    seq_len = test_config.context_length
    block = TransformerBlock(
        d_in=test_config.n_embd,
        d_out=test_config.n_embd,
        context_length=test_config.context_length,
        num_heads=test_config.n_heads,
        dropout=test_config.dropout,
        qkv_bias=test_config.qkv_bias,
        activation=test_config.activation
    )
    x = torch.randn(batch_size, seq_len, test_config.n_embd)
    output = block(x)
    assert output.shape == x.shape

# --- Test GPTModel ---

def test_gptmodel_init(test_config):
    model = GPTModel(test_config)
    assert isinstance(model.embedding, nn.Embedding)
    assert model.embedding.num_embeddings == test_config.vocab_size
    assert model.embedding.embedding_dim == test_config.n_embd
    assert isinstance(model.position_embedding, nn.Embedding)
    assert model.position_embedding.num_embeddings == test_config.context_length
    assert model.position_embedding.embedding_dim == test_config.n_embd
    assert isinstance(model.dropout, nn.Dropout)
    assert isinstance(model.transformer_blocks, nn.ModuleList)
    assert len(model.transformer_blocks) == test_config.n_layers
    assert isinstance(model.transformer_blocks[0], TransformerBlock)
    assert isinstance(model.norm_final, LayerNorm)
    assert isinstance(model.out_ff, nn.Linear)
    assert model.out_ff.out_features == test_config.vocab_size

def test_gptmodel_forward(test_config):
    batch_size = 4
    seq_len = test_config.context_length // 2 # Test shorter sequence
    model = GPTModel(test_config)
    model.eval() # Disable dropout for consistent testing
    x = torch.randint(0, test_config.vocab_size, (batch_size, seq_len))
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (batch_size, seq_len, test_config.vocab_size)

def test_gptmodel_generate(test_config):
    batch_size = 2
    start_seq_len = 5
    max_new_tokens = 10
    model = GPTModel(test_config)
    model.eval()
    idx = torch.randint(0, test_config.vocab_size, (batch_size, start_seq_len))

    # Test generation with temperature=0 (greedy)
    generated_idx_greedy = model.generate(idx, max_new_tokens=max_new_tokens, temperature=0.0)
    assert generated_idx_greedy.shape == (batch_size, start_seq_len + max_new_tokens)

    # Test generation with temperature > 0 and top_k
    generated_idx_sample = model.generate(idx, max_new_tokens=max_new_tokens, temperature=0.8, top_k=5)
    assert generated_idx_sample.shape == (batch_size, start_seq_len + max_new_tokens)

    # Test generation with eos_id (though it might not stop early with random weights)
    eos_id = test_config.vocab_size - 1 # Use last token as EOS
    generated_idx_eos = model.generate(idx, max_new_tokens=max_new_tokens, temperature=0.8, top_k=5, eos_id=eos_id)
    assert generated_idx_eos.shape[0] == batch_size
    assert generated_idx_eos.shape[1] <= start_seq_len + max_new_tokens # Might stop early