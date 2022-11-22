# %%
from typing import Optional
from dataclasses import dataclass

MAIN = __name__ == "__main__"


@dataclass
class FlaxGPTConfig:
    """Dataclass for configuring a GPT-style transformer model.

    Args:
        d_model: Dimensionality of the residual stream
        n_heads: Number of attention heads
        n_layers: Number of transformer blocks
        n_ctx: Max number of tokens in the context
        d_vocab: Dimensionality of the input token embedding
        d_vocab_out: Dimensionality of the output token embedding
        layer_norm_eps: Epsilon for layer normalization
        attn_only: If True, only use attention layers for each transformer block
        mlp_dim: Dimensionality of the MLP layer in the transformer block. Defaults
            to 4 * d_model
        activation: Activation function to use in the transformer block. Defaults
            to "gelu" - only relu and gelu are currently supported.
        bidirectional: If True, use a bidirectional transformer block
        n_classes: Number of classes for classification, defaults to self.d_vocab_out.

    """

    d_model: int
    n_heads: int
    n_layers: int
    n_ctx: int
    d_vocab: int
    d_vocab_out: Optional[int] = None
    layer_norm_eps: float = 1e-5
    attn_only: bool = False
    mlp_dim: Optional[int] = None
    activation: Optional[str] = None
    bidirectional: bool = False
    n_classes: Optional[int] = None

    def __post_init__(self):
        self.d_head = self.d_model // self.n_heads
        if not self.attn_only:
            if not self.mlp_dim:
                self.mlp_dim = 4 * self.d_model
            if not self.activation:
                self.activation = "gelu"
            else:
                assert (
                    self.activation == "gelu" or self.activation == "relu"
                ), "Only GELU and ReLU Activations supported right now."
        self.d_vocab_out = self.d_vocab_out or self.d_vocab
        if self.bidirectional and not self.n_classes:
            self.n_classes = self.d_vocab_out


if MAIN:
    config = FlaxGPTConfig(768, 12, 12, 50257, 1024)
    print(config)

# %%
