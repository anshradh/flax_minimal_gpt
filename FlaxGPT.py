# %%
import jax
from jax import random, numpy as jnp
import flax.linen as nn
from typing import Optional, Union
from dataclasses import dataclass
from einops import rearrange

MAIN = __name__ == "__main__"
# %%
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


if MAIN:
    config = FlaxGPTConfig(768, 12, 12, 50257, 1024)
    print(config)
# %%
class FlaxGPTInnerAttention(nn.Module):
    """Attention layer for GPT-style transformer models. This is currently set up
    for a single sequence - it'll be vmapped later to handle batches efficiently.

    Args:
        config: GPTConfig object containing the model configuration
    """

    config: FlaxGPTConfig

    def setup(self):
        self.qkv_proj = nn.Dense(self.config.d_model * 3)
        self.out_proj = nn.Dense(self.config.d_model)

    def __call__(self, x):
        seq_len, _ = x.shape
        qkv = self.qkv_proj(x)
        q_init, k_init, v_init = jnp.array_split(
            qkv,
            3,
            axis=1,
        )
        q = rearrange(
            q_init,
            "seq (n_heads d_head) -> n_heads seq d_head",
            n_heads=self.config.n_heads,
        )
        k = rearrange(
            k_init,
            "seq (n_heads d_head) -> n_heads seq d_head",
            n_heads=self.config.n_heads,
        )
        v = rearrange(
            v_init,
            "seq (n_heads d_head) -> n_heads seq d_head",
            n_heads=self.config.n_heads,
        )
        attn = jax.vmap(lambda q, k: jnp.einsum("... q h, ... k h-> ... q k", q, k,),)(
            q,
            k,
        ) / jnp.sqrt(self.config.d_head)
        masked_attn = jnp.where(
            jnp.arange(seq_len)[:, None] >= jnp.arange(seq_len)[None, :],
            attn,
            float("-inf"),
        )
        softmaxed_attn = nn.softmax(masked_attn)
        combined_values = jax.vmap(
            lambda attn, v: jnp.einsum(
                "... q k, ... k h -> ... q h",
                attn,
                v,
            ),
            in_axes=(0, 0),
        )(softmaxed_attn, v)
        combined_values_rearranged = rearrange(
            combined_values,
            "n_heads seq d_head -> seq (n_heads d_head)",
        )
        out = self.out_proj(combined_values_rearranged)
        return out


class FlaxGPTAttention(nn.Module):
    """Batched attention layer for GPT-style transformer models.

    Args:
        config: GPTConfig object containing the model configuration
    """

    config: FlaxGPTConfig

    @nn.compact
    def __call__(self, x):
        return nn.vmap(
            FlaxGPTInnerAttention,
            in_axes=0,
            out_axes=0,
            variable_axes=dict(params=None),
            split_rngs=dict(params=False),
        )(config)(x)


if MAIN:
    key1, key2 = random.split(random.PRNGKey(0))
    x = random.normal(key1, (2, 5, 8))
    config = FlaxGPTConfig(8, 2, 2, 10, 10)
    attn_module = FlaxGPTAttention(config)
    attn_module_params = attn_module.init(key2, x)
    out = attn_module.apply(attn_module_params, x)
    print(out.shape)
    print(out)
# %%
class FlaxGPTMLP(nn.Module):
    """MLP layer for GPT-style transformer models.

    Args:
        config: GPTConfig object containing the model configuration
    """

    config: FlaxGPTConfig

    def setup(self):
        assert self.config.mlp_dim
        self.ff_1 = nn.Dense(self.config.mlp_dim)
        self.ff_2 = nn.Dense(self.config.d_model)

    def __call__(self, x):
        ff_1_out = self.ff_1(x)
        post_act = (
            nn.gelu(ff_1_out) if self.config.activation == "gelu" else nn.relu(ff_1_out)
        )
        out = self.ff_2(post_act)
        return out


if MAIN:
    key1, key2 = random.split(random.PRNGKey(0))
    x = random.normal(key1, (2, 5, 8))
    config = FlaxGPTConfig(8, 2, 2, 10, 10)
    mlp_module = FlaxGPTMLP(config)
    mlp_module_params = mlp_module.init(key2, x)
    out = mlp_module.apply(mlp_module_params, x)
    print(out.shape)
    print(out)
# %%
class ResidualAndLayerNormConnection(nn.Module):
    """Residual connection with layer normalization applied before the inner
    module is applied.

    Args:
        config: GPTConfig object containing the model configuration
        inner_module: Module wrapped by the residual and layer norm connection
    """

    config: FlaxGPTConfig
    inner_module: Union[FlaxGPTAttention, FlaxGPTMLP]

    def setup(self):
        self.norm = nn.LayerNorm(self.config.layer_norm_eps)

    def __call__(self, x):
        return x + self.inner_module(self.norm(x))


if MAIN:
    key1, key2 = random.split(random.PRNGKey(0))
    x = random.normal(key1, (2, 5, 8))
    config = FlaxGPTConfig(8, 2, 2, 10, 10)
    attn_module = FlaxGPTAttention(config)
    mlp_module = FlaxGPTMLP(config)
    attn_norm_module = ResidualAndLayerNormConnection(config, attn_module)
    mlp_norm_module = ResidualAndLayerNormConnection(config, mlp_module)
    attn_norm_module_params = attn_norm_module.init(key2, x)
    mlp_norm_module_params = mlp_norm_module.init(key2, x)
    out_attn = attn_norm_module.apply(attn_norm_module_params, x)
    out_mlp = mlp_norm_module.apply(mlp_norm_module_params, x)
    print(out_attn.shape)
    print(out_attn)
    print(out_mlp.shape)
    print(out_mlp)
# %%
class FlaxGPTBlock(nn.Module):
    """GPT-style transformer block.

    Args:
        config: GPTConfig object containing the model configuration
    """

    config: FlaxGPTConfig

    def setup(self):
        self.attn = ResidualAndLayerNormConnection(
            self.config,
            FlaxGPTAttention(self.config),
        )
        if not self.config.attn_only:
            self.mlp = ResidualAndLayerNormConnection(
                self.config,
                FlaxGPTMLP(self.config),
            )

    def __call__(self, x):
        post_attn = self.attn(x)
        if self.config.attn_only:
            return post_attn
        return self.mlp(post_attn)


if MAIN:
    key1, key2 = random.split(random.PRNGKey(0))
    x = random.normal(key1, (2, 5, 8))
    config = FlaxGPTConfig(8, 2, 2, 10, 10)
    block_module = FlaxGPTBlock(config)
    block_module_params = block_module.init(key2, x)
    out = block_module.apply(block_module_params, x)
    print(out.shape)
    print(out)
# %%
class FlaxGPT(nn.Module):
    """GPT-style transformer model.

    Args:
        config: GPTConfig object containing the model configuration
    """

    config: FlaxGPTConfig

    def setup(self):
        self.tok_embed = nn.Embed(self.config.d_vocab, self.config.d_model)
        self.pos_embed = nn.Embed(self.config.n_ctx, self.config.d_model)
        self.blocks = nn.Sequential(
            [FlaxGPTBlock(self.config) for _ in range(self.config.n_layers)],
        )

    def __call__(self, x):
        _, seq = x.shape
        embed = self.tok_embed(x) + self.pos_embed(jnp.arange(seq))
        post_blocks = self.blocks(embed)
        return post_blocks


if MAIN:
    key1, key2 = random.split(random.PRNGKey(0))
    x = random.randint(key1, (2, 5), 0, 10)
    config = FlaxGPTConfig(8, 2, 2, 10, 10, 12)
    gpt_module = FlaxGPT(config)
    gpt_module_params = gpt_module.init(key2, x)
    out = gpt_module.apply(gpt_module_params, x)
    print(out.shape)
    print(out)

# %%
class FlaxGPTLM(nn.Module):
    """GPT-style transformer language model.

    Args:
        config: GPTConfig object containing the model configuration
    """

    config: FlaxGPTConfig

    def setup(self):
        self.gpt = FlaxGPT(self.config)
        self.ln_final = nn.LayerNorm(self.config.layer_norm_eps)
        self.lm_head = nn.Dense(self.config.d_vocab_out)

    def __call__(self, x):
        x = self.gpt(x)
        x = self.ln_final(x)
        return self.lm_head(x)


if MAIN:
    key1, key2 = random.split(random.PRNGKey(0))
    x = random.randint(key1, (2, 5), 0, 10)
    config = FlaxGPTConfig(8, 2, 2, 10, 10, 12)
    gpt_lm_module = FlaxGPTLM(config)
    gpt_lm_module_params = gpt_lm_module.init(key2, x)
    out = gpt_lm_module.apply(gpt_lm_module_params, x)
    print(out.shape)
    print(out)
# %%
