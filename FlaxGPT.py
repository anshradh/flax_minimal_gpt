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
class GPTConfig:
    d_model: int
    n_heads: int
    n_layers: int
    d_vocab: int
    n_ctx: int
    d_vocab_out: Optional[int] = None
    layer_norm_eps: float = 1e-5
    attn_only: bool = False
    mlp_dim: Optional[int] = None
    activation: Optional[str] = None

    def __post_init__(self):
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
    config = GPTConfig(768, 12, 12, 50257, 1024)
    print(config)
# %%
class InnerGPTAttention(nn.Module):
    config: GPTConfig

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
        attn = jax.vmap(
            lambda q, k: jnp.einsum(
                "... q h, ... k h-> ... q k",
                q,
                k,
            ),
        )(q, k)
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


class GPTAttention(nn.Module):
    config: GPTConfig

    @nn.compact
    def __call__(self, x):
        return nn.vmap(
            InnerGPTAttention,
            in_axes=0,
            out_axes=0,
            variable_axes=dict(params=None),
            split_rngs=dict(params=False),
        )(config)(x)


if MAIN:
    key1, key2 = random.split(random.PRNGKey(0))
    x = random.normal(key1, (2, 5, 8))
    config = GPTConfig(8, 2, 2, 10, 10)
    attn_module = GPTAttention(config)
    attn_module_params = attn_module.init(key2, x)
    out = attn_module.apply(attn_module_params, x)
    print(out.shape)
    print(out)
# %%
class GPTMLP(nn.Module):
    config: GPTConfig

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
    config = GPTConfig(8, 2, 2, 10, 10)
    mlp_module = GPTMLP(config)
    mlp_module_params = mlp_module.init(key2, x)
    out = mlp_module.apply(mlp_module_params, x)
    print(out.shape)
    print(out)
# %%
class ResidualAndLayerNormConnection(nn.Module):
    config: GPTConfig
    inner_module: Union[GPTAttention, GPTMLP]

    def setup(self):
        self.norm = nn.LayerNorm(self.config.layer_norm_eps)

    def __call__(self, x):
        return x + self.inner_module(self.norm(x))


if MAIN:
    key1, key2 = random.split(random.PRNGKey(0))
    x = random.normal(key1, (2, 5, 8))
    config = GPTConfig(8, 2, 2, 10, 10)
    attn_module = GPTAttention(config)
    mlp_module = GPTMLP(config)
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
class GPTBlock(nn.Module):
    config: GPTConfig

    def setup(self):
        self.attn = ResidualAndLayerNormConnection(
            self.config, GPTAttention(self.config)
        )
        self.mlp = ResidualAndLayerNormConnection(self.config, GPTMLP(self.config))

    def __call__(self, x):
        return self.mlp(self.attn(x))


if MAIN:
    key1, key2 = random.split(random.PRNGKey(0))
    x = random.normal(key1, (2, 5, 8))
    config = GPTConfig(8, 2, 2, 10, 10)
    block_module = GPTBlock(config)
    block_module_params = block_module.init(key2, x)
    out = block_module.apply(block_module_params, x)
    print(out.shape)
    print(out)
# %%
class GPT(nn.Module):
    config: GPTConfig

    def setup(self):
        self.tok_embed = nn.Embed(self.config.d_vocab, self.config.d_model)
        self.pos_embed = nn.Embed(self.config.n_ctx, self.config.d_model)
        self.blocks = [GPTBlock(self.config) for _ in range(self.config.n_layers)]

    def __call__(self, x):
        _, seq = x.shape
        x = self.tok_embed(x) + self.pos_embed(jnp.arange(seq))
        for block in self.blocks:
            x = block(x)
        return x


if MAIN:
    key1, key2 = random.split(random.PRNGKey(0))
    x = random.randint(key1, (2, 5), 0, 10)
    config = GPTConfig(8, 2, 2, 10, 10, 12)
    gpt_module = GPT(config)
    gpt_module_params = gpt_module.init(key2, x)
    out = gpt_module.apply(gpt_module_params, x)
    print(out.shape)
    print(out)

# %%
class GPTLM(nn.Module):
    config: GPTConfig

    def setup(self):
        self.gpt = GPT(self.config)
        self.ln_final = nn.LayerNorm(self.config.layer_norm_eps)
        self.lm_head = nn.Dense(self.config.d_vocab_out)

    def __call__(self, x):
        x = self.gpt(x)
        x = self.ln_final(x)
        return self.lm_head(x)


if MAIN:
    key1, key2 = random.split(random.PRNGKey(0))
    x = random.randint(key1, (2, 5), 0, 10)
    config = GPTConfig(8, 2, 2, 10, 10, 12)
    gpt_module = GPTLM(config)
    gpt_module_params = gpt_module.init(key2, x)
    out = gpt_module.apply(gpt_module_params, x)
    print(out.shape)
    print(out)
# %%
