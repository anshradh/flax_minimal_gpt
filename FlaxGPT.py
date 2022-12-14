# %%
import jax
from jax import random, numpy as jnp
import flax.linen as nn
from typing import Optional, Union
from dataclasses import dataclass
from einops import rearrange
from fancy_einsum import einsum
from transformers import GPT2PreTrainedModel, GPT2TokenizerFast, GPT2LMHeadModel
from flax.core import FrozenDict
from FlaxGPTConfig import FlaxGPTConfig
from FlaxGPTKeyValueCache import FlaxGPTKeyValueCache, FlaxGPTKeyValueCacheEntry

MAIN = __name__ == "__main__"

# %%
class FlaxGPTAttention(nn.Module):
    """Attention layer for GPT-style transformer models.

    Args:
        config: GPTConfig object containing the model configuration
    """

    config: FlaxGPTConfig

    def setup(self):
        self.qkv_proj = nn.Dense(self.config.d_model * 3)
        self.out_proj = nn.Dense(self.config.d_model)

    def __call__(
        self, x, cache: Optional[FlaxGPTKeyValueCacheEntry] = None, padding_mask=None
    ):
        seq_len = x.shape[-2]
        qkv = self.qkv_proj(x)
        q_init, k_init, v_init = jnp.array_split(
            qkv,
            3,
            axis=-1,
        )
        q = rearrange(
            q_init,
            "... seq (n_heads d_head) -> ... n_heads seq d_head",
            n_heads=self.config.n_heads,
        )
        k = rearrange(
            k_init,
            "... seq (n_heads d_head) -> ... n_heads seq d_head",
            n_heads=self.config.n_heads,
        )
        v = rearrange(
            v_init,
            "... seq (n_heads d_head) -> ... n_heads seq d_head",
            n_heads=self.config.n_heads,
        )
        if cache is not None:
            past_seq_len = cache.keys.shape[-2]
            if past_seq_len != 0:
                assert seq_len == 1

            k, v = cache.update_and_return_keys_and_values(k, v)
        else:
            past_seq_len = 0
        attn = jax.vmap(lambda q, k: jnp.einsum("... q h, ... k h-> ... q k", q, k,),)(
            q,
            k,
        ) / jnp.sqrt(self.config.d_head)
        if self.config.bidirectional:
            assert padding_mask is not None
            padding_mask = jnp.where(padding_mask, 0, float("-inf"))
            masked_attn = attn + padding_mask
        else:
            masked_attn = jnp.where(
                jnp.arange(past_seq_len, past_seq_len + seq_len)[:, None]
                >= jnp.arange(past_seq_len + seq_len)[None, :],
                attn,
                float("-inf"),
            )
        softmaxed_attn = nn.softmax(masked_attn, -1)
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
            "... n_heads seq d_head -> ... seq (n_heads d_head)",
        )
        out = self.out_proj(combined_values_rearranged)
        return out


if MAIN:
    key1, key2 = random.split(random.PRNGKey(0))
    x = random.normal(key1, (2, 5, 8))
    config = FlaxGPTConfig(8, 2, 2, 10, 10)
    attn_module = FlaxGPTAttention(config)
    cache_entry = FlaxGPTKeyValueCacheEntry(
        keys=jnp.empty((2, 2, 0, 4)),
        values=jnp.empty((2, 2, 0, 4)),
    )
    attn_module_params = attn_module.init(key2, x)
    jit_attn_module_apply = jax.jit(attn_module.apply)
    out = jit_attn_module_apply(attn_module_params, x, cache_entry)
    assert out.shape == x.shape
    x_n = random.normal(key1, (2, 1, 8))
    out_n = jit_attn_module_apply(attn_module_params, x_n, cache_entry)
    assert out_n.shape == x_n.shape
    bidirectional_config = FlaxGPTConfig(8, 2, 2, 10, 10, bidirectional=True)
    bidirectional_attn_module = FlaxGPTAttention(bidirectional_config)
    padding_mask = rearrange(
        jnp.array(
            [
                [True, True, True, False, False],
                [True, True, False, False, False],
            ],
        ),
        "batch seq -> batch 1 seq 1",
    )
    bidirectional_attn_module_params = bidirectional_attn_module.init(
        key2,
        x,
        None,
        padding_mask,
    )
    jit_bidirectional_attn_module_apply = jax.jit(bidirectional_attn_module.apply)
    out = jit_bidirectional_attn_module_apply(
        bidirectional_attn_module_params,
        x,
        None,
        padding_mask,
    )
    assert out.shape == x.shape

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
    jit_mlp_module_apply = jax.jit(mlp_module.apply)
    out = jit_mlp_module_apply(mlp_module_params, x)
    assert out.shape == x.shape
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

    def __call__(
        self, x, cache: Optional[FlaxGPTKeyValueCacheEntry] = None, padding_mask=None
    ):
        if isinstance(self.inner_module, FlaxGPTAttention):
            return x + self.inner_module(self.norm(x), cache, padding_mask)
        return x + self.inner_module(self.norm(x))


if MAIN:
    key1, key2 = random.split(random.PRNGKey(0))
    x = random.normal(key1, (2, 5, 16))
    config = FlaxGPTConfig(16, 2, 2, 10, 10)
    attn_module = FlaxGPTAttention(config)
    mlp_module = FlaxGPTMLP(config)
    attn_norm_module = ResidualAndLayerNormConnection(config, attn_module)
    mlp_norm_module = ResidualAndLayerNormConnection(config, mlp_module)
    cache_entry = FlaxGPTKeyValueCacheEntry(
        keys=jnp.empty((2, 2, 0, 8)),
        values=jnp.empty((2, 2, 0, 8)),
    )
    attn_norm_module_params = attn_norm_module.init(key2, x)
    mlp_norm_module_params = mlp_norm_module.init(key2, x)
    jit_attn_norm_module_apply = jax.jit(attn_norm_module.apply)
    jit_mlp_norm_module_apply = jax.jit(mlp_norm_module.apply)
    out_attn = jit_attn_norm_module_apply(attn_norm_module_params, x, cache_entry)
    out_mlp = jit_mlp_norm_module_apply(mlp_norm_module_params, x)
    assert out_attn.shape == x.shape
    assert out_mlp.shape == x.shape
    x_n = random.normal(key1, (2, 1, 16))
    out_attn_n = jit_attn_norm_module_apply(attn_norm_module_params, x_n, cache_entry)
    assert out_attn_n.shape == x_n.shape
    bidirectional_config = FlaxGPTConfig(16, 2, 2, 10, 10, bidirectional=True)
    bidirectional_attn_module = FlaxGPTAttention(bidirectional_config)
    bidirectional_attn_norm_module = ResidualAndLayerNormConnection(
        bidirectional_config,
        bidirectional_attn_module,
    )
    padding_mask = rearrange(
        jnp.array(
            [
                [True, True, True, False, False],
                [True, True, False, False, False],
            ],
        ),
        "batch seq -> batch 1 seq 1",
    )
    bidirectional_attn_norm_module_params = bidirectional_attn_norm_module.init(
        key2,
        x,
        None,
        padding_mask,
    )
    jit_bidirectional_attn_norm_module_apply = jax.jit(
        bidirectional_attn_norm_module.apply,
    )
    out = jit_bidirectional_attn_norm_module_apply(
        bidirectional_attn_norm_module_params,
        x,
        None,
        padding_mask,
    )
    assert out.shape == x.shape
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

    def __call__(
        self, x, cache: Optional[FlaxGPTKeyValueCacheEntry] = None, padding_mask=None
    ):
        post_attn = self.attn(x, cache, padding_mask)
        if self.config.attn_only:
            return post_attn
        return self.mlp(post_attn)


if MAIN:
    key1, key2 = random.split(random.PRNGKey(0))
    x = random.normal(key1, (2, 5, 32))
    config = FlaxGPTConfig(32, 2, 2, 10, 10)
    block_module = FlaxGPTBlock(config)
    block_module_params = block_module.init(key2, x)
    cache_entry = FlaxGPTKeyValueCacheEntry(
        keys=jnp.empty((2, 2, 0, 16)),
        values=jnp.empty((2, 2, 0, 16)),
    )
    jit_block_module_apply = jax.jit(block_module.apply)
    out = jit_block_module_apply(block_module_params, x, cache_entry)
    assert out.shape == x.shape
    x_n = random.normal(key1, (2, 1, 32))
    out_n = jit_block_module_apply(block_module_params, x_n, cache_entry)
    assert out_n.shape == x_n.shape
    bidirectional_config = FlaxGPTConfig(32, 2, 2, 10, 10, bidirectional=True)
    bidirectional_block_module = FlaxGPTBlock(bidirectional_config)
    padding_mask = rearrange(
        jnp.array(
            [
                [True, True, True, False, False],
                [True, True, False, False, False],
            ],
        ),
        "batch seq -> batch 1 seq 1",
    )
    bidirectional_block_module_params = bidirectional_block_module.init(
        key2,
        x,
        None,
        padding_mask,
    )
    jit_bidirectional_block_module_apply = jax.jit(
        bidirectional_block_module.apply,
    )
    out = jit_bidirectional_block_module_apply(
        bidirectional_block_module_params,
        x,
        None,
        padding_mask,
    )
    assert out.shape == x.shape

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
        self.blocks = [FlaxGPTBlock(self.config) for _ in range(self.config.n_layers)]

    def __call__(
        self, x, cache: Optional[FlaxGPTKeyValueCache] = None, padding_mask=None
    ):
        _, seq = x.shape
        x = self.tok_embed(x) + self.pos_embed(jnp.arange(seq))
        for i, block in enumerate(self.blocks):
            if cache is not None:
                x = block(x, cache[i], padding_mask)
            else:
                x = block(x, None, padding_mask)
        return x


if MAIN:
    key1, key2 = random.split(random.PRNGKey(0))
    x = random.randint(key1, (2, 5), 0, 10)
    config = FlaxGPTConfig(24, 2, 2, 10, 10, 12)
    gpt_module = FlaxGPT(config)
    gpt_module_params = gpt_module.init(key2, x)
    jit_gpt_module_apply = jax.jit(gpt_module.apply)
    cache = FlaxGPTKeyValueCache.init_cache(config, 2)
    out = jit_gpt_module_apply(gpt_module_params, x, cache)
    assert out.shape == x.shape + (config.d_model,)
    x_n = random.randint(key1, (2, 1), 0, 10)
    out_n = jit_gpt_module_apply(gpt_module_params, x_n, cache)
    assert out_n.shape == x_n.shape + (config.d_model,)
    bidirectional_config = FlaxGPTConfig(
        24,
        2,
        2,
        10,
        10,
        12,
        bidirectional=True,
    )
    bidirectional_gpt_module = FlaxGPT(bidirectional_config)
    padding_mask = rearrange(
        jnp.array(
            [
                [True, True, True, False, False],
                [True, True, False, False, False],
            ],
        ),
        "batch seq -> batch 1 seq 1",
    )
    bidirectional_gpt_module_params = bidirectional_gpt_module.init(
        key2,
        x,
        None,
        padding_mask,
    )
    jit_bidirectional_gpt_module_apply = jax.jit(
        bidirectional_gpt_module.apply,
    )
    out = jit_bidirectional_gpt_module_apply(
        bidirectional_gpt_module_params,
        x,
        None,
        padding_mask,
    )
    assert out.shape == x.shape + (bidirectional_config.d_model,)

# %%
class FlaxGPTLM(nn.Module):
    """GPT-style transformer language model.

    Args:
        config: GPTConfig object containing the model configuration
    """

    config: FlaxGPTConfig

    def __post__init__(self):
        assert (
            self.config.bidirectional is False
        ), "bidirectional not allowed for language modeling"
        super().__post__init__()

    def setup(self):
        self.gpt = FlaxGPT(self.config)
        self.ln_final = nn.LayerNorm(self.config.layer_norm_eps)
        self.lm_head = nn.Dense(self.config.d_vocab_out)

    def __call__(self, x, cache: Optional[FlaxGPTKeyValueCache] = None):
        post_base = self.gpt(x, cache)
        post_final_ln = self.ln_final(post_base)
        return self.lm_head(post_final_ln)


if MAIN:
    key1, key2 = random.split(random.PRNGKey(0))
    x = random.randint(key1, (10, 10), 0, 10)
    config = FlaxGPTConfig(16, 2, 2, 10, 10)
    gpt_lm_module = FlaxGPTLM(config)
    gpt_lm_module_params = gpt_lm_module.init(key2, x)
    jit_gpt_lm_module_apply = jax.jit(gpt_lm_module.apply)
    cache = FlaxGPTKeyValueCache.init_cache(config, 10)
    out = jit_gpt_lm_module_apply(gpt_lm_module_params, x, cache)
    assert out.shape == x.shape + (config.d_vocab_out,)
    x_n = random.randint(key1, (10, 1), 0, 10)
    out_n = jit_gpt_lm_module_apply(gpt_lm_module_params, x_n, cache)
    assert out_n.shape == x_n.shape + (config.d_vocab_out,)
    bidirectional_config = FlaxGPTConfig(
        16,
        2,
        2,
        10,
        10,
        bidirectional=True,
    )
    try:
        bidirectional_gpt_lm_module = FlaxGPTLM(bidirectional_config)
    except AssertionError:
        pass
# %%
class FlaxGPTClasifier(nn.Module):
    """GPT-style transformer classifier. Returns logits at the first sequence position.

    Args:
        config: GPTConfig object containing the model configuration
    """

    config: FlaxGPTConfig

    def __post__init__(self):
        assert self.config.bidirectional, "bidirectional requires for classification"
        super().__post__init__()

    def setup(self):
        self.gpt = FlaxGPT(self.config)
        self.ln_final = nn.LayerNorm(self.config.layer_norm_eps)
        self.classifier_head = nn.Dense(self.config.n_classes)

    def __call__(self, x, padding_mask):
        post_base = self.gpt(x, padding_mask=padding_mask)
        post_final_ln = self.ln_final(post_base)
        return self.classifier_head(post_final_ln)[:, 0]


if MAIN:
    key1, key2 = random.split(random.PRNGKey(0))
    x = random.randint(key1, (2, 5), 0, 10)
    config = FlaxGPTConfig(16, 2, 2, 10, 10, bidirectional=True, n_classes=2)
    gpt_classifier_module = FlaxGPTClasifier(config)
    padding_mask = rearrange(
        jnp.array(
            [
                [True, True, True, False, False],
                [True, True, False, False, False],
            ],
        ),
        "batch seq -> batch 1 seq 1",
    )
    gpt_classifier_module_params = gpt_classifier_module.init(key2, x, padding_mask)
    jit_gpt_classifier_module_apply = jax.jit(gpt_classifier_module.apply)
    out = jit_gpt_classifier_module_apply(gpt_classifier_module_params, x, padding_mask)
    assert out.shape == (x.shape[0], config.n_classes)
    unidirectional_config = FlaxGPTConfig(
        16,
        2,
        2,
        10,
        10,
    )
    try:
        unidirectional_gpt_classifier_module = FlaxGPTClasifier(unidirectional_config)
    except AssertionError:
        pass

# %%
# Test same output as Tranformers Model
if MAIN:
    gpt2_small_hf = GPT2LMHeadModel.from_pretrained("gpt2")
    gpt2_small_hf.eval()
    gpt2_small_hf.to("cpu")
    gpt2_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
    gpt2_tokenizer.pad_token_id = gpt2_tokenizer.eos_token_id
    pt_tokenized = gpt2_tokenizer(
        ["The dog caught the ball and", "The cat caught the mouse and"],
        padding=True,
        return_tensors="pt",
    )
    jnp_tokenized = gpt2_tokenizer(
        ["The dog caught the ball and", "The cat caught the mouse and"],
        padding=True,
        return_tensors="jax",
    )
    hf_text = gpt2_tokenizer.batch_decode(
        gpt2_small_hf(pt_tokenized["input_ids"])["logits"].argmax(-1)
    )
# %%
def to_frozen(x):
    if hasattr(x, "keys"):
        return FrozenDict({k: to_frozen(v) for k, v in x.items()})
    return x


# %%
if MAIN:
    key = random.PRNGKey(0)
    flax_gpt2_small_config = FlaxGPTConfig(768, 12, 12, 1024, 50257)
    flax_gpt2_small = FlaxGPTLM(flax_gpt2_small_config)
    flax_gpt2_small_params = jax.jit(flax_gpt2_small.init)(
        key, jnp_tokenized["input_ids"]
    )
    jit_flax_gpt2_small_apply = jax.jit(flax_gpt2_small.apply)
# %%
if MAIN:
    new_flax_gpt2_small_params = dict(
        params=dict(
            gpt=dict(
                tok_embed=dict(),
                pos_embed=dict(),
                blocks_0=dict(
                    attn=dict(
                        norm=dict(),
                        inner_module=dict(
                            out_proj=dict(),
                            qkv_proj=dict(),
                        ),
                    ),
                    mlp=dict(
                        norm=dict(),
                        inner_module=dict(
                            ff_1=dict(),
                            ff_2=dict(),
                        ),
                    ),
                ),
                blocks_1=dict(
                    attn=dict(
                        norm=dict(),
                        inner_module=dict(
                            out_proj=dict(),
                            qkv_proj=dict(),
                        ),
                    ),
                    mlp=dict(
                        norm=dict(),
                        inner_module=dict(
                            ff_1=dict(),
                            ff_2=dict(),
                        ),
                    ),
                ),
                blocks_2=dict(
                    attn=dict(
                        norm=dict(),
                        inner_module=dict(
                            out_proj=dict(),
                            qkv_proj=dict(),
                        ),
                    ),
                    mlp=dict(
                        norm=dict(),
                        inner_module=dict(
                            ff_1=dict(),
                            ff_2=dict(),
                        ),
                    ),
                ),
                blocks_3=dict(
                    attn=dict(
                        norm=dict(),
                        inner_module=dict(
                            out_proj=dict(),
                            qkv_proj=dict(),
                        ),
                    ),
                    mlp=dict(
                        norm=dict(),
                        inner_module=dict(
                            ff_1=dict(),
                            ff_2=dict(),
                        ),
                    ),
                ),
                blocks_4=dict(
                    attn=dict(
                        norm=dict(),
                        inner_module=dict(
                            out_proj=dict(),
                            qkv_proj=dict(),
                        ),
                    ),
                    mlp=dict(
                        norm=dict(),
                        inner_module=dict(
                            ff_1=dict(),
                            ff_2=dict(),
                        ),
                    ),
                ),
                blocks_5=dict(
                    attn=dict(
                        norm=dict(),
                        inner_module=dict(
                            out_proj=dict(),
                            qkv_proj=dict(),
                        ),
                    ),
                    mlp=dict(
                        norm=dict(),
                        inner_module=dict(
                            ff_1=dict(),
                            ff_2=dict(),
                        ),
                    ),
                ),
                blocks_6=dict(
                    attn=dict(
                        norm=dict(),
                        inner_module=dict(
                            out_proj=dict(),
                            qkv_proj=dict(),
                        ),
                    ),
                    mlp=dict(
                        norm=dict(),
                        inner_module=dict(
                            ff_1=dict(),
                            ff_2=dict(),
                        ),
                    ),
                ),
                blocks_7=dict(
                    attn=dict(
                        norm=dict(),
                        inner_module=dict(
                            out_proj=dict(),
                            qkv_proj=dict(),
                        ),
                    ),
                    mlp=dict(
                        norm=dict(),
                        inner_module=dict(
                            ff_1=dict(),
                            ff_2=dict(),
                        ),
                    ),
                ),
                blocks_8=dict(
                    attn=dict(
                        norm=dict(),
                        inner_module=dict(
                            out_proj=dict(),
                            qkv_proj=dict(),
                        ),
                    ),
                    mlp=dict(
                        norm=dict(),
                        inner_module=dict(
                            ff_1=dict(),
                            ff_2=dict(),
                        ),
                    ),
                ),
                blocks_9=dict(
                    attn=dict(
                        norm=dict(),
                        inner_module=dict(
                            out_proj=dict(),
                            qkv_proj=dict(),
                        ),
                    ),
                    mlp=dict(
                        norm=dict(),
                        inner_module=dict(
                            ff_1=dict(),
                            ff_2=dict(),
                        ),
                    ),
                ),
                blocks_10=dict(
                    attn=dict(
                        norm=dict(),
                        inner_module=dict(
                            out_proj=dict(),
                            qkv_proj=dict(),
                        ),
                    ),
                    mlp=dict(
                        norm=dict(),
                        inner_module=dict(
                            ff_1=dict(),
                            ff_2=dict(),
                        ),
                    ),
                ),
                blocks_11=dict(
                    attn=dict(
                        norm=dict(),
                        inner_module=dict(
                            out_proj=dict(),
                            qkv_proj=dict(),
                        ),
                    ),
                    mlp=dict(
                        norm=dict(),
                        inner_module=dict(
                            ff_1=dict(),
                            ff_2=dict(),
                        ),
                    ),
                ),
            ),
            ln_final=dict(),
            lm_head=dict(),
        ),
    )
    for param_name, param in gpt2_small_hf.state_dict().items():
        if param_name == "transformer.wte.weight":
            new_flax_gpt2_small_params["params"]["gpt"]["tok_embed"][
                "embedding"
            ] = jnp.array(
                param.cpu().numpy(),
            )
        if param_name == "transformer.wpe.weight":
            new_flax_gpt2_small_params["params"]["gpt"]["pos_embed"][
                "embedding"
            ] = jnp.array(
                param.cpu().numpy(),
            )
        if param_name.split(".")[1] == "h":
            block_num = int(param_name.split(".")[2])
            if "ln_1.weight" in param_name:
                new_flax_gpt2_small_params["params"]["gpt"][f"blocks_{block_num}"][
                    "attn"
                ]["norm"]["scale"] = jnp.array(
                    param.cpu().numpy(),
                )
            if "ln_1.bias" in param_name:
                new_flax_gpt2_small_params["params"]["gpt"][f"blocks_{block_num}"][
                    "attn"
                ]["norm"]["bias"] = jnp.array(
                    param.cpu().numpy(),
                )
            if "attn.c_attn.weight" in param_name:
                new_flax_gpt2_small_params["params"]["gpt"][f"blocks_{block_num}"][
                    "attn"
                ]["inner_module"]["qkv_proj"]["kernel"] = jnp.array(
                    param.cpu().numpy(),
                )
            if "attn.c_attn.bias" in param_name:
                new_flax_gpt2_small_params["params"]["gpt"][f"blocks_{block_num}"][
                    "attn"
                ]["inner_module"]["qkv_proj"]["bias"] = jnp.array(param.cpu().numpy())
            if "attn.c_proj.weight" in param_name:
                new_flax_gpt2_small_params["params"]["gpt"][f"blocks_{block_num}"][
                    "attn"
                ]["inner_module"]["out_proj"]["kernel"] = jnp.array(
                    param.cpu().numpy(),
                )
            if "attn.c_proj.bias" in param_name:
                new_flax_gpt2_small_params["params"]["gpt"][f"blocks_{block_num}"][
                    "attn"
                ]["inner_module"]["out_proj"]["bias"] = jnp.array(
                    param.cpu().numpy(),
                )
            if "ln_2.weight" in param_name:
                new_flax_gpt2_small_params["params"]["gpt"][f"blocks_{block_num}"][
                    "mlp"
                ]["norm"]["scale"] = jnp.array(
                    param.cpu().numpy(),
                )
            if "ln_2.bias" in param_name:
                new_flax_gpt2_small_params["params"]["gpt"][f"blocks_{block_num}"][
                    "mlp"
                ]["norm"]["bias"] = jnp.array(
                    param.cpu().numpy(),
                )
            if "mlp.c_fc.weight" in param_name:
                new_flax_gpt2_small_params["params"]["gpt"][f"blocks_{block_num}"][
                    "mlp"
                ]["inner_module"]["ff_1"]["kernel"] = jnp.array(
                    param.cpu().numpy(),
                )
            if "mlp.c_fc.bias" in param_name:
                new_flax_gpt2_small_params["params"]["gpt"][f"blocks_{block_num}"][
                    "mlp"
                ]["inner_module"]["ff_1"]["bias"] = jnp.array(
                    param.cpu().numpy(),
                )
            if "mlp.c_proj.weight" in param_name:
                new_flax_gpt2_small_params["params"]["gpt"][f"blocks_{block_num}"][
                    "mlp"
                ]["inner_module"]["ff_2"]["kernel"] = jnp.array(
                    param.cpu().numpy(),
                )
            if "mlp.c_proj.bias" in param_name:
                new_flax_gpt2_small_params["params"]["gpt"][f"blocks_{block_num}"][
                    "mlp"
                ]["inner_module"]["ff_2"]["bias"] = jnp.array(
                    param.cpu().numpy(),
                )
        if "ln_f.weight" in param_name:
            new_flax_gpt2_small_params["params"]["ln_final"]["scale"] = jnp.array(
                param.cpu().numpy(),
            )
        if "ln_f.bias" in param_name:
            new_flax_gpt2_small_params["params"]["ln_final"]["bias"] = jnp.array(
                param.cpu().numpy(),
            )
        if "lm_head.weight" in param_name:
            new_flax_gpt2_small_params["params"]["lm_head"]["kernel"] = jnp.array(
                param.cpu().numpy(),
            ).T
    new_flax_gpt2_small_params["params"]["lm_head"]["bias"] = jnp.zeros_like(
        flax_gpt2_small_params["params"]["lm_head"]["bias"],
    )
    frozen_params = to_frozen(new_flax_gpt2_small_params)
# %%
if MAIN:
    cache = FlaxGPTKeyValueCache.init_cache(flax_gpt2_small_config, 2)
    jax_outputs = jit_flax_gpt2_small_apply(
        frozen_params,
        jnp_tokenized["input_ids"],
        cache,
    )
    jax_ids = jnp.argmax(jax_outputs, axis=-1)
    jax_text = gpt2_tokenizer.batch_decode(jax_ids, skip_special_tokens=True)
    assert jax_text == hf_text
# %%
