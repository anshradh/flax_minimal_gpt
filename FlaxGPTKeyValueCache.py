# %%
import jax.numpy as jnp
import jax
from dataclasses import dataclass
from FlaxGPTConfig import FlaxGPTConfig
from typing import List


@dataclass
class FlaxGPTKeyValueCacheEntry:
    """Data class for storing past keys and values for an attention layer.

    Args:
        keys: jnp.ndarray of shape (batch, num_heads, seq_len, d_head)
        values: jnp.ndarray of shape (batch, num_heads, seq_len, d_head)
    """

    keys: jnp.ndarray
    values: jnp.ndarray

    def update_and_return_keys_and_values(self, new_keys, new_values):
        self.keys = jnp.concatenate([self.keys, new_keys], axis=-2)
        self.values = jnp.concatenate([self.values, new_values], axis=-2)
        return self.keys, self.values


@dataclass
class FlaxGPTKeyValueCache:
    """Class storing past key and value vectors for a GPT model."""

    entries: List[FlaxGPTKeyValueCacheEntry]

    @classmethod
    def init_cache(cls, config: FlaxGPTConfig, batch_size: int = 1):
        return cls(
            entries=[
                FlaxGPTKeyValueCacheEntry(
                    keys=jnp.zeros((batch_size, config.n_heads, 0, config.d_head)),
                    values=jnp.zeros((batch_size, config.n_heads, 0, config.d_head)),
                )
                for _ in range(config.n_layers)
            ],
        )

    def __getitem__(self, idx):
        return self.entries[idx]


# %%
