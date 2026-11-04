# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Performance patches for sequence_layers JAX self-attention.

When streaming, self-attention projects the new frame to Q/K/V and appends
K/V to the KV cache with a concatenate. On GPU, XLA fuses the key projection
(a batch-1 matvec) into that concatenate, which emits a poorly parallelized
kernel: on an RTX 3090 the 3072x3072 projection in mrt2_base takes ~650us
inside the fusion versus ~45us as a standalone reduce fusion, ~13ms per step
across the 20 temporal layers.

An optimization_barrier on the get_qkv outputs keeps the projections out of
the concatenate fusion. Values are unchanged up to floating-point reduction
order.
"""

import jax
from sequence_layers.jax.attention import dot_product_self_attention
from sequence_layers.jax.attention import local_dot_product_self_attention

_PATCHED_CLASSES = (
    dot_product_self_attention.DotProductSelfAttention,
    local_dot_product_self_attention.LocalDotProductSelfAttention,
)
_ORIGINAL_GET_QKV = {}


def install() -> None:
  """Wraps get_qkv in an optimization_barrier. Idempotent."""
  for cls in _PATCHED_CLASSES:
    if cls in _ORIGINAL_GET_QKV:
      continue
    original = cls.get_qkv
    _ORIGINAL_GET_QKV[cls] = original

    def get_qkv(self, *args, _original=original, **kwargs):
      return jax.lax.optimization_barrier(_original(self, *args, **kwargs))

    cls.get_qkv = get_qkv


def uninstall() -> None:
  """Restores the original get_qkv methods."""
  for cls, original in _ORIGINAL_GET_QKV.items():
    cls.get_qkv = original
  _ORIGINAL_GET_QKV.clear()
