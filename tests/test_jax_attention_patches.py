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

"""Checks the JAX self-attention patch leaves streaming outputs unchanged.

Compiles the streaming step with and without the get_qkv optimization_barrier
and runs both from the same warmed-up state.
"""

import functools
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from magenta_rt import musiccoca
from magenta_rt import paths
from magenta_rt.jax import attention_patches
from magenta_rt.jax import system

_MODEL = 'mrt2_small'
_CHECKPOINT = paths.resolve_checkpoint(f'{_MODEL}.safetensors')
_WARMUP_STEPS = 30


@unittest.skipUnless(
    _CHECKPOINT.exists(), f'Checkpoint not found: {_CHECKPOINT}'
)
class AttentionPatchEquivalenceTest(unittest.TestCase):

  def test_step_matches_unpatched(self):
    attention_patches.install()
    mrt = system.MagentaRT2System(
        size=_MODEL, style_model=musiccoca.MockMusicCoCa()
    )
    # Greedy sampling so tiny numeric differences cannot flip sampled tokens.
    block, constants = mrt._build_conditioning({}, None, 1.0, 1)
    patched_step = mrt._jit_streaming_step

    state = mrt._jit_init_state(mrt._params, {})
    for _ in range(_WARMUP_STEPS):
      _, state, _ = patched_step(mrt._params, block, constants, state)

    attention_patches.uninstall()
    try:
      sampler = mrt._sampler
      rngs = {'params': jax.random.PRNGKey(42), 'random': jax.random.PRNGKey(0)}

      @functools.partial(jax.jit, donate_argnums=(3,))
      def unpatched(params, x, constants, state):
        return sampler.apply(
            params, x=x, state=state, constants=constants, training=False,
            rngs=rngs, method=sampler.step_with_emits,
        )

      unpatched_step = unpatched.lower(
          mrt._params, block, constants, state
      ).compile()
    finally:
      attention_patches.install()

    def run(step):
      state_copy = jax.tree.map(lambda x: jnp.array(x, copy=True), state)
      out, new_state, _ = step(mrt._params, block, constants, state_copy)
      return jax.device_get(out.values), jax.device_get(new_state)

    out_patched, state_patched = run(patched_step)
    out_unpatched, state_unpatched = run(unpatched_step)

    np.testing.assert_array_equal(out_patched, out_unpatched)
    for a, b in zip(
        jax.tree.leaves(state_patched), jax.tree.leaves(state_unpatched)
    ):
      a, b = np.asarray(a), np.asarray(b)
      if a.dtype.kind in 'biu':
        np.testing.assert_array_equal(a, b)
      else:
        a, b = a.astype(np.float32), b.astype(np.float32)
        scale = max(float(np.abs(b).max()), 1e-6)
        # bf16 KV-cache entries differ by reduction order only.
        self.assertLess(float(np.abs(a - b).max()) / scale, 2e-2)


if __name__ == '__main__':
  unittest.main()
