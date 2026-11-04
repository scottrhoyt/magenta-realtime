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

"""Tests for `MagentaRT2System.generate` with a compiled frame step.

With `compile` (the default), `generate` runs each frame's sampler step under
`mx.compile` and queues frames ahead. These tests check that it gives the same
audio as the uncompiled loop, sample for sample, through changes of
conditioning, temperature and top-k between chunks; that those changes replay
the trace made in warm-up instead of tracing again; and that a state of the
wrong structure is refused.
"""

import unittest

import numpy as np

import magenta_rt  # noqa: F401 — activates vendored sequence_layers
from magenta_rt import paths
from magenta_rt.config import MUSICCOCA
from magenta_rt.mlx import system
from magenta_rt.mlx.state import flatten_state, unflatten_state

# CI checkpoint: mrt2_small.safetensors (downloaded by the CI workflow).
_CI_CHECKPOINT = 'mrt2_small.safetensors'
_FRAMES = 4


class _NoStyleModel:
  """Stands in for MusicCoCa: the tests pass style tokens, not embeddings."""


class TestCompiledGenerate(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    if not paths.resolve_checkpoint(_CI_CHECKPOINT).exists():
      raise unittest.SkipTest(f'Checkpoint not found: {_CI_CHECKPOINT}')
    cls.mrt = system.MagentaRT2System(
        size='mrt2_small',
        style_model=_NoStyleModel(),
        bits=8,
    )
    cfg = next(
        c for c in cls.mrt._model.input_configs if c.key == MUSICCOCA.key
    )
    n, size = cfg.rvq_truncation_level, cfg.codebook_size
    cls.style_a = [(7 * i) % size for i in range(n)]
    cls.style_b = [(11 * i + 3) % size for i in range(n)]

  def setUp(self):
    self.mrt.compile = True

  def _script(self):
    """Chunks of (generate kwargs): a style change, temperature, top-k, CFG."""
    a = {MUSICCOCA.key: self.style_a}
    b = {MUSICCOCA.key: self.style_b}
    return [
        dict(conditioning=a),
        dict(conditioning=a),
        dict(conditioning=b),
        dict(conditioning=b, temperature=0.9),
        dict(conditioning=b, top_k=20, cfg_scales={'musiccoca': 5.0}),
        dict(conditioning={}),
    ]

  def _run(self, compile_):
    self.mrt.compile = compile_
    state, chunks = None, []
    for kwargs in self._script():
      wav, state = self.mrt.generate(frames=_FRAMES, state=state, **kwargs)
      chunks.append(wav.samples)
    return chunks

  def test_compiled_matches_uncompiled(self):
    compiled = self._run(True)
    uncompiled = self._run(False)
    for i, (c, u) in enumerate(zip(compiled, uncompiled)):
      self.assertEqual(c.shape, (_FRAMES * 1920, 2))
      self.assertTrue(np.array_equal(c, u), f'chunk {i} differs')
    # The conditioning reaches the traced step: a style change changes it.
    self.assertFalse(np.array_equal(compiled[1], compiled[2]))

  def test_chunked_matches_one_call(self):
    cond = {MUSICCOCA.key: self.style_a}
    one, _ = self.mrt.generate(conditioning=cond, frames=3 * _FRAMES)
    state, chunks = None, []
    for _ in range(3):
      wav, state = self.mrt.generate(
          conditioning=cond,
          frames=_FRAMES,
          state=state,
      )
      chunks.append(wav.samples)
    np.testing.assert_array_equal(one.samples, np.concatenate(chunks))

  def test_conditioning_changes_replay_the_trace(self):
    """After warm-up, nothing in the script traces again. Tracing calls the
    sampler's Python `step_with_emits`; replaying the compiled step does not."""
    sampler = self.mrt._sampler
    real = sampler.step_with_emits
    calls = []

    def counting(*args, **kwargs):
      calls.append(1)
      return real(*args, **kwargs)

    sampler.step_with_emits = counting
    try:
      self._run(True)
      replayed = len(calls)
      # Positive control: a new compiled step traces, and the counter sees it.
      saved = dict(self.mrt._compiled_steps)
      self.mrt._compiled_steps.clear()
      try:
        self.mrt.generate(frames=1)
      finally:
        self.mrt._compiled_steps.update(saved)
    finally:
      del sampler.step_with_emits
    self.assertEqual(replayed, 0)
    self.assertTrue(calls, 'a new compiled step traced without being counted')

  def test_state_round_trips_through_flatten(self):
    _, state = self.mrt.generate(frames=1)
    flat, structure = flatten_state(state)
    again, structure_again = flatten_state(unflatten_state(flat, structure))
    self.assertEqual(structure, structure_again)
    self.assertEqual([id(x) for x in flat], [id(x) for x in again])

  def test_state_of_another_structure_is_refused(self):
    _, state = self.mrt.generate(frames=1)
    for compile_ in (True, False):
      self.mrt.compile = compile_
      with self.assertRaises(ValueError):
        self.mrt.generate(frames=1, state=state[:-1])


if __name__ == '__main__':
  unittest.main()
