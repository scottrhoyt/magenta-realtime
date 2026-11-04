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

"""The MLX sampler's streaming state as a flat list of arrays.

The state nests tuples, lists and `sequence_layers` sequences. `mx.compile` and
`.mlxfn` export take arrays and plain containers, so the state is flattened to a
list of arrays and a structure, and rebuilt from them.
"""

import mlx.core as mx
import sequence_layers.mlx as sl


def flatten_state(state):
  """Flatten a nested pytree state into a list of mx.array.

  Handles tuples, lists, and mx.array leaves. Empty tuples contribute
  zero arrays.

  Args:
    state: Nested tuple/list of mx.array.

  Returns:
    (flat_arrays, structure) where structure encodes the nesting.
  """
  flat = []

  def _record(node):
    if isinstance(node, mx.array):
      flat.append(node)
      return 'array'
    elif isinstance(node, tuple):
      children = [_record(child) for child in node]
      return ('tuple', children)
    elif isinstance(node, list):
      children = [_record(child) for child in node]
      return ('list', children)
    elif isinstance(node, dict):
      # We shouldn't hit dicts, if we do we need to handle keys properly.
      raise TypeError(f'Unsupported state node type: {type(node)}')
    elif isinstance(node, sl.MaskedSequence):
      children = [_record(child) for child in [node.values, node.mask]]
      return ('MaskedSequence', children)
    elif isinstance(node, sl.Sequence):
      children = [_record(child) for child in [node.values, node.mask]]
      return ('Sequence', children)
    else:
      raise TypeError(f'Unsupported state node type: {type(node)}')

  structure = _record(state)
  return flat, structure


def unflatten_state(flat, structure):
  """Reconstruct a nested state from a flat array list and structure.

  Args:
    flat: List of mx.array.
    structure: Structure descriptor from flatten_state.

  Returns:
    Nested tuple/list matching the original structure.
  """
  idx = [0]

  def _rebuild(struct):
    if struct == 'array':
      result = flat[idx[0]]
      idx[0] += 1
      return result
    elif isinstance(struct, tuple) and struct[0] == 'tuple':
      return tuple(_rebuild(s) for s in struct[1])
    elif isinstance(struct, tuple) and struct[0] == 'list':
      return [_rebuild(s) for s in struct[1]]
    elif struct[0] == 'MaskedSequence':
      return sl.MaskedSequence(_rebuild(struct[1][0]), _rebuild(struct[1][1]))
    elif struct[0] == 'Sequence':
      return sl.Sequence(_rebuild(struct[1][0]), _rebuild(struct[1][1]))
    else:
      raise ValueError(f'Unknown structure node: {struct}')

  result = _rebuild(structure)
  if idx[0] != len(flat):
    raise ValueError(f'Not all arrays consumed: used {idx[0]} of {len(flat)}')
  return result
