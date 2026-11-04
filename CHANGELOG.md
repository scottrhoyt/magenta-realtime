# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- MLX `MagentaRT2System.generate` runs each frame's sampler step under
  `mx.compile` and queues frames ahead with `mx.async_eval`. The audio is the
  same, sample for sample; a frame takes about 26 ms instead of 36 on
  `mrt2_base` at 8 bits on Apple Silicon. `compile=False` (or the `compile`
  attribute) restores the uncompiled loop. Warm-up runs uncompiled steps
  first, then traces the compiled step.
- MLX `generate` evaluates every array in the carried state each frame, not
  only the output. Unevaluated state arrays chained each step's graph to the
  last (about 120 B per frame), until Metal's buffer limit ended a long stream
  (about 18 000 frames where MLX falls back to 499 000 buffers).
- The state flatten helpers moved from `magenta_rt.mlx.export` to
  `magenta_rt.mlx.state` (`flatten_state`, `unflatten_state`).

## [2.0.3] - 2026-07-30

- Fix prefilling forced token bug
- Increase modularity on the MRT2 model input/conditioning configs
- Add mrt checkpoints download CLI
- Improvements and bug fixes to examples

## [2.0.2] - 2026-06-04

### Added

- First release of Magenta RealTime 2.
- - Model weights via Hugging Face
- - PyPI package `magenta-rt[mlx]` supports JAX and optionally MLX
- - C++ inference library
- New examples:
- - Jam v0.0.1
- - Collider v0.0.1
- - Audio Unit (AUv3) v0.0.1
- - Standalone v0.0.1
- - Pure Data v0.0.1
- - SuperCollider v0.0.1
- - Max/MSP v0.0.1

### Removed

- Removed Magenta RealTime 1. It can be found at the branch [v1_legacy](https://github.com/magenta/magenta-realtime/commits/v1_legacy).
