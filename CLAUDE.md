# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Magenta RealTime 2 (MRT2): an open-weights real-time music generation model. The repo has three layers:

1. **`magenta_rt/`** — Python package (`magenta-rt` on PyPI, CLI entry point `mrt`) for inference with JAX or MLX backends, model download, and exporting the model to `.mlxfn` for C++.
2. **`core/`** — `magentart::core`, a static C++ library that runs exported `.mlxfn` models on Apple Silicon via MLX + TFLite + SentencePiece.
3. **`examples/`** — macOS apps/plugins built on `magentart::core` (AUv3, standalone, Jam, Collider, Max/Pd/SuperCollider externals, `hello_mrt2` CLI).

The shipped streaming engine (C++ `RealtimeRunner`, MLX) requires Apple Silicon. The Python library does inference on NVIDIA GPUs (JAX), and that is fast enough for real time: measured on an RTX 3090 (WSL2), `mrt2_small` ≈ 12 ms/step and `mrt2_base` ≈ 29 ms/step against the 40 ms (25 fps) budget. The current dev environment is Linux/WSL with that GPU, where MLX and all C++/example builds are unavailable.

## Setup

```bash
git clone --recurse-submodules ...      # sequence-layers is a git submodule
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[mlx,dev]"          # or ".[jax,dev]" on non-Mac
mrt models init                         # MusicCoCa + SpectroStream resources
mrt models download                     # exported .mlxfn models (interactive)
mrt checkpoints download mrt2_small     # .safetensors checkpoints (needed by tests)
```

Git remotes: `origin` is upstream `magenta/magenta-realtime`; `fork` is `scottrhoyt/magenta-realtime` (push branches there).

Assets live under `$MAGENTA_HOME/magenta-rt-v2/` (default `~/Documents/Magenta/magenta-rt-v2/`): `resources/` (musiccoca, spectrostream), `models/<name>/`, checkpoints, outputs. See `magenta_rt/paths.py`. `--source gcs` is an alternative to Hugging Face for downloads.

## Common commands

```bash
# Inference
mrt mlx generate --prompt "disco funk" --duration 4.0 --model=mrt2_small
mrt mlx generate --model=mrt2_small --no-mlxfn --bits=8   # pure-Python MLX model instead of .mlxfn
mrt jax generate --model=mrt2_small
XLA_PYTHON_CLIENT_PREALLOCATE=false mrt jax generate --model=mrt2_base   # base aborts on the 3090 in XLA's allocator autotune (bfc_allocator check failure, exit 134) without this

# Export for C++ (writes to ~/Documents/Magenta/magenta-rt-v2/models/<name>)
mrt mlx export --output-name=mrt2_base --bits=8
mrt mlx export --output-name=debug --num-layers=2 --depth-num-layers=2 --bits=8 --skip-restore  # tiny untrained model
mrt mlx export-spectrostream

# Tests (require downloaded resources + mrt2_small checkpoint; MLX-based)
pytest -s tests/test_musiccoca.py
pytest -s tests/test_gptq.py
pytest -s tests/test_prefill_correctness.py
python scripts/generate_test_reference.py && pytest -s tests/test_bitlevel_parity.py   # parity needs reference generated first
pytest -s tests/test_gptq.py -k <name>                    # single test

# Benchmark regression
python scripts/bench_track.py && python scripts/bench_show.py --samples

# C++ (macOS only; needs cmake<3.28 per README, CI uses -DCMAKE_POLICY_VERSION_MINIMUM=3.5 with newer cmake)
cmake . -B build
cmake --build build --target hello_mrt2 -j10
cmake --build build --target numpy_random_state_test && ./build/core/numpy_random_state_test   # self-contained C++ unit test
bash examples/scripts/build-all.sh     # build, sign, deploy all apps/externals locally

# Example web UIs (npm workspaces; must be built before the native app targets embed them)
cd examples && npm install && npm run build --workspaces --if-present

# Docs (Sphinx + MyST)
uv pip install -r docs/requirements.txt && sphinx-build -b html docs docs/_build/html
```

The PR template (`.github/pull_request_template.md`) expects the generate commands, the pytest list above, and the benchmark regression output to be run and pasted.

## Code style

- Python is formatted with **pyink**: 2-space indentation, 80-column lines, majority quotes (`[tool.pyink]` in `pyproject.toml`).
- Pre-commit (`pre-commit install`) inserts the Apache license header (`LICENSE_HEADER.txt`) into `.py`, `.h/.cpp/.cc/.mm/.m`, `.sh`, `.tsx`, and `CMakeLists.txt` files. New files need this header.
- Version is `__version__` in `magenta_rt/__init__.py` (flit dynamic version); update `CHANGELOG.md` alongside.

## Architecture

### Python package

- **Parallel JAX and MLX implementations.** `magenta_rt/jax/` and `magenta_rt/mlx/` mirror each other module-for-module (`depthformer.py`, `transformer.py`, `model.py`, `system.py`, `generate.py`, `spectrostream`). Changes to model behavior generally need to be made in both, and `tests/test_bitlevel_parity.py` checks numerical agreement.
- **`system.py` is the user-facing API.** `MagentaRT2System` wraps the style model (MusicCoCa), the Depthformer LM, and the SpectroStream codec into a streaming `embed_style(...)` → `generate(style=..., frames=N, state=...)` loop that returns audio plus carried state. MLX additionally has `MagentaRT2SystemMlxfn` / `MagentaRT2SystemStdMlxfn`, which run an exported `.mlxfn` graph instead of constructing the model in Python — this is the same graph the C++ engine runs, so it's the Python reference for C++ behavior. Top-level `magenta_rt.MagentaRT2Jax/Mlx/StdMlxfn` are lazily imported so a missing backend doesn't break the package.
- **Streaming with `generate`** (JAX and MLX share the signature): `generate(conditioning, frames, state)` returns `(Waveform, state)`. Chunked calls carrying `state` are bit-identical to one long call (verified on JAX: 20×5 frames vs 100 frames), so a streaming loop can run any chunk size and change `conditioning` between chunks. `conditioning[MUSICCOCA.key]` may be a float embedding (tokenized inside) or a list of ints (used as-is, so tokenize once yourself; `tokenize_style` takes ~1.5 ms). Repeated calls in one process with the same embedding are deterministic. 1 frame = 40 ms (25 fps), 1920 samples/channel at 48 kHz stereo.
- **`mrt jax generate` embeds with `use_mapper=True`** (seed 0, in `jax/generate.py`), so anything generated through the CLI path uses mapper-shifted text embeddings, not raw ones.
- **`config.py`** defines `ModelSpec` (model sizes, e.g. `mrt2_small`, `mrt2_base`) and `TokensConfig`/conditioning inputs (MusicCoCa style tokens, piano-roll notes, drums) with per-input CFG. The exported `.mlxfn` function signature (cond, temperature, top_k, CFG scales, negative prompts, forced tokens, `*state`) is defined by `mlx/export.py` + `MagentaRT2SystemStdMlxfn._build_graph_args` and must stay in sync with `core/src/mlx_engine.cpp`.
- **`musiccoca.py`** runs MusicCoCa text/audio style embedding via TFLite (`ai-edge-litert`) + SentencePiece — the same TFLite assets the C++ engine loads.
- **Style conditioning / prompting.** `docs/prompting.md` is a preliminary guide from experiments on `mrt2_small` (see its caveats and unknowns before relying on it). Facts worth knowing when touching this code: the text encoder lowercases and truncates at 127 tokens; style reaches the LM only as 12 RVQ tokens; reorder/punctuation/case barely change the embedding; the embedding saturates by ~30 tokens, so details in long prompts are ignored; negation ("no drums") is unreliable; blending embeddings (average, then renormalize) is the way to mix styles, but the audio response to the blend weight is nonlinear (little change below ~0.3, most change ~0.3–0.6), and style-token flip counts do not predict audible change. Embedding-level measurements there used raw text embeddings while generation used the mapper.
- **`mlx/gptq.py`** implements GPTQ for int4 export (`--quantize-method=gptq`).
- **Vendored `sequence_layers`.** `magenta_rt/_vendor/sequence-layers` is a git submodule; `_vendor_hook.install()` (called from `magenta_rt/__init__.py`) makes it importable as `sequence_layers` if not installed. Both backends build on `sequence_layers` (`sl.SerialCombinatorMixin`, `sl.Emitting`). The `NOTE(vendoring)` comments in `pyproject.toml` describe the planned migration once it's on PyPI.
- **CLI** (`magenta_rt/cli/`): click group `mrt` with subgroups `mlx`, `jax`, `models`, `checkpoints`.

### C++ core

- `MLXEngine` (`core/include/magentart/mlx_engine.h`) — low-level pipeline: loads `.mlxfn` + MusicCoCa TFLite assets, `generate_frame(L, R)`. Lifecycle methods are controller-thread only; atomic setters (sampling params, MIDI, drum mode) are thread-safe.
- `RealtimeRunner` — wraps `MLXEngine` with an inference thread, lock-free SPSC stereo ring buffers (`ring_buffer.h`), volume/mute smoothing, MIDI gate, and prompt-surface blending. `read_audio_stereo` is safe to call from an audio callback. All bundled hosts use this.
- `numpy_random_state.h` reproduces NumPy's RNG so C++ sampling can match Python (`scripts/compare_python_n_cpp.py`, `scripts/compare_mlxfn.py`).
- Gotchas: long-running loops calling MLX must use `magentart::detail::AutoreleasePool` per iteration (MLX/Metal leaks Obj-C objects otherwise); the runner issues dummy GPU ops while idle to prevent macOS GPU downclocking.
- Root `CMakeLists.txt` FetchContents MLX/TFLite/SentencePiece, sets macOS 14.0 deployment target, adds `core` and every example, and defines `magentart_add_notarize_target`. `MAGENTART_DEBUG_LOG` option enables AUv3 debug logging.

### Example apps

- Native shells are Objective-C++ (`*.mm`) hosting a **React/TypeScript UI in a WKWebView**; the UI talks to native code via `webkit.messageHandlers` / injected JS. Shared React components live in `examples/common/react_ui/`, shared Obj-C (model download/management, settings) in `examples/common/objc/`, shared C++ in `examples/common/cpp/`.
- Each UI's `postbuild.js` rewrites the Vite `dist/index.html` so it loads from `file://` (strips `type="module"`/`crossorigin`).
- Every app bundle must copy `mlx.metallib` next to its executable and codesign it before the parent bundle. New apps: follow `docs/apps/developer.md` and register in `examples/scripts/build-all.sh` / `notarize-all.sh`.
- **Collider**: `examples/collider/.agents/rules/collider_interactions.md` is the interaction-design spec. When changing Collider interaction behavior, update that document in the same change (user-facing behavior only, no implementation details) and tell the user what changed.
