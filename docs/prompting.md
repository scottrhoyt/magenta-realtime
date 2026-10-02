# Prompting guide (preliminary)

```{warning}
This guide is preliminary. It is based on a small set of experiments with
`mrt2_small` on a single machine (JAX, CUDA), not on a systematic evaluation.
Most findings come from measuring the MusicCoCa style embedding; a smaller
number come from generated audio, which was scored with MusicCoCa audio
embeddings and a spectral centroid, **not by listening**. See
[What we don't know](#what-we-dont-know) before relying on any claim here.
```

## How a prompt becomes style

A text prompt does not talk to the language model directly. The path is:

1. The text is lowercased and tokenized with SentencePiece. Anything past
   127 tokens is dropped (`max_text_length = 128` in `magenta_rt/musiccoca.py`).
2. The MusicCoCa text encoder maps it to a 768-dim embedding. Text and audio
   share this space, so an audio clip can be used as a prompt instead
   (clips are 10 s at 16 kHz mono and are averaged over time).
3. The embedding is quantized to 12 RVQ tokens (10 bits each). These 12 tokens
   are the only style information the language model sees on every frame.
4. Optionally (`use_mapper=True`, text only), the embedding first passes
   through a noise-conditioned mapper that moves it toward audio space.

Everything below follows from this pipeline: the model only "hears" what
survives into those 12 tokens.

## What we found

### Case, punctuation, and word order barely matter

- `disco funk`, `Disco Funk`, `DISCO FUNK`, and `disco, funk` produce identical
  embeddings (cosine similarity 1.00).
- Reordering two genres (`jazz, heavy metal` vs `heavy metal, jazz`) gives
  0.99–1.00 similarity. The same genre dominates in both orders, so which
  phrase wins depends on the phrase, not on whether it comes first or last.
- Shuffling 12 descriptors gave a mean similarity of 0.95 to the original.
- Placing one phrase at the front, middle, or end of a long prompt made no
  consistent difference.

Grouping matters a little more when words compete for the same meaning.
`acoustic metal guitar` and its permutations ranged from 0.94 to 0.99
similarity, and moving a comma changed spectral brightness noticeably in the
audio. But in generated audio the differences between orderings were of the
same size as rerunning the same prompt (see below).

### Short is more sensitive than long

Appending comma-separated descriptors to `jazz`:

| Descriptors added | Style tokens matching the full 101-token prompt (of 12) |
|---|---|
| 0–2 | 0 |
| 3–5 | 2–3 |
| 10 | 6 |
| 20 | 10 |
| 30 | 12 |

- The first handful of phrases move the style the most. After about 30 tokens,
  each extra phrase changes the embedding by well under 1%.
- The reverse also holds: in a 101-token prompt, replacing one phrase with
  `heavy metal distortion` (even the genre word at the front) left 10–12 of 12
  style tokens unchanged and the generated audio stayed close to the original
  (similarity 0.77–0.79 vs 0.88 for a plain rerun). A prompt of
  `heavy metal distortion` alone scored 0.42 against that text; the swapped
  long prompts scored 0.02–0.05.

**Implication:** a single element buried in a long prompt is effectively
ignored. If something needs to matter, keep the prompt short.

### Some words dominate

In multi-genre prompts, certain words overpower others regardless of position:
`heavy metal` beats `jazz`, `salsa` beats `ambient pads`, `disco funk` beats
`baroque`. In `disco funk piano`, `disco` alone is already 0.95 similar to the
full prompt, so `piano` adds little.

### Negation is unreliable

`jazz piano, no drums` embeds 0.72 similar to `jazz piano`, and only a small
drop in a drums probe was seen in one generated clip, alongside an unrelated
drop in the jazz-piano score. Treat "no X" as at best a weak hint, and prefer
describing what you want.

### Mood words are weak handles

`sad`, `upbeat happy`, and `aggressive` sit 0.57–0.70 similar to each other
and to instrument prompts, while distinct genres sit 0.4–0.65 from each other.
Genre and instrument tags separate cleanly; moods do not.

### The mapper is a variation knob, not a fidelity knob

`use_mapper=True` moves a text embedding to about 0.5 cosine similarity from
the raw embedding (as low as 0.16 for abstract prompts such as `ambient pads`).
Seeds produce different outputs: 0.85–0.86 similarity across seeds for genre
prompts, 0.37 for `ambient pads`.

The `mrt jax generate` path (`magenta_rt/jax/generate.py`) calls
`embed_style(prompt, use_mapper=True)` with the default seed 0, so **all
generated audio in this guide used mapped embeddings**. The embedding-space and
style-token measurements, however, were taken on **raw** text embeddings
(`use_mapper=False`). Because the mapper moves embeddings so far, those
embedding-level findings (length saturation, word order, grouping, negation)
have not been confirmed on the mapped embeddings that the model actually
receives from the CLI. The audio-level results are the ones that reflect the
real path.

### Run-to-run variation

Separate `mrt jax generate` invocations of the same prompt gave audio
embeddings only ~0.85–0.93 similar. Within a single Python process, calling
`generate` twice with the same embedding gave 1.00 similarity (the blending
experiment below). We have not found out why the two differ, so treat 0.85–0.93
as the floor for comparisons across separate processes, and don't assume it for
repeated calls in one process.

## Sweet spot

Based on the above, aim for roughly **3–10 phrases, about 5–25 tokens**: a
genre or style anchor first, then a few concrete descriptors (instruments,
tempo, texture). Use commas freely; they don't hurt.

Token counts are from the SentencePiece vocabulary.

| Prompt | Tokens | Status |
|---|---|---|
| `disco funk` | 2 | Generated; audio scored closest to its own text (0.47) |
| `funky disco with slap bass and four-on-the-floor drums` | 14 | Generated; closest to `disco funk` (0.40) |
| `jazz, piano, trio, slow tempo, brushed drums, upright bass` | 14 | Embedding-level only (a prefix of the tested jazz prompt) |
| `jazz piano` | 2 | Generated; audio scored 0.43 against its text |
| `ambient pads with sub bass` | 5 | From the README; not analysed |
| `string ensemble, slow, warm, cinematic` | 8 | **Untested** |
| `salsa, brass section, congas, upbeat` | 8 | **Untested** |
| `ambient pads, evolving, spacious, sub bass` | 9 | **Untested** |
| `thrash metal, fast double-kick drums, distorted guitars` | 11 | **Untested** |
| `lo-fi hip hop, mellow keys, vinyl crackle, laid-back drums` | 17 | **Untested** |
| `house, four-on-the-floor kick, warm bassline, analog synths` | 17 | **Untested** |

The "untested" rows follow the pattern above (one anchor, a few concrete
descriptors, under 20 tokens) but were not run through the model. Treat them
as starting points, not verified recipes.

Things to avoid, given what we measured:

- Long lists of 30+ descriptors, where later phrases have almost no effect.
- Burying the one thing you care about in the middle of a long prompt.
- Relying on `no <instrument>` to remove something.
- Mixing a dominant genre word with a quiet one and expecting an even blend.
  To blend styles, use weighted embedding mixing instead (see below).

## Blending instead of long prompts

Because text and audio share an embedding space, you can mix prompts by
averaging their embeddings rather than putting everything in one string:

```python
from magenta_rt import musiccoca
import numpy as np

style_model = musiccoca.MusicCoCa()
a = style_model.embed('disco funk')
b = style_model.embed('ambient pads')
tokens = style_model.tokenize(np.mean([a, b], axis=0))
```

Collider does this with weights. This is the documented way to control the
balance between two styles; we did not compare it against single-string
prompts in audio.

### Sweeping the blend weight (one experiment)

We blended the mapped embeddings (`use_mapper=True`, seed 0) of
`calm ambient pads` and `upbeat house` at weights 0, 0.1, ... 1.0
(`(1 - w) * A + w * B`, renormalized to unit length), generated 8 s of audio
for each with `mrt2_small`, and scored the clips with MusicCoCa audio
embeddings. Each clip is a static blend generated from scratch, one clip per
weight.

| Weight toward house | Audio vs ambient text | Audio vs house text | Similarity to w=0 clip | Similarity to w=1 clip | Similarity to previous step |
|---|---|---|---|---|---|
| 0.0 | 0.32 | 0.11 | 1.00 | 0.09 | – |
| 0.2 | 0.29 | 0.13 | 0.94 | 0.14 | 0.89 |
| 0.3 | 0.27 | 0.16 | 0.75 | 0.14 | 0.69 |
| 0.4 | 0.19 | 0.17 | 0.48 | 0.26 | 0.82 |
| 0.5 | 0.16 | 0.23 | 0.30 | 0.47 | 0.55 |
| 0.6 | 0.11 | 0.42 | 0.16 | 0.84 | 0.50 |
| 0.8 | 0.12 | 0.48 | 0.11 | 0.88 | 0.91 |
| 1.0 | 0.12 | 0.44 | 0.09 | 1.00 | 0.88 |

(Weights 0.1, 0.7 and 0.9 omitted for brevity; they sit between their
neighbours.)

What this showed:

- **The response is not linear.** Weights 0–0.2 stay close to the pure-ambient
  clip (0.92–0.94 similarity); most of the change happens between roughly 0.3
  and 0.6; from 0.6 up the clips are within 0.83–0.88 of the pure-house clip.
  A linear sensor-to-weight mapping will therefore not give a linear change in
  sound.
- **The middle is a hybrid.** At 0.4–0.5 the clips score low against both
  anchor texts (below 0.25), so they are neither anchor. Loudness (RMS) rose
  from about 0.04 to 0.11 across the sweep, roughly in step with the house
  weight.
- **Token flips don't predict audio change.** The number of style tokens
  matching the w=0 tokens went 12, 11, 4, 3, 2, 1, 1, 0, 0, 0, 0. At w=0.2 only
  4 tokens matched, yet the clip was 0.94 similar to the w=0 clip.
- **Normalization changes the tokens.** The mean of two unit vectors has norm
  below 1. Quantizing the plain mean vs the renormalized blend gave different
  tokens (only 4–6 of 12 matching in the middle weights). We generated from the
  renormalized blend only and did not compare the audio, so we don't know which
  sounds better.

This tests static blends only. In a live stream the model carries state across
frames, so a style change during playback will behave differently. We have not
tested that, and it needs the MLX/C++ path (Apple Silicon). The result is from
a single anchor pair and a single seed, scored by proxy metrics, with no
listening test.

## What we don't know

- **How audible the effects are.** Everything about word order, grouping, and
  detail was measured in embedding space or via proxies on single or double
  clips. We have not run a listening test.
- **Model dependence.** All audio was generated with `mrt2_small` at the
  default sampling settings (temperature 1.3, top-k 40, `--cfg-musiccoca 3.0`).
  `mrt2_base` may respond to fine detail differently.
- **Effect of CFG and temperature.** The `--cfg-musiccoca` scale probably sets
  how strongly the audio follows the prompt, and temperature adds randomness.
  Neither was swept.
- **Whether quantization flips matter.** Reordering flips 1–5 of the 12 style
  tokens. In one experiment the number of flipped tokens did not predict how
  different the audio was. We don't know when a flip is audible.
- **Genre coverage.** The detail-sensitivity experiments used mostly jazz, plus
  a few genre pairs. Other genres, and non-Western styles in particular
  (training data is mostly Western, instrumental stock music), are unexplored.
- **Vocals and lyrics.** The model is not designed for words. We did not test
  how vocal terms behave beyond one embedding comparison.
- **Mapped vs raw embeddings.** See above: the embedding-level findings were
  measured without the mapper, while generation used it. Whether they hold
  after the mapper, and whether mapped or raw embeddings sound more faithful to
  the text, is untested.
- **Audio prompts.** Not covered here at all, including how clip length,
  pooling, and mixing with text behave.
- **Style changes over time.** How quickly and smoothly the output follows a
  changed prompt in a live stream was not examined.

## What to explore next

1. Run a small listening study on a handful of prompts (a short genre tag vs
   the same tag plus 3 and 10 descriptors) so the embedding-space findings can
   be checked by ear.
2. Repeat the length and ordering sweeps across genres and on `mrt2_base`.
3. Sweep `--cfg-musiccoca` and temperature, with several seeds per prompt, to
   separate prompt effects from sampling noise.
4. Repeat the length, ordering, and grouping measurements on mapped
   embeddings, and generate from raw embeddings to compare against the mapped
   ones.
5. Compare one long prompt against an equivalent weighted blend of short
   prompts, and repeat the blend-weight sweep with other anchor pairs, seeds,
   and with and without renormalization.
6. Test negation phrasings, and whether a separate negative prompt is a better
   route than `no X` (the exported function signature does list negative
   prompts).
7. Test audio prompts and text/audio blends.
