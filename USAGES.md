# Detailed usage

[README](README.md) · [中文说明](README_zh.md) · [中文使用指南](USAGES_zh.md)

## 1. Install

Requirements:

- Python 3.12 or newer
- `uv`
- NVIDIA GPU recommended for MuScriptor audio transcription

```bash
uv sync
```

The project installs PyTorch from the CUDA 12.8 index on Windows/Linux.
MIDI conversion and model placement can run on CPU; audio transcription is
substantially faster with CUDA. MuScriptor model access may require a
Hugging Face token and accepting the model license.

Check the installation:

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
uv run noteblockify --help
```

## 2. One-command conversion

MIDI:

```bash
uv run noteblockify \
  --in "song.mid" \
  --stage model \
  --ranker mc_ranker.auto.best.pt \
  --out "song.nbs"
```

Audio:

```bash
uv run noteblockify \
  --in "song.mp3" \
  --stage model \
  --ranker mc_ranker.auto.best.pt \
  --out "song.nbs"
```

The audio path is:

```text
audio -> MuScriptor MIDI transcription -> OpenNBS pre conversion -> smooth MC placement -> NBS
```

Audio transcription is cached beside the input as `<name>.mid`.
Use `--model small`, `--model medium`, or `--model large` to select the
MuScriptor transcription model.

`--ranker mc_ranker.auto.best.pt` is optional. With no ranker, the deterministic
smooth voice decoder is used. The checked-in checkpoint is the current trained
placement model, not an edit model.

## 3. Two-stage conversion

Use this when you want to listen to the raw baseline or inspect intermediate
files:

```bash
uv run noteblockify --in "song.mid" --stage pre --out "song.pre.nbs"
uv run noteblockify --in "song.pre.nbs" --stage model \
  --ranker mc_ranker.auto.best.pt --out "song.nbs"
```

`pre.nbs` is intentionally not Minecraft-playable for every source note. It
preserves raw mapped keys, including keys outside `33–57`, and is the best
reference for checking whether the MIDI transcription and OpenNBS mapping are
correct.

## 4. Placement algorithm

The Minecraft stage has a hard key window:

```text
33 <= NBS key <= 57
```

For every source voice, the decoder:

1. Builds all legal same-pitch-class octave candidates.
2. Uses dynamic programming over the complete voice sequence.
3. Penalizes large contour errors and octave/register switches.
4. Accounts for same-tick collisions with other voices.
5. Repeats voice passes so register choices are coordinated across channels.

This is intentionally voice-level. A low note that cannot fit the MC window is
not independently raised while its preceding notes remain low; the surrounding
phrase may move to the same legal octave. This prevents the `38 -> 43`
reversal that occurs with independent nearest-octave folding.

The production stage changes keys only. It preserves:

```text
tick, layer, instrument, velocity, panning, note count
```

The output is suitable for OpenNBS and Meteor when all keys are in `33–57`.

## 5. Candidate generation

Deterministic baseline octave candidates:

```bash
uv run noteblockify --in song.pre.nbs --stage model \
  --candidates --out candidates/
```

This writes deterministic alternatives such as `balanced`, `nearest`, `low`,
`high`, and `spread`. These candidates use the older fixed-policy decoder;
the normal `model` path uses the smooth voice-level decoder.

Bounded edit alternatives:

```bash
uv run noteblockify --in song.pre.nbs --stage model \
  --edit-candidates --out edit-candidates/
```

This writes finite candidates such as:

```text
balanced, drop, replace, clamp, phrase, smooth, add
```

The edit candidates are for listening and labeling. Their note counts may
differ. The default production path remains key-only and does not silently
drop notes.

## 6. Training

### Existing key-only ranker

Record a human A/B decision:

```bash
uv run noteblockify-feedback \
  --data data/preferences.jsonl \
  --pre song.pre.nbs \
  --preferred candidates/song.high.nbs \
  --rejected candidates/song.low.nbs \
  --note "clearer lead"
```

Train from human preferences:

```bash
uv run noteblockify-train \
  --data data/preferences.jsonl \
  --output mc_ranker.pt
```

The key-only preference format requires equal note counts and only validates
pitch-class-preserving key changes.

### Optional edit ranker

Edit candidates use a separate format because note counts may differ:

```bash
uv run noteblockify-feedback \
  --data data/edit_preferences.jsonl \
  --pre song.pre.nbs \
  --preferred edit-candidates/song.replace.nbs \
  --rejected edit-candidates/song.drop.nbs \
  --note "replace is better"

uv run noteblockify-train-edits \
  --edit-data data/edit_preferences.jsonl \
  --output mc_edit_ranker.pt
```

Use the edit ranker explicitly:

```bash
uv run noteblockify \
  --in song.pre.nbs \
  --stage model \
  --edit-ranker mc_edit_ranker.pt \
  --out song.edit.nbs
```

Do not use automatic proxy labels as a substitute for listening labels. A
ranker can learn a bad preference consistently if the examples are bad.

## 7. NBS and Meteor behavior

The OpenNBS format stores note key, velocity, panning, and fine pitch. Meteor's
current NBS decoder parses the file but keeps only instrument and key for
physical NoteBot playback:

- per-note velocity is discarded;
- per-note panning is discarded;
- per-note fine pitch is discarded;
- physical block position determines stereo perception;
- NoteBot tunes blocks and then attacks them on its 20-game-tick schedule.

Therefore `--pitch` is not a fix for Meteor playback. It is only useful when
the target player honors NBS fine-pitch fields.

Put a final file in:

```text
.minecraft/meteor-client/notebot/
```

Meteor supports classic NBS and OpenNBS v5. In Exact Instruments mode, the
nearby blocks must provide the requested vanilla instruments. NoteBot may warn
about missing instrument/key combinations if the physical setup is incomplete.

## 8. Scoring

```python
from noteblockify.hear import compare

score = compare("song.mid", "song.nbs")
print(score.f1, score.instrument, score.total)
```

The score is an objective conversion check, not a musical-quality judge:

- `f1`: one-to-one note matching by time, pitch class, and event presence;
- `instrument`: mapped vanilla-instrument agreement;
- `total`: `0.6 * f1 + 0.4 * instrument`.

It cannot determine whether one legal octave sounds more musical than another.
Use direct OpenNBS/Meteor listening for that decision.

## 9. Troubleshooting

### Model output sounds like an octave reversal

Use the current smooth model path, not an old `.model.nbs` generated before the
voice-level decoder was enabled. Regenerate from the original MIDI or
`.pre.nbs`:

```bash
uv run noteblockify --in song.mid --stage model \
  --ranker mc_ranker.auto.best.pt --out song.nbs
```

### Audio transcription fails

Set `HF_TOKEN` in `.env`, accept the MuScriptor model license, and retry. If
CUDA memory is insufficient, use `--model small` or run the transcription on
CPU.

### Output does not load in NoteBot

Verify that it is a valid NBS file and that all note keys are in `33–57`:

```bash
uv run python - <<'PY'
import pynbs
song = pynbs.read("song.nbs")
assert all(33 <= note.key <= 57 for note in song.notes)
print(len(song.notes), len(song.layers))
PY
```

### The `add.nbs` candidate is corrupt

Use a current output generated after the unique-layer fix. Multiple added notes
must never share the same NBS layer at the same tick.

## 10. Verification

```bash
uv run python -m py_compile noteblockify/*.py
uv run python -m pytest tests/ -q
```
