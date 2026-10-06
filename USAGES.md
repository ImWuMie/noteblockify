# Usage guide

[English README](README.md) | [中文说明](README_zh.md) | [使用指南](USAGES_zh.md)

## Install


```bash
# Requires Python 3.12+, uv, and an NVIDIA GPU for the audio path
uv sync
```

Dependencies: torch (CUDA 12.8 index pinned in pyproject), muscriptor,
mido, pynbs. Pure MIDI conversion runs fine on CPU; only audio
transcription needs the GPU (~5 GB VRAM, medium model).

## Converting audio to NBS

```bash
# Stage 1: raw converter baseline
noteblockify --in song.mp3                  # writes song.pre.nbs
noteblockify --in song.mp3 --out out.pre.nbs
noteblockify --in song.mp3 --model small    # lighter transcription model
# Stage 2: MC-constrained refinement of the saved baseline
noteblockify --in song.pre.nbs --stage model # writes song.model.nbs
# Generate legal alternatives for human A/B listening
noteblockify --in song.pre.nbs --stage model --candidates --out candidates/
```

Steps performed:

1. muscriptor `medium` transcribes the audio to MIDI (cached: the .mid
   is reused if it already exists).
2. `noteblockify.song.arrange_pre` converts the MIDI to a raw `.pre.nbs` song.
3. `--stage model` reads the saved `.pre.nbs` and chooses legal same-pitch-class octaves with a voice-level continuity decoder.

First transcription takes ~2 minutes per 4 minutes of audio.

## Converting MIDI files

One file:

```bash
noteblockify --in song.mid
```

A folder:

```bash
for f in data/midi/*.mid; do noteblockify --in "$f"; done
```

Programmatic API:

```python
import pynbs
from noteblockify.mc_model import refine
from noteblockify.song import arrange_pre

pre = arrange_pre("song.mid")
pre.save("song.pre.nbs")
model = refine(pynbs.read("song.pre.nbs"))
model.save("song.model.nbs")
```

## What the converter does to your music

| Situation | Handling |
|---|---|
| Raw pre stage | preserves the mapped OpenNBS key, including out-of-window keys |
| MC model stage | chooses legal same-pitch-class octaves for each voice, including alternate octaves for in-window notes |
| Model objective | stable voice register, pitch-class preservation, local contour, and collision avoidance |
| Notes, ticks, instruments, layers, velocity, panning | all preserved; no deletion |
| GM program | mapped per the OpenNBS 128-entry table, with channel-level timbre diversification |
| Drum channel (10) | mapped per the OpenNBS GM drum table |
| More simultaneous notes than layers | extra layers are created — nothing dropped |
| Song longer than 65535 ticks | proportionally re-gridded (relative timing preserved) |
| Multiple tempo events | the first one defines the timing (OpenNBS behavior) |

## Scoring a conversion

```python
from noteblockify.hear import compare

score = compare("song.mid", "song.nbs")
print(score.f1, score.instrument, score.total)   # 0..1 each
```

- `f1` — notes matched by onset (±60 ms + drift) and pitch (±1 semitone)
- `instrument` — matched notes carrying the mapped vanilla instrument
- `total` — weighted 0.6·f1 + 0.4·instrument

A faithful conversion of a well-formed MIDI scores ≈0.99; the last
fraction of a point is tick-grid quantization.

## Playing the NBS in Minecraft (Meteor Client)

1. Copy the `.nbs` into `.minecraft/meteor-client/notebot/`.
2. Place note blocks around you (NoteBot scans a 6³ region within
   reach), enable the Notebot module, and load the song.
3. The module tunes the blocks to each note, then plays by attacking
   them.

In-game limitations (verified against Meteor's decoder source):

- **Velocity is ignored** — the decoder reads and discards the per-note
  velocity byte. All notes play at full volume.
- **Panning is ignored** — stereo comes from where the blocks physically
  are.
- **Exact instruments mode** — NoteBot matches note blocks by
   instrument type; make sure you have blocks of the needed instrument
   types (harp/bass/drum/... derived from what's under the note block).
- **Timing jitter** — Meteor re-derives timing at 20 game ticks per
   second, so sub-tick placement gets ±1 tick (~50 ms) rounding. All
   NBS files are subject to this.

## Model training

The decoder first generates hard-valid candidates, then a PyTorch ranker
chooses each note's legal octave inside a voice-level dynamic program. Lower
raw-register voices receive lower legal targets, preventing dense songs such
as multi-track arrangements from collapsing every bass part into the middle
register. Without `mc_ranker.pt`, it falls back to the deterministic
`balanced` candidate. MIDI-guided training runs on CUDA when available:

```bash
noteblockify-train --midi-guided data/flitered \\
  --output mc_ranker.pt --width 192 --depth 4
noteblockify-feedback --data data/preferences.jsonl \\
  --pre song.pre.nbs \\
  --preferred candidates/song.high.nbs \\
  --rejected candidates/song.low.nbs \\
  --note "clearer lead"
noteblockify-train --data data/preferences.jsonl \\
  --init mc_ranker.pt --output mc_ranker.pt
```

The automatic objective favors MIDI proximity, contour preservation, and
lower-register separation; it is not human truth. Human A/B labels remain
the stronger signal when available. The pre file remains the unchanged
listening baseline.

## Troubleshooting

- **`ValueError: every note was dropped`** — the song is entirely bass
  below the window; it cannot be represented in vanilla note blocks.
- **muscriptor model download fails** — set `HF_TOKEN` in `.env` and
  accept the MuScriptor model license on HuggingFace.
- **CUDA out of memory** — use a MIDI instead of audio, or run
  transcription on CPU (much slower).
