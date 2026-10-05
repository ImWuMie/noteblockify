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
uv run python main.py
```

Edit the constants at the top of `main.py`:

```python
AUDIO = Path("yuai.mp3")     # your input file (mp3/wav/flac)
```

Steps performed:

1. muscriptor `medium` transcribes the audio to MIDI (cached: the .mid
   is reused if it already exists).
2. `noteblockify.song.arrange` converts the MIDI to an NBS song.
3. The result is saved next to the audio with a `.nbs` extension.

First transcription takes ~2 minutes per 4 minutes of audio.

## Converting MIDI files

One file:

```bash
uv run python -c "from noteblockify.song import arrange; arrange('song.mid').save('song.nbs')"
```

A folder:

```bash
uv run python -c "
from pathlib import Path
from noteblockify.song import arrange
out = Path('out'); out.mkdir(exist_ok=True)
for mid in Path('data/midi').glob('*.mid'):
    song = arrange(mid)
    song.save(out / (mid.stem + '.nbs'))
    print(mid.name, len(song.notes), 'notes')
"
```

Programmatic API:

```python
from noteblockify.song import arrange

song = arrange("song.mid")     # returns pynbs.File
song.save("song.nbs")
```

## What the converter does to your music

| Situation | Handling |
|---|---|
| Note outside F#3–F#5 | folded by octave to the nearest in-window pitch class |
| Whole voice in a bad octave | octave model shifts it by ±1/±2 octaves |
| Bass note needing +1 octave or more to fit | deleted (would muddy the melody register) |
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

## Retraining the octave model

```bash
uv run python -m noteblockify.octave        # trains on data/midi/*.mid, saves octave.pt
uv run python score_model.py       # train-set, held-out, and conversion scores
```

The trained model ships as `octave.pt`; conversions use it out of
the box. Retrain to specialize to your repertoire. Add MIDI files to
`data/midi/` to specialize the model to your repertoire.

## Troubleshooting

- **`ValueError: every note was dropped`** — the song is entirely bass
  below the window; it cannot be represented in vanilla note blocks.
- **muscriptor model download fails** — set `HF_TOKEN` in `.env` and
  accept the MuScriptor model license on HuggingFace.
- **CUDA out of memory** — use a MIDI instead of audio, or run
  transcription on CPU (much slower).
