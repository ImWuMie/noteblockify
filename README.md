# noteblockify — Audio/MIDI to Minecraft NBS

[中文说明](README_zh.md) · [Detailed usage](USAGES.md) · [详细使用指南](USAGES_zh.md)

`noteblockify` converts MIDI or audio into `.nbs` songs for Open Note Block
Studio and Meteor Client's NoteBot. The converter keeps a raw OpenNBS-style
baseline and a separate Minecraft-compatible voice-placement stage.

## Pipeline

```text
audio ── MuScriptor ──▶ MIDI ── OpenNBS mapping ──▶ pre ──▶ smooth MC placement ──▶ NBS
MIDI  ─────────────────▶ MIDI ── OpenNBS mapping ──▶ pre ──▶ smooth MC placement ──▶ NBS
```

The `model` stage can consume MIDI/audio directly; an intermediate `.pre.nbs`
file is optional.

## Quick start

```bash
uv sync

# One-step MIDI -> Minecraft-compatible NBS
uv run noteblockify \
  --in "song.mid" \
  --stage model \
  --ranker mc_ranker.auto.best.pt \
  --out "song.nbs"

# One-step audio -> NBS; the first run downloads/loads MuScriptor weights
uv run noteblockify \
  --in "song.mp3" \
  --stage model \
  --ranker mc_ranker.auto.best.pt \
  --out "song.nbs"
```

The checked-in `mc_ranker.auto.best.pt` is the trained placement checkpoint.
Passing it with `--ranker` enables learned candidate scoring. Without a
ranker, `model` still runs using the deterministic smooth voice decoder.

For an explicit two-stage workflow:

```bash
uv run noteblockify --in "song.mid" --stage pre --out "song.pre.nbs"
uv run noteblockify --in "song.pre.nbs" --stage model \
  --ranker mc_ranker.auto.best.pt --out "song.nbs"
```

## Placement behavior

### Raw `pre` stage

- Replicates the OpenNBS GM program map and drum map.
- Preserves timing grid, channel layer bands, velocity, panning, and raw keys.
- Does not force keys into Minecraft's two-octave range.
- Produces the best direct-listening baseline.

### Minecraft `model` stage

Minecraft note blocks represent keys `33–57`. The decoder:

- generates every legal same-pitch-class octave for each source note;
- solves each voice as a sequence instead of making isolated note decisions;
- penalizes unnecessary octave/register switches;
- preserves local melodic direction and voice register;
- separates dense voices and avoids avoidable same-tick unisons;
- preserves note count, ticks, instruments, layers, velocity, and panning.

For `224264 - 室内系的TrackMaker`, this changes the problematic Fantasia
phrase from an octave reversal such as `38 → 43` into a continuous register
path such as `50 → 43`.

The default production model does **not** add or remove notes. Optional edit
candidates are a separate human-listening workflow.

## CLI modes

```bash
# Raw baseline
uv run noteblockify --in song.mid --stage pre --out song.pre.nbs

# Smooth deterministic MC placement
uv run noteblockify --in song.mid --stage model --out song.nbs

# Smooth placement plus the checked-in learned ranker
uv run noteblockify --in song.mid --stage model \
  --ranker mc_ranker.auto.best.pt --out song.nbs

# Generate deterministic baseline octave candidates for A/B listening
uv run noteblockify --in song.pre.nbs --stage model \
  --candidates --out candidates/

# Generate bounded drop/replace/clamp/phrase/smooth/add alternatives
uv run noteblockify --in song.pre.nbs --stage model \
  --edit-candidates --out edit-candidates/
```

Supported input formats:

```text
.mid, .midi, .mp3, .wav, .flac, .ogg, .m4a, .pre.nbs
```

Audio transcription is cached beside the input as `<name>.mid`.

## Optional note-edit workflow

Edit candidates may have different note counts. They are intended for human
A/B listening, not automatic truth generation:

```bash
uv run noteblockify-feedback \
  --data data/edit_preferences.jsonl \
  --pre song.pre.nbs \
  --preferred edit-candidates/song.replace.nbs \
  --rejected edit-candidates/song.drop.nbs \
  --note "replace sounds better"

uv run noteblockify-train-edits \
  --edit-data data/edit_preferences.jsonl \
  --output mc_edit_ranker.pt

uv run noteblockify \
  --in song.pre.nbs \
  --stage model \
  --edit-ranker mc_edit_ranker.pt \
  --out song.edit.nbs
```

The edit stage ranks the generated bounded candidates. It does not invent
unbounded MIDI events.

## Minecraft and NBS limitations

OpenNBS stores velocity, panning, and fine pitch in modern NBS files. Meteor's
current NBS decoder reads the instrument and key but discards per-note velocity,
panning, and fine-pitch fields. Therefore:

- velocity is not audible in Meteor;
- stereo comes from the physical layout of note blocks, not NBS panning;
- `--pitch` is useful for OpenNBS-compatible players, but **not** for Meteor
  physical note-block playback;
- Meteor's NoteBot tunes and plays physical blocks, so all requested keys must
  be in the playable `33–57` window.

## Project layout

| Path | Role |
|---|---|
| `noteblockify/song.py` | OpenNBS mapping, timing, layers, tempo |
| `noteblockify/mc_model.py` | Smooth voice-level MC placement |
| `noteblockify/preference_model.py` | Learned octave candidate ranker |
| `noteblockify/edit_model.py` | Optional unequal-length edit ranker |
| `noteblockify/hear.py` | Objective MIDI/NBS event scorer |
| `noteblockify/cli.py` | Main CLI |
| `tests/test_song.py` | Conversion and placement invariants |
| `mc_ranker.auto.best.pt` | Checked-in trained placement checkpoint |

## Verification

```bash
uv run python -m pytest tests/ -q
uv run python -m py_compile noteblockify/*.py
```

## License

MIT. GM mappings are replicated from
[OpenNBS/NoteBlockStudio](https://github.com/OpenNBS/NoteBlockStudio).
MuScriptor is a third-party dependency; audio quality is bounded by its
transcription output.
