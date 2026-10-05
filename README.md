# noteblockify — Audio/MIDI to Minecraft NBS

Convert any song into a Minecraft note block song that plays entirely
inside the two-octave vanilla window (F#3–F#5), with musically sensible
octave placement per voice.

Pipeline:

```
audio (mp3/wav)  ──muscriptor──▶  MIDI  ──noteblockify──▶  .nbs
MIDI file        ──────────────────────────────▶  .nbs
```

- **Faithful OpenNBS import core** — the 128-entry GM program map, the
  GM drum map, 2x time precision grid, and layer band layout are
  replicated from [OpenNBS/NoteBlockStudio](https://github.com/OpenNBS/NoteBlockStudio)
  (MIT), so a converted MIDI sounds the way OpenNBS would import it.
- **Hard window handling** — every note is folded by octave into keys
  33–57; vanilla note blocks cannot sound anything else.
- **Learned octave placement** — a small weighted MLP (`octave.pt`)
  decides which octave each voice sits in, trained on the MIDI itself
  (median-per-channel snapped to whole octaves). Bass notes that would
  have to fold up into the melody register are deleted instead of
  muddying the mix.
- **No dropped notes** — channel layer bands grow as tall as needed;
  nothing is silently discarded on collision.
- **Long-song regridding** — songs longer than the 65535-tick limit are
  proportionally re-gridded instead of failing.
- **Stereo layer panning** — layers carry the sound field (notes follow
  their layer), spread across the stereo image.
- **Objective scoring** — `noteblockify.hear` maps the NBS back to MIDI events
  and scores note F1 plus instrument agreement, with a drift-aware time
  tolerance that absorbs the NBS tempo field's quantization.

## Quick start

```bash
uv sync

# Convert audio (transcribes with muscriptor first; needs ~5 GB VRAM)
uv run python main.py            # expects yuai.mp3, or edit AUDIO in main.py

# Convert a MIDI directly
uv run python -c "from noteblockify.song import arrange; arrange('song.mid').save('song.nbs')"

# Batch-convert a folder
uv run python -c "
from pathlib import Path
from noteblockify.song import arrange
for mid in Path('data/midi').glob('*.mid'):
    arrange(mid).save(Path('out') / (mid.stem + '.nbs'))
"
```

The output `.nbs` plays in [Open Note Block Studio](https://opennbs.org/)
and in Meteor Client's NoteBot (drop it into
`.minecraft/meteor-client/notebot/`). Note that Meteor ignores velocity
and panning — see USAGES.md for what survives in-game.

## Retrain the octave model

```bash
uv run python -m noteblockify.octave              # weighted MLP, ~25 s on GPU
uv run python score_model.py             # full report card
```

Labels are derived from the MIDI itself — no manual annotation. Add
songs to `data/midi/` and retrain to specialize.

## Project layout

| file | role |
|---|---|
| `noteblockify/song.py` | converter: OpenNBS maps, folding, layers, tempo |
| `noteblockify/octave.py` | octave model, training, features |
| `noteblockify/hear.py` | NBS→MIDI event scorer |
| `main.py` | end-to-end audio→NBS entry point |
| `score_model.py` | model evaluation harness |

`sounds/` contains the 16 vanilla instrument OGGs from OpenNBS (MIT)
used by the scorer and available for preview rendering.

## License

MIT — see LICENSE. The GM program/drum maps are replicated from
OpenNBS/NoteBlockStudio (MIT). muscriptor is a third-party dependency;
audio transcription quality is bounded by it.
