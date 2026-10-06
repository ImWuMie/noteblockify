# noteblockify — Audio/MIDI to Minecraft NBS

[中文说明](README_zh.md) | [Usage guide](USAGES.md) | [使用指南](USAGES_zh.md)


Convert any song into a Minecraft note block song that plays entirely
Convert any song into a Minecraft note block song with an explicit raw
baseline and a separate MC-constrained refinement stage.

Pipeline:

```
audio (mp3/wav)  ──muscriptor──▶  MIDI  ──pre──▶  raw .pre.nbs
                                               └─model──▶ .model.nbs
MIDI file        ──────────────────────────────▶  raw .pre.nbs
```

- **Faithful OpenNBS import core** — the 128-entry GM program map, the
  GM drum map, 2x time precision grid, and layer band layout are
  replicated from [OpenNBS/NoteBlockStudio](https://github.com/OpenNBS/NoteBlockStudio)
  (MIT), so a converted MIDI sounds the way OpenNBS would import it.
- **Raw pre baseline** — OpenNBS mapping, timing grid, layers, velocity,
  and panning are preserved. Raw keys outside 33–57 are intentionally
  left untouched so this file is the most faithful listening reference.
- **MC-constrained model stage** — reads the saved `.pre.nbs`, changes
  keys only, and guarantees every result is in 33–57. Every note may choose a
  legal same-pitch-class octave; dynamic programming keeps each voice in a
  stable octave/register path, preserves local melodic contour, gives lower
  raw-register voices lower legal targets, and avoids avoidable same-tick
  unisons. No notes,
  onsets, instruments, or layers are deleted.
- **Optional edit stage** — `--edit-candidates` creates bounded `drop`,
  `replace`, and source-grounded `add` alternatives for human A/B labeling.
  These alternatives are not enabled by the legacy key-only ranker.
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

# Stage 1: transcribe (if needed) and write the raw baseline
noteblockify --in a.mp3                 # writes a.pre.nbs
noteblockify --in song.mid              # MIDI converts directly, no GPU
# Stage 2: refine the saved baseline for Minecraft
noteblockify --in a.pre.nbs --stage model  # writes a.model.nbs
# Generate several legal candidates for human A/B listening
noteblockify --in a.pre.nbs --stage model --candidates --out candidates/
# Generate note-edit alternatives for the problematic low-register notes
noteblockify --in a.pre.nbs --stage model --edit-candidates --out edit-candidates/

```

Batch-convert a folder:

```bash
for f in data/midi/*.mid; do noteblockify --in "$f"; done
```

The output `.nbs` plays in [Open Note Block Studio](https://opennbs.org/)
and in Meteor Client's NoteBot (drop it into
`.minecraft/meteor-client/notebot/`). Note that Meteor ignores velocity
and panning — see USAGES.md for what survives in-game.

The MC decoder first generates hard-valid candidates, then a PyTorch
preference ranker chooses among them. Without `mc_ranker.pt` it falls back
to `balanced`. Record human A/B choices and train it:

```bash
noteblockify-feedback --data data/preferences.jsonl \
  --pre song.pre.nbs \
  --preferred candidates/song.high.nbs \
  --rejected candidates/song.low.nbs \
  --note "clearer lead"
noteblockify-train --data data/preferences.jsonl
```

The weights come from human A/B choices, not fabricated automatic labels.
The raw pre file remains available for direct listening comparison.

For edit candidates, record preferences in a separate JSONL file and train
the edit ranker; candidates may have different note counts:

```bash
noteblockify-feedback --data data/edit_preferences.jsonl \\
  --pre song.pre.nbs \\
  --preferred edit-candidates/song.replace.nbs \\
  --rejected edit-candidates/song.drop.nbs
noteblockify-train-edits --edit-data data/edit_preferences.jsonl \\
  --output mc_edit_ranker.pt
```

## Project layout

| file | role |
|---|---|
| `noteblockify/song.py` | converter: OpenNBS maps, folding, layers, tempo |
| `noteblockify/mc_model.py` | MC-window constrained placement optimizer |
| `noteblockify/hear.py` | NBS→MIDI event scorer |
| `noteblockify/cli.py` | `noteblockify` CLI entry point |
| `tests/test_song.py` | converter and MC-constraint invariants |

`sounds/` contains the 16 vanilla instrument OGGs from OpenNBS (MIT)
used by the scorer and available for preview rendering.

## License

MIT — see LICENSE. The GM program/drum maps are replicated from
OpenNBS/NoteBlockStudio (MIT). muscriptor is a third-party dependency;
audio transcription quality is bounded by it.
