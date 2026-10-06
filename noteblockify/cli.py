"""noteblockify command line: convert audio or MIDI into an NBS song.

Usage:
    noteblockify --in a.mp3               # writes a.nbs next to the input
    noteblockify --in a.mp3 --out b.nbs
    noteblockify --in song.mid            # MIDI converts directly, no GPU

Audio inputs (mp3/wav/flac) are transcribed to MIDI by muscriptor first;
the transcription is cached as <input>.mid next to the input and reused on
the next run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_AUDIO_SUFFIXES = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="noteblockify",
        description="Convert audio or MIDI into a Minecraft NBS song.")
    parser.add_argument(
        "--in", dest="source", required=True, metavar="FILE",
        help="input audio (mp3/wav/flac/ogg/m4a) or MIDI (.mid/.midi)")
    parser.add_argument(
        "--out", dest="target", metavar="FILE", default=None,
        help="output .nbs path (default: the input name with a .nbs suffix)")
    parser.add_argument(
        "--model", default="medium", choices=("small", "medium", "large"),
        help="muscriptor model size for audio transcription (default: medium)")
    parser.add_argument(
        "--vocal-instrument", dest="vocal", type=int, default=None,
        metavar="0-15",
        help="vanilla instrument for transcribed vocal tracks "
             "(voice/vocal/lead; 7=bell, 15=pling, 6=flute default)")
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="print per-voice octave decisions and ranges")
    parser.add_argument(
        "--max-per-tick", dest="max_per_tick", type=int, default=4,
        metavar="N",
        help="cap simultaneous notes per tick to thin dense transcriptions "
             "(default: 4; 0 disables)")
    args = parser.parse_args(argv)

    source = Path(args.source)
    if not source.exists():
        print(f"error: no such file: {source}", file=sys.stderr)
        return 2
    target = Path(args.target) if args.target else source.with_suffix(".nbs")

    midi = source
    if source.suffix.lower() in _AUDIO_SUFFIXES:
        cached = source.with_suffix(".mid")
        if cached.exists():
            print(f"reusing {cached}")
        else:
            from muscriptor import TranscriptionModel

            model = TranscriptionModel.load_model(args.model)
            cached.write_bytes(model.transcribe_to_midi(source))
            print(f"transcribed {cached} ({cached.stat().st_size} bytes)")
        midi = cached

    from noteblockify.song import arrange

    try:
        song = arrange(midi, vocal_instrument=args.vocal,
                       max_per_tick=args.max_per_tick or None)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    keys = [note.key for note in song.notes]
    assert keys and 0 <= min(keys) and max(keys) <= 87
    song.save(target)
    info = getattr(song, "octave_info", None)
    if info:
        if info["model"]:
            verdicts = " ".join(
                f"ch{c}:{r['shift']:+d}"
                for c, r in info["channels"].items() if r["shift"] or True)
            print(f"octave: model-assigned, voices: {verdicts}")
        else:
            print("octave: no model file, pure folding (run "
                  "`uv run python -m noteblockify.octave` to train)")
        if args.verbose:
            print("voices:")
            inst_names = ("harp", "bass", "basedrum", "snare", "hat",
                          "guitar", "flute", "bell", "chime", "xylophone",
                          "iron xylophone", "cow bell", "didgeridoo",
                          "bit", "banjo", "pling")
            for c, r in info["channels"].items():
                inst = r["instrument"]
                inst = inst_names[inst] if isinstance(inst, int) else inst
                print(f"  ch{c} {r['name'][:18]:20s} -> {inst:12s} "
                      f"{r['notes']:4d} notes  raw {r['raw_range'][0]:2d}-"
                      f"{r['raw_range'][1]:2d}  shift {r['shift']:+d}  "
                      f"final {r['final_range'][0]:2d}-{r['final_range'][1]:2d}")
    print(f"wrote {target} ({target.stat().st_size} bytes), "
          f"{len(song.notes)} notes, keys {min(keys)}-{max(keys)}, "
          f"tempo {song.header.tempo} tps, {song.header.song_layers} layers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
