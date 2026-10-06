"""noteblockify command line: convert audio or MIDI into an NBS song.

Two stages, two files:

  1. ``--stage pre``  (default) — the pure conversion: OpenNBS maps,
     window folding, layers, tempo. Nothing learned, nothing thinned.
     Writes ``<name>.pre.nbs``.
  2. ``--stage model`` — takes the pre conversion and applies the octave
     model's per-voice shifts (majority vote) plus centroid folding.
     Writes ``<name>.model.nbs``.

Audio inputs (mp3/wav/flac/ogg/m4a) are transcribed to MIDI by
muscriptor first; the transcription is cached as <input>.mid next to the
input and reused on the next run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_AUDIO_SUFFIXES = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}


def _transcribe(source: Path, model_size: str) -> Path:
    cached = source.with_suffix(".mid")
    if cached.exists():
        print(f"reusing {cached}")
        return cached
    from muscriptor import TranscriptionModel

    model = TranscriptionModel.load_model(model_size)
    cached.write_bytes(model.transcribe_to_midi(source))
    print(f"transcribed {cached} ({cached.stat().st_size} bytes)")
    return cached


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="noteblockify",
        description="Convert audio or MIDI into a Minecraft NBS song.")
    parser.add_argument(
        "--in", dest="source", required=True, metavar="FILE",
        help="input audio (mp3/wav/flac/ogg/m4a) or MIDI (.mid/.midi)")
    parser.add_argument(
        "--stage", choices=("pre", "model"), default="pre",
        help="pre: pure conversion only (default); model: octave model "
             "applied on top of the pre conversion")
    parser.add_argument(
        "--out", dest="target", metavar="FILE", default=None,
        help="output .nbs path (default: <input>.<stage>.nbs)")
    parser.add_argument(
        "--model", default="medium", choices=("small", "medium", "large"),
        help="muscriptor model size for audio transcription (default: medium)")
    args = parser.parse_args(argv)

    source = Path(args.source)
    if not source.exists():
        print(f"error: no such file: {source}", file=sys.stderr)
        return 2

    if source.suffix.lower() in _AUDIO_SUFFIXES:
        midi = _transcribe(source, args.model)
    else:
        midi = source
    target = Path(args.target) if args.target else \
        midi.with_suffix(f".{args.stage}.nbs")

    if args.stage == "pre":
        from noteblockify.song import arrange_pre

        song = arrange_pre(midi)
    else:
        from noteblockify.song import arrange

        song = arrange(midi)

    keys = [note.key for note in song.notes]
    assert keys and 0 <= min(keys) and max(keys) <= 87
    song.save(target)
    print(f"wrote {target} ({target.stat().st_size} bytes), "
          f"{len(song.notes)} notes, keys {min(keys)}-{max(keys)}, "
          f"tempo {song.header.tempo} tps, {song.header.song_layers} layers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
