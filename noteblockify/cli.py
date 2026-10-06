"""noteblockify command line: convert audio or MIDI into an NBS song.

Two stages, two files:

  1. ``--stage pre``  (default) — the pure conversion: OpenNBS maps,
     window folding, layers, tempo. Nothing learned, nothing thinned.
     Writes ``<name>.pre.nbs``.
  2. ``--stage model`` — accepts MIDI/audio directly or an existing
     ``.pre.nbs`` file and applies smooth voice-level MC placement; an
     optional learned ranker scores legal same-pitch-class octave choices.
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
        help="pre: convert MIDI/audio to a raw .pre.nbs (default); model: "
             "convert directly or refine an existing .pre.nbs file")
    parser.add_argument(
        "--out", dest="target", metavar="FILE", default=None,
        help="output .nbs path (default: <input>.<stage>.nbs)")
    parser.add_argument(
        "--model", default="medium", choices=("small", "medium", "large"),
        help="muscriptor model size for audio transcription (default: medium)")
    parser.add_argument(
        "--candidates", action="store_true",
        help="write all deterministic legal placement candidates instead of "
             "one model output")
    parser.add_argument(
        "--edit-candidates", action="store_true",
        help="write bounded drop/replace/add candidates for human A/B labeling")
    parser.add_argument(
        "--edit-ranker", type=Path, default=None,
        help="learned ranker for bounded note-edit candidates")
    parser.add_argument(
        "--pitch", action="store_true",
        help="preserve out-of-window absolute pitch using NBS fine tuning; "
             "not supported by Meteor physical-note playback")
    parser.add_argument(
        "--ranker", type=Path, default=None,
        help="PyTorch preference checkpoint for learned per-note decoding")
    args = parser.parse_args(argv)

    source = Path(args.source)
    if not source.exists():
        print(f"error: no such file: {source}", file=sys.stderr)
        return 2

    if args.stage == "model":
        from pynbs import read
        from noteblockify.song import arrange_pre
        from noteblockify.mc_model import (
            changed_keys, edit_candidates, refine, refine_candidates,
            refine_pitch)
        from noteblockify.edit_model import candidate_score, load_ranker as load_edit_ranker
        from noteblockify.preference_model import load_ranker

        if source.suffix.lower() == ".nbs":
            if not source.name.endswith(".pre.nbs"):
                print("error: NBS model input must end with .pre.nbs",
                      file=sys.stderr)
                return 2
            pre_song = read(source)
            default_stem = source.name[:-len(".pre.nbs")]
        else:
            if source.suffix.lower() in _AUDIO_SUFFIXES:
                midi = _transcribe(source, args.model)
            else:
                midi = source
            pre_song = arrange_pre(midi)
            default_stem = midi.stem
        target = Path(args.target) if args.target else \
            source.with_name(default_stem + ".model.nbs")
        if args.pitch:
            song = refine_pitch(pre_song)
            target = target.with_name(default_stem + ".pitch.nbs")
        elif args.candidates or args.edit_candidates:
            stem = default_stem
            output_dir = Path(args.target) if args.target else source.parent
            output_dir.mkdir(parents=True, exist_ok=True)
            generated = (edit_candidates(pre_song) if args.edit_candidates
                         else refine_candidates(pre_song))
            for policy, candidate in generated.items():
                output = output_dir / f"{stem}.{policy}.nbs"
                candidate.save(output)
                print(f"wrote {output} ({len(candidate.notes)} notes)")
            return 0
        if args.pitch:
            target.parent.mkdir(parents=True, exist_ok=True)
            song.save(target)
            keys = [note.key for note in song.notes]
            print(f"wrote {target} ({target.stat().st_size} bytes), "
                  f"{len(song.notes)} notes, pitch-preserving NBS mode")
            return 0
        if args.edit_ranker:
            ranker = load_edit_ranker(args.edit_ranker)
            if ranker is None:
                print(f"error: cannot load edit ranker {args.edit_ranker}",
                      file=sys.stderr)
                return 2
            generated = edit_candidates(pre_song)
            song = max(generated.values(),
                       key=lambda candidate: candidate_score(
                           ranker, pre_song, candidate))
            target = target.with_name(default_stem + ".edit.nbs")
        else:
            ranker = load_ranker(args.ranker)
            song = refine(pre_song, ranker=ranker)
        keys = [note.key for note in song.notes]
        if keys and not all(33 <= key <= 57 for key in keys):
            print("error: model produced a key outside the MC window",
                  file=sys.stderr)
            return 1
        target.parent.mkdir(parents=True, exist_ok=True)
        song.save(target)
        key_range = f"{min(keys)}-{max(keys)}" if keys else "no notes"
        dropped = len(pre_song.notes) - len(song.notes)
        print(f"wrote {target} ({target.stat().st_size} bytes), "
              f"{len(song.notes)} notes, dropped {dropped}, keys {key_range}, "
              f"changed keys {changed_keys(pre_song, song)}, "
              f"tempo {song.header.tempo} tps, "
              f"{song.header.song_layers} layers")
        return 0

    if source.suffix.lower() in _AUDIO_SUFFIXES:
        midi = _transcribe(source, args.model)
    else:
        midi = source
    target = Path(args.target) if args.target else midi.with_suffix(".pre.nbs")

    from noteblockify.song import arrange_pre

    song = arrange_pre(midi)

    keys = [note.key for note in song.notes]
    if keys and not all(0 <= key <= 255 for key in keys):
        print("error: raw pre key cannot fit in the NBS key byte",
              file=sys.stderr)
        return 1
    song.save(target)
    key_range = f"{min(keys)}-{max(keys)}" if keys else "no notes"
    print(f"wrote {target} ({target.stat().st_size} bytes), "
          f"{len(song.notes)} notes, keys {key_range}, "
          f"tempo {song.header.tempo} tps, {song.header.song_layers} layers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
