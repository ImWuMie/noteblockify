"""Smoke tests for the conversion pipeline.

Synthetic MIDI files exercise the edge cases the converter promises to
handle; a real corpus song is not required (tests run without data/).
"""

from pathlib import Path

import mido
import pynbs
import pytest

from noteblockify.hear import compare
from noteblockify.song import (
    FOLD_HI,
    FOLD_LO,
    _fold,
    arrange,
)


def make_midi(path, notes, tpb=480, tempo=500_000):
    mid = mido.MidiFile(ticks_per_beat=tpb)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))
    last = 0
    for pos, ch, note, vel in notes:
        track.append(mido.Message(
            "note_on", channel=ch, note=note, velocity=vel,
            time=pos - last))
        last = pos
    mid.save(path)
    return Path(path)


def test_fold_hard_window():
    # Out-of-window keys fold by octave to the same pitch class.
    assert FOLD_LO <= _fold(0) <= FOLD_HI
    assert FOLD_LO <= _fold(87) <= FOLD_HI
    # In-window keys stay put.
    for k in (FOLD_LO, 45, FOLD_HI):
        assert _fold(k) == k


def test_fold_keeps_pitch_class():
    for key in range(88):
        folded = _fold(key)
        assert FOLD_LO <= folded <= FOLD_HI
        assert (folded - key) % 12 == 0


def test_all_keys_inside_mc_window(tmp_path):
    midi = make_midi(tmp_path / "extreme.mid", [
        (0, 0, 0, 100),      # lowest MIDI note
        (480, 0, 127, 100),  # highest
        (960, 0, 60, 100),
    ])
    song = arrange(midi)
    assert song.notes
    assert all(FOLD_LO <= n.key <= FOLD_HI for n in song.notes)


def test_no_note_dropped_on_collisions(tmp_path):
    # 50 notes on the same tick and channel: all must survive.
    notes = [(0, 0, 60 + i, 100) for i in range(50)]
    midi = make_midi(tmp_path / "chord.mid", notes)
    song = arrange(midi)
    assert len(song.notes) == 50


def test_drum_out_of_range_dropped(tmp_path):
    # GM drum notes outside 24..84 are dropped; 36 (kick) survives.
    midi = make_midi(tmp_path / "drums.mid", [
        (0, 9, 20, 100),   # out of range
        (10, 9, 36, 100),  # kick
        (20, 9, 100, 100), # out of range
    ])
    song = arrange(midi)
    assert len(song.notes) == 1
    # _DRUM[36] = (basedrum=2, key 6+33=39): the kick's mapped key.
    assert song.notes[0].instrument == 2
    assert song.notes[0].key == 39


def test_empty_midi(tmp_path):
    midi = make_midi(tmp_path / "empty.mid", [])
    song = arrange(midi)
    assert song.notes == []


def test_long_song_regridded(tmp_path):
    # 10 million ticks: far beyond the 65535 limit, must still fit.
    midi = make_midi(tmp_path / "long.mid", [
        (0, 0, 60, 100),
        (10_000_000, 0, 64, 100),
    ])
    song = arrange(midi)
    assert song.header.song_length <= 65535
    assert len(song.notes) == 2


def test_faithful_round_trip(tmp_path):
    notes = [(0, 0, 60, 90), (240, 0, 64, 80), (480, 1, 67, 100)]
    midi = make_midi(tmp_path / "simple.mid", notes)
    song = arrange(midi)
    out = tmp_path / "simple.nbs"
    song.save(out)
    back = pynbs.read(out)
    assert len(back.notes) == len(notes)
    score = compare(str(midi), str(out))
    assert score.total > 0.95
