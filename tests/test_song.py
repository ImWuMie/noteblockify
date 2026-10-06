"""Smoke tests for the conversion pipeline.

Synthetic MIDI files exercise the edge cases the converter promises to
handle; a real corpus song is not required (tests run without data/).
"""

from pathlib import Path

import mido
import pynbs
import pytest

from noteblockify.hear import compare
from noteblockify.mc_model import edit_candidates, refine, refine_candidates
import noteblockify.mc_model as mc_model
from noteblockify.preference_model import (
    append_preference,
    assignment_features,
    candidate_score,
    load_ranker,
    train_preferences,
)
from noteblockify.song import (
    FOLD_HI,
    FOLD_LO,
    _fold,
    arrange,
    arrange_pre,
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


def test_pre_is_raw_and_model_only_applies_mc_constraints(tmp_path):
    midi = make_midi(tmp_path / "raw.mid", [
        (0, 0, 24, 100),   # mapped raw key 3; needs +3 octaves
        (240, 0, 60, 90),  # mapped raw key 39; must not move
        (480, 0, 96, 80),  # mapped raw key 75; needs -2 octaves
    ])
    pre = arrange_pre(midi)
    model = refine(pre)

    assert [note.key for note in pre.notes] == [3, 39, 75]
    assert len(model.notes) == len(pre.notes)
    assert all(FOLD_LO <= note.key <= FOLD_HI for note in model.notes)
    for before, after in zip(pre.notes, model.notes):
        assert (after.key - before.key) % 12 == 0
        assert (before.tick, before.layer, before.instrument,
                before.velocity, before.panning) == (
                    after.tick, after.layer, after.instrument,
                    after.velocity, after.panning)
    assert model.notes[1].key == pre.notes[1].key


def test_preference_candidates_are_hard_valid_and_scorable(tmp_path):
    midi = make_midi(tmp_path / "candidates.mid", [
        (0, 0, 24, 100),
        (240, 0, 60, 90),
        (480, 0, 96, 80),
    ])
    pre = arrange_pre(midi)
    candidates = refine_candidates(pre)

    assert "balanced" in candidates
    assert len(candidates) >= 1
    for candidate in candidates.values():
        assert len(candidate.notes) == len(pre.notes)
        assert all(FOLD_LO <= note.key <= FOLD_HI
                   for note in candidate.notes)
        assert assignment_features(pre, candidate)
        assert candidate_score(None, pre, candidate) == 0.0
        for before, after in zip(pre.notes, candidate.notes):
            assert (after.key - before.key) % 12 == 0
            assert (before.tick, before.layer, before.instrument,
                    before.velocity, before.panning) == (
                        after.tick, after.layer, after.instrument,
                        after.velocity, after.panning)


def test_preference_ranker_learns_an_ab_choice(tmp_path):
    midi = make_midi(tmp_path / "train.mid", [
        (0, 0, 24, 100),
        (240, 0, 60, 90),
        (480, 0, 96, 80),
    ])
    pre = arrange_pre(midi)
    candidates = refine_candidates(pre)
    preferred = tmp_path / "preferred.nbs"
    rejected = tmp_path / "rejected.nbs"
    pre_path = tmp_path / "pre.nbs"
    pre.save(pre_path)
    candidates["high"].save(preferred)
    candidates["low"].save(rejected)
    data = tmp_path / "preferences.jsonl"
    append_preference(data, pre_path, preferred, rejected)

    weights = tmp_path / "ranker.pt"
    train_preferences(data, weights, steps=80)
    ranker = load_ranker(weights)
    assert ranker is not None
    assert candidate_score(ranker, pre, candidates["high"]) > \
        candidate_score(ranker, pre, candidates["low"])


def test_edit_candidates_can_drop_or_replace_low_fold_jump(tmp_path):
    midi = make_midi(tmp_path / "edit.mid", [
        (0, 0, 61, 100),
        (240, 0, 59, 100),
        (480, 0, 52, 100),
        (720, 0, 49, 100),
    ])
    pre = arrange_pre(midi)
    candidates = edit_candidates(pre)
    assert {"balanced", "drop", "replace"} <= candidates.keys()
    assert len(candidates["drop"].notes) < len(pre.notes)
    assert len(candidates["replace"].notes) == len(pre.notes)
    for candidate in candidates.values():
        assert all(FOLD_LO <= note.key <= FOLD_HI
                   for note in candidate.notes)
    if "add" in candidates:
        add_path = tmp_path / "add.nbs"
        candidates["add"].save(add_path)
        parsed = pynbs.read(add_path)
        assert len(parsed.notes) == len(candidates["add"].notes)
        assert all(0 <= note.layer < len(parsed.layers)
                   for note in parsed.notes)


def test_ranked_decoder_can_mix_choices_per_note(tmp_path, monkeypatch):
    midi = make_midi(tmp_path / "mix.mid", [
        (0, 0, 24, 100),
        (240, 0, 60, 90),
        (480, 0, 96, 80),
        (720, 1, 24, 100),
        (960, 1, 60, 90),
        (1200, 1, 96, 80),
    ])
    pre = arrange_pre(midi)
    balanced = refine_candidates(pre)["balanced"]

    class PreferMixed:
        def eval(self):
            return self

    def fake_load(_path=None):
        return PreferMixed()

    def fake_scores(_model, song, index, candidate_keys, contexts=None):
        # Alternate between high and low legal choices by source voice.
        if mc_model._voice_id(song, song.notes[index]) == "ch0":
            target = max(candidate_keys)
        else:
            target = min(candidate_keys)
        return [1.0 if key == target else 0.0 for key in candidate_keys]

    monkeypatch.setattr("noteblockify.preference_model.load_ranker", fake_load)
    monkeypatch.setattr("noteblockify.preference_model.score_options", fake_scores)
    mixed = mc_model.refine(pre)

    assert all(FOLD_LO <= note.key <= FOLD_HI for note in mixed.notes)
    assert len(mixed.notes) == len(pre.notes)
    assert any(a.key != b.key for a, b in zip(balanced.notes, mixed.notes))


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
