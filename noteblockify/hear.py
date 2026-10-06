"""Score event fidelity against the raw converter mapping.

The pre file is intentionally the listening reference.  Its mapped MIDI
keys are kept exactly, including keys outside Minecraft's playable window.
The MC model may then move a key by whole octaves, so matching uses pitch
class rather than absolute octave.  Timing and instrument are still matched
one-to-one.  This score measures whether conversion preserved the musical
events; it does not claim to judge whether one legal octave sounds better
than another.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pynbs

from noteblockify.song import _DRUM, _PROGRAM, _events

WEIGHTS = (0.6, 0.4)
TIME_TOLERANCE = 0.06  # seconds
DRIFT_ALLOWANCE = 0.005  # seconds of drift per second of song
PITCH_TOLERANCE = 1  # semitone


@dataclass
class Score:
    f1: float
    instrument: float

    @property
    def total(self) -> float:
        return sum(w * v for w, v in zip(WEIGHTS, (self.f1, self.instrument)))


from noteblockify.song import _DRUM, _PROGRAM, _diversify, _events


def _midi_events(midi_path: str | Path):
    """Source events as (seconds, key, instrument), before MC placement."""
    path = Path(midi_path)
    events, programs, tpb, usec, _names = _events(path)
    usec = usec or 500_000
    base = min(e[0] for e in events) if events else 0
    channels = sorted({e[1] for e in events})
    choice = _diversify(programs, channels)
    out = []
    for pos, ch, note, vel in events:
        seconds = (pos - base) * usec / 1e6 / tpb
        if ch == 9:
            instrument, key = _DRUM[note]
            key += 33
        else:
            instrument, octave = _PROGRAM[programs[ch]]
            key = note - 21 + 12 * octave
            instrument = choice[ch]
        out.append((seconds, key, instrument))
    return out


def _nbs_events(song: pynbs.File):
    """Result events as (seconds, key, instrument)."""
    tps = song.header.tempo or 10.0
    return [(n.tick / tps, n.key, n.instrument) for n in song.notes]


def _match(midi: list[tuple], nbs: list[tuple]):
    """One-to-one greedy match, earliest NBS note first.

    Each NBS note takes the closest still-free MIDI event within the
    time and pitch tolerances. The time tolerance grows with song
    length to absorb tempo quantization drift. Events are bucketed by
    key so each search only touches candidates of nearby pitch.
    """
    free = sorted(midi, key=lambda e: e[0])
    span = max((e[0] for e in free), default=0.0)
    tolerance = TIME_TOLERANCE + DRIFT_ALLOWANCE * span
    # Bucket source events by key for O(1) candidate lookup.
    # Bucket source events by pitch class. The MC model may change octave,
    # but it must preserve pitch class.
    by_class: dict[int, list[tuple[float, int, int]]] = {}
    order = {id(e): i for i, e in enumerate(free)}
    taken = [False] * len(free)
    for e in free:
        by_class.setdefault(e[1] % 12, []).append((e[0], e[2], id(e)))
    pairs = []
    for j, (t, k, i) in enumerate(nbs):
        best, best_cost = None, None
        for mt, mi, eid in by_class.get(k % 12, []):
            if mt > t + tolerance:
                continue
            if abs(mt - t) > tolerance:
                continue
            idx = order[eid]
            if taken[idx]:
                continue
            cost = abs(mt - t) + 0.01 * (mi != i)
            if best_cost is None or cost < best_cost:
                best, best_cost = (mt, k, idx), cost
        if best is not None:
            taken[best[2]] = True
            pairs.append((free[best[2]], j))
    return pairs


def compare_song(midi_path: str | Path, song: pynbs.File) -> Score:
    midi_events = _midi_events(midi_path)
    nbs_events = _nbs_events(song)
    pairs = _match(midi_events, nbs_events)

    hit = len(pairs)
    precision = hit / len(nbs_events) if nbs_events else 0.0
    recall = hit / len(midi_events) if midi_events else 0.0
    f1 = 2 * precision * recall / (precision + recall) if hit else 0.0

    agree = sum(1 for midi_event, j in pairs
                if midi_event[2] == nbs_events[j][2])
    inst = agree / hit if hit else 0.0
    return Score(f1=f1, instrument=inst)


def compare(midi_path: str | Path, nbs_path: str | Path) -> Score:
    return compare_song(midi_path, pynbs.read(nbs_path))
