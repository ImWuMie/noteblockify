"""Score an NBS arrangement by mapping it back to MIDI and comparing notes.

The result NBS is read as note events — onset in seconds (tick / tempo),
pitch as the key actually heard, instrument as the vanilla instrument —
and matched one-to-one against the source MIDI's events. Onsets are
compared in seconds; the tolerance absorbs the NBS tempo field's
hundredth-of-a-tps quantization, which on long songs accumulates beyond
any fixed window. Two measures:

- Note F1 (weight 0.6): greedy one-to-one match within the time and
  pitch tolerances.
- Instrument agreement (weight 0.4): the fraction of matched pairs whose
  vanilla instrument is the one the OpenNBS map prescribes for that MIDI
  note.

Both sides are compared in "heard pitch" space: the MIDI side passes
through the same OpenNBS program/drum maps the importer used, so an
octave shift the map prescribes is not an error.
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


from noteblockify.song import (_BASS, _DRUM, _PROGRAM, _NAME, _diversify, _events,
                      _fold, _model_shifts)


def _midi_events(midi_path: str | Path):
    """Source events as (seconds, key, instrument).

    The same expectation path the converter uses: program/drum maps,
    channel diversification, and range folding, so the score measures
    conversion fidelity rather than rule disagreement.
    """
    path = Path(midi_path)
    events, programs, tpb, usec = _events(path)
    usec = usec or 500_000
    base = min(e[0] for e in events) if events else 0
    channels = sorted({e[1] for e in events})
    choice = _diversify(programs, channels)
    prepared = []
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
        prepared.append((seconds, instrument, key))
    # Neighbour context per channel, mirroring the converter.
    where: dict[int, list[int]] = {}
    for i, _row in enumerate(prepared):
        where.setdefault(events[i][1], []).append(i)
    rows = [None] * len(prepared)
    for idxs in where.values():
        for n, i in enumerate(idxs):
            _s, inst, key = prepared[i]
            prev_key = prepared[idxs[n - 1]][2] if n > 0 else key
            next_key = prepared[idxs[n + 1]][2] if n + 1 < len(idxs) else key
            rows[i] = (inst, key, prev_key, next_key, events[i][3])
    shifts = _model_shifts(rows)
    for (seconds, instrument, key), shift in zip(prepared, shifts):
        shifted = max(0, min(87, key + 12 * shift))
        folded = _fold(shifted, instrument)
        # Same drop rule as the converter: bass folded up a whole octave
        # or more is expected to be deleted, not played.
        if instrument in _BASS and folded - shifted >= 12:
            continue
        out.append((seconds, folded, instrument))
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
    by_key: dict[int, list[tuple[float, int, int]]] = {}
    order = {id(e): i for i, e in enumerate(free)}
    taken = [False] * len(free)
    for e in free:
        by_key.setdefault(e[1], []).append((e[0], e[2], id(e)))
    pairs = []
    for j, (t, k, i) in enumerate(nbs):
        best, best_cost = None, None
        for mk in (k - PITCH_TOLERANCE, k, k + PITCH_TOLERANCE):
            for mt, mi, eid in by_key.get(mk, []):
                if mt > t + tolerance:
                    continue
                if abs(mt - t) > tolerance:
                    continue
                idx = order[eid]
                if taken[idx]:
                    continue
                cost = abs(mt - t) + abs(mk - k) + 0.01 * (mi != i)
                if best_cost is None or cost < best_cost:
                    best, best_cost = (mt, mk, idx), cost
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
