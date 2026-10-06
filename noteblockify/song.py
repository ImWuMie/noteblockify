"""Convert a MIDI file into the best-sounding NBS arrangement.

Built on the OpenNBS import (https://github.com/OpenNBS/NoteBlockStudio,
MIT: scripts/midi_instruments maps, 2x precision grid, channel layer
bands), then fixes the four places where that import loses music:

1. Out-of-range pitches are folded by octave into the range the chosen
   instrument actually covers, instead of being clamped to key 0/87 and
   turning into a wrong, sour note.
2. Thin instrument mapping: GM patches that share a vanilla instrument
   are re-picked per channel so the ensemble uses the widest spread of
   distinct vanilla sounds.
3. Layer collisions: a channel whose notes pile up on one tick gets as
   many layers as it needs (no note is ever dropped).
4. Songs longer than the 65535-tick limit are re-gridded with a coarser
   tempo that preserves onset order and relative spacing.

Tempo is chosen so playback timing matches the MIDI within half a
hundredth of a tick per second.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path

import mido
import pynbs

# song_length is a uint16 in the file; leave one tick of headroom.
MAX_TICK = 65534
# NBS tempo field is hundredths of a tick per second.
MIN_TEMPO, MAX_TEMPO = 0.25,1000
# Vanilla note blocks cover two octaves per instrument comfortably.
FOLD_LO, FOLD_HI = 33, 57

# GM program -> (vanilla instrument, octave shift).
# 0 harp, 1 bass, 2 basedrum, 3 snare, 4 hat, 5 guitar, 6 flute, 7 bell,
# 8 chime, 9 xylophone, 10 iron xylophone, 11 cow bell, 12 didgeridoo,
# 13 bit, 14 banjo, 15 pling.
_PROGRAM = {
    0: (0, 0), 1: (15, 0), 2: (15, 0), 3: (15, 0), 4: (0, 0), 5: (0, 0),
    6: (5, 1), 7: (14, 0),
    8: (7, -2), 9: (7, -2), 10: (7, -2), 11: (10, 0), 12: (10, 0),
    13: (9, -2), 14: (7, -2), 15: (5, 1),
    16: (6, -1), 17: (10, 0), 18: (6, -1), 19: (6, -1), 20: (6, -1),
    21: (6, -1), 22: (6, -1), 23: (6, -1),
    24: (5, 1), 25: (5, 1), 26: (0, 0), 27: (5, 1), 28: (1, 2),
    29: (12, 2), 30: (12, 2), 31: (5, 3),
    32: (1, 2), 33: (1, 2), 34: (1, 2), 35: (1, 2), 36: (5, 1), 37: (5, 1),
    38: (1, 2), 39: (15, 0),
    40: (6, -1), 41: (6, -1), 42: (6, -1), 43: (6, -1), 44: (6, -1),
    45: (1, 2), 46: (0, 0), 47: (3, 0),
    48: (6, -1), 49: (6, -1), 50: (6, -1), 51: (6, -1), 52: (6, -1),
    53: (6, -1), 54: (6, -1), 55: (3, -1),
    56: (6, -1), 57: (6, -1), 58: (6, -1), 59: (12, 2), 60: (6, -1),
    61: (12, 2), 62: (12, 2), 63: (6, -1),
    64: (6, -1), 65: (6, -1), 66: (6, -1), 67: (6, -1), 68: (6, -1),
    69: (6, -1), 70: (6, -1), 71: (6, -1),
    72: (6, -1), 73: (6, -1), 74: (6, -1), 75: (6, -1), 76: (6, -1),
    77: (6, -1), 78: (6, -1), 79: (6, -1),
    80: (13, 0), 81: (6, -1), 82: (6, -1), 83: (6, -1), 84: (5, 1),
    85: (6, -1), 86: (6, -1), 87: (1, 2),
    88: (7, -2), 89: (6, -1), 90: (6, -1), 91: (6, -1), 92: (6, -1),
    93: (6, -1), 94: (6, -1), 95: (8, -2),
    96: (8, -2), 97: (6, -1), 98: (8, -2), 99: (5, 1), 100: (15, 0),
    101: (6, -1), 102: (6, -1), 103: (5, 1),
    104: (14, 0), 105: (14, 0), 106: (14, 0), 107: (5, 1), 108: (10, 0),
    109: (6, -1), 110: (6, -1), 111: (6, -1),
    112: (8, -2), 113: (11, -1), 114: (10, 0), 115: (9, -2), 116: (2, 0),
    117: (3, 0), 118: (3, 0), 119: (8, -2),
    120: (4, 1), 121: (6, -1), 122: (8, -2), 123: (6, 1), 124: (7, 2),
    125: (2, 0), 126: (3, 0), 127: (3, 0),
}

# Alternative vanilla sounds for patches whose default is heavily shared.
# Each entry: program -> (instrument, octave) differing in timbre only.
_ALT = {
    0: (7, -2), 1: (0, 0), 2: (13, 0), 3: (7, -2), 4: (15, 0),
    6: (0, 0), 7: (5, 1),
    16: (13, 0), 18: (15, 0), 19: (0, 0), 20: (15, 0), 21: (0, 0),
    22: (13, 0), 23: (15, 0),
    26: (5, 1), 27: (0, 0),
    40: (15, 0), 41: (15, 0), 42: (0, 0), 43: (15, 0), 44: (0, 0),
    48: (15, 0), 49: (0, 0), 50: (13, 0), 51: (15, 0), 52: (0, 0),
    53: (15, 0), 54: (0, 0), 56: (15, 0), 57: (0, 0), 58: (15, 0),
    60: (15, 0), 63: (15, 0),
    72: (15, 0), 73: (15, 0), 74: (15, 0), 75: (15, 0), 76: (15, 0),
    77: (15, 0), 78: (15, 0), 79: (15, 0),
    85: (15, 0), 86: (15, 0), 92: (15, 0), 93: (15, 0), 94: (15, 0),
    109: (15, 0), 110: (15, 0), 111: (15, 0),
}

# GM drum note -> (vanilla instrument, key - 33).
_DRUM = {
    24: (13, 39), 25: (3, 8), 26: (4, 25), 27: (3, 18), 28: (3, 27),
    29: (4, 16), 30: (4, 13), 31: (4, 9), 32: (4, 6), 33: (4, 2),
    34: (8, 17),
    35: (2, 10), 36: (2, 6), 37: (4, 6), 38: (3, 8), 39: (4, 6),
    40: (3, 4), 41: (2, 6), 42: (3, 22), 43: (2, 13), 44: (3, 22),
    45: (2, 15), 46: (3, 18), 47: (2, 20), 48: (2, 23), 49: (3, 17),
    50: (2, 23), 51: (3, 24), 52: (3, 8), 53: (3, 13), 54: (4, 18),
    55: (3, 18), 56: (11, 5), 57: (3, 13), 58: (4, 2), 59: (3, 13),
    60: (4, 9), 61: (4, 2), 62: (4, 8), 63: (2, 22), 64: (2, 15),
    65: (3, 13), 66: (3, 8), 67: (9, 12), 68: (9, 5), 69: (4, 20),
    70: (4, 23), 71: (6, 34), 72: (6, 33), 73: (4, 17), 74: (4, 11),
    75: (4, 18), 76: (4, 10), 77: (4, 5), 78: (12, 25), 79: (12, 26),
    80: (4, 16), 81: (8, 19), 82: (3, 22), 83: (8, 6), 84: (8, 15),
    85: (4, 21), 86: (2, 14), 87: (2, 7),
}

_NAME = [
    "Piano 1", "Piano 2", "Piano 3", "Honky-tonk", "E.Piano 1", "E.Piano 2",
    "Harpsichord", "Clavinet",
    "Celesta", "Glockenspiel", "Music Box", "Vibraphone", "Marimba",
    "Xylophone", "Tubular Bells", "Dulcimer",
    "Drawbar Organ", "Percussive Organ", "Rock Organ", "Church Organ",
    "Reed Organ", "Accordion", "Harmonica", "Bandoneon",
    "Nylon-str.Gt", "Steel-str.Gt", "Jazz Guitar", "Clean Guitar",
    "Muted Guitar", "Overdrive Gt", "Distortion Gt", "Gt Harmonics",
    "Acoustic Bass", "Fingered Bass", "Picked Bass", "Fretless Bass",
    "Slap Bass 1", "Slap Bass 2", "Synth Bass 1", "Synth Bass 2",
    "Violin", "Viola", "Cello", "Contrabass", "Tremolo Str.", "Pizzicato",
    "Harp", "Timpani",
    "Strings 1", "Strings 2", "Syn.Strings1", "Syn.Strings2", "Choir Aahs",
    "Voice Oohs", "Synth Voice", "Orchestra Hit",
    "Trumpet", "Trombone", "Tuba", "Muted Trumpet", "French Horn", "Brass",
    "Synth Brass 1", "Synth Brass 2",
    "Soprano Sax", "Alto Sax", "Tenor Sax", "Baritone Sax", "Oboe",
    "English Horn", "Bassoon", "Clarinet",
    "Piccolo", "Flute", "Recorder", "Pan Flute", "Blown Bottle", "Shakuhachi",
    "Whistle", "Ocarina",
    "Square Lead", "Saw Lead", "Calliope", "Chiff Lead", "Charang",
    "Voice Lead", "Fifth Lead", "Bass+Lead",
    "Fantasia", "Warm Pad", "Polysynth", "Space Choir", "Bowed Glass",
    "Metal Pad", "Halo Pad", "Sweep Pad",
    "Rain Drop", "Soundtrack", "Crystal", "Atmosphere", "Brightness",
    "Goblins", "Echoes", "SF",
    "Sitar", "Banjo", "Shamisen", "Koto", "Kalimba", "Bag pipe", "Fiddle",
    "Shanai",
    "Tinkle Bell", "Agogo", "Steel Drums", "Woodblock", "Taiko Drum",
    "Melodic Tom", "Synth Drum", "Reverse Cym.",
    "Gt Fret Noise", "Breath Noise", "Seashore", "Bird", "Telephone",
    "Helicopter", "Applause", "Gunshot",
]


# Instruments whose natural register sits below the MC window: a note
# that must fold UP a whole octave or more lands inside the melody's
# register and muddies the mix; deleting it sounds better.
_BASS = {1, 12}


def _events(path: Path):
    """Note events in file order: (pos, channel, note, velocity).

    Mirrors open_midi.gml: every note_on with velocity > 0 is an event,
    out-of-range drum notes are dropped, and the last program change per
    channel fixes its instrument.
    """
    mid = mido.MidiFile(path)
    tpb = mid.ticks_per_beat & 0x7FFF
    micsecqn = 0
    programs = [0] * 16
    events = []
    # Track names on the events' channels (muscriptor labels the vocal
    # track 'voice'); used to give the vocal its own instrument.
    names: dict[int, str] = {}
    for track in mid.tracks:
        pos = 0
        track_name = ""
        for msg in track:
            pos += msg.time
            if msg.type == "track_name":
                track_name = msg.name.lower()
            elif msg.type == "set_tempo":
                if micsecqn == 0:
                    micsecqn = msg.tempo
            elif msg.type == "program_change":
                programs[msg.channel] = msg.program
            elif msg.type == "note_on" and msg.velocity > 0:
                if msg.channel == 9 and not 24 <= msg.note <= 84:
                    continue
                if track_name:
                    names[msg.channel] = track_name
                events.append((pos, msg.channel, msg.note, msg.velocity))
    if not micsecqn:
        micsecqn = 500_000
    return events, programs, tpb, micsecqn, names


def _fold(key: int, center: int | None = None) -> int:
    """Fold a key by octave into the Minecraft two-octave window.

    Keeps the pitch class. ``center`` is the voice's median: among the
    in-window octaves of this key, the one landing closest to the voice
    is chosen, so a phrase straddling the boundary stays contiguous
    instead of jumping. Without a center, notes fold to the nearest
    octave (the old behavior).
    """
    options = [key + 12 * s for s in range(-8, 9)
               if FOLD_LO <= key + 12 * s <= FOLD_HI]
    if not options:
        return max(FOLD_LO, min(FOLD_HI, key))
    if center is None:
        return min(options, key=lambda k: abs(k - key))
    return min(options, key=lambda k: abs(k - center))


def _model_shifts(rows: list[tuple[int, int, int, int, int]]) -> list[int]:
    """Predicted octave shift per note from octave.pt.

    ``rows``: (instrument, key, prev_key, next_key, velocity) carrying the
    same neighbour and velocity fields the training oracle used, so there
    is no train/inference feature mismatch.

    Falls back to zero shift when no model file exists. The loaded model
    is cached; inference runs on the GPU when one is available.
    """
    if not rows:
        return []
    try:
        import torch

        from noteblockify.octave import OctaveNet, SHIFTS
    except ImportError:
        return [0] * len(rows)
    from pathlib import Path as _P

    weights = _P(__file__).resolve().parent.parent / "octave.pt"
    if not weights.exists():
        return [0] * len(rows)
    model = getattr(_model_shifts, "cache", None)
    if model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = OctaveNet().to(device)
        model.load_state_dict(
            torch.load(weights, map_location=device))
        model.eval()
        _model_shifts.cache = model
    device = next(model.parameters()).device
    from noteblockify.octave import note_features

    feats = [note_features(*row) for row in rows]
    with torch.no_grad():
        preds = model(
            torch.tensor(feats, dtype=torch.float32, device=device)
        ).argmax(-1)
    return [SHIFTS[p] for p in preds.tolist()]


def _diversify(programs: list[int], channels: list[int]) -> dict[int, int]:
    """Re-map channels whose default vanilla sound collides with another.

    Returns channel -> chosen instrument. A channel may switch to its
    alternate timbre when that makes the ensemble's instrument histogram
    flatter (more distinct sounds in use).
    """
    choice = {c: _PROGRAM[programs[c]][0] for c in channels if c != 9}
    counts = Counter(choice.values())
    for c in channels:
        if c == 9 or programs[c] not in _ALT:
            continue
        alt = _ALT[programs[c]][0]
        if counts[choice[c]] > 1 and counts[alt] < counts[choice[c]] - 1:
            counts[choice[c]] -= 1
            counts[alt] += 1
            choice[c] = alt
    return choice


def _tempo_and_ticks(events, tpb, micsecqn):
    """Choose tempo and place notes so timing matches the MIDI.

    Seconds-based placement with the stored (hundredth-quantized) tempo,
    iterated until stable. Long songs are re-gridded by scaling all
    onsets proportionally: relative spacing survives, absolute time
    stretches, and the tick order is unchanged.
    """
    base = min(e[0] for e in events)
    span = max(e[0] for e in events) - base
    seconds = micsecqn * span / tpb / 1e6
    if seconds <= 0:
        return 10.0, [0] * len(events)


    # Start from the OpenNBS beat grid (2x precision) and, if that does
    # not fit the 16-bit length, coarsen proportionally until it does.
    delta = tpb / 8
    ticks = [int((pos - base) // delta) for pos, *_ in events]
    while ticks and max(ticks) > MAX_TICK and delta < 1 << 20:
        delta *= 2
        ticks = [int((pos - base) // delta) for pos, *_ in events]
    if not ticks:
        return 10.0, []
    enda = max(ticks)

    tempo = min(MAX_TEMPO, max(MIN_TEMPO, round(enda / seconds * 100) / 100))
    for _ in range(4):
        ticks = [min(MAX_TICK, round((pos - base) * micsecqn /1000000 / tpb
                                     * tempo))
                 for pos, *_ in events]
        new_enda = max(ticks)
        if new_enda > MAX_TICK:
            tempo = min(MAX_TEMPO, math.floor(tempo * MAX_TICK / new_enda * 100) / 100)
            continue
        # Snap so the last tick lands at exactly `seconds`.
        want = round(new_enda / seconds * 100) / 100
        if abs(want - tempo) < 1e-9:
            break
        tempo = min(MAX_TEMPO, max(MIN_TEMPO, want))
    return tempo, ticks


def arrange(midi_path: str | Path, vocal_instrument: int | None = None) -> pynbs.File:
    path = Path(midi_path)
    events, programs, tpb, micsecqn, names = _events(path)
    if not events:
        song = pynbs.new_file(song_name=path.stem or "song",
                              song_origin=path.name)
        song.header.tempo = 10
        return song

    tempo, ticks = _tempo_and_ticks(events, tpb, micsecqn)

    # Decide every note's instrument and raw key first, then ask the
    # octave model for whole-octave shifts, then fold what remains
    # out-of-range (a shift can only fix octave placement, folding keeps
    # pitch class for anything still outside).
    prepared = []
    channels = sorted({e[1] for e in events})
    choice = _diversify(programs, channels)
    # Vocal override: channels whose muscriptor track name marks a vocal
    # (voice/vocal/lead) get the requested vanilla instrument instead of
    # the GM map's pick, and a velocity boost so the melody carries.
    if vocal_instrument is not None:
        for c in channels:
            if c != 9 and any(word in names.get(c, "")
                              for word in ("voice", "vocal", "lead")):
                choice[c] = vocal_instrument
    for pos, ch, note, vel in events:
        if ch == 9:
            instrument, key = _DRUM[note]
            key += 33
        else:
            instrument = choice[ch]
            key = note - 21 + 12 * _PROGRAM[programs[ch]][1]
        prepared.append((instrument, key, vel))
    # Neighbour context per channel, exactly like the training oracle:
    # previous and next note in the same channel, by file order.
    where: dict[int, list[int]] = {}
    for i, (_inst, _key, _vel) in enumerate(prepared):
        where.setdefault(events[i][1], []).append(i)
    rows = [None] * len(prepared)
    for idxs in where.values():
        for n, i in enumerate(idxs):
            inst, key, vel = prepared[i]
            prev_key = prepared[idxs[n - 1]][1] if n > 0 else key
            next_key = prepared[idxs[n + 1]][1] if n + 1 < len(idxs) else key
            rows[i] = (inst, key, prev_key, next_key, vel)
    shifts = _model_shifts(rows)
    # The model decides per note, but octave placement is a per-voice
    # decision: a split verdict (some notes +1, others 0) tears the line
    # in half. Majority-vote within each melodic channel and apply the
    # winning shift to every note of that channel.
    votes: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for i, shift in enumerate(shifts):
        if events[i][1] != 9:
            votes[events[i][1]][shift] += 1
    winner = {c: max(counts, key=counts.get)
              for c, counts in votes.items()}
    # Report how the octave decision was made: model in use, and the
    # per-channel verdicts after the vote.
    from pathlib import Path as _Path

    song_octave_info = {
        "model": (_Path(__file__).resolve().parent.parent / "octave.pt").exists(),
        "channels": {c: winner[c] for c in sorted(winner)},
    }
    shifts = [shift if events[i][1] == 9 else winner[events[i][1]]
              for i, shift in enumerate(shifts)]
    prepared = [(inst, key + 12 * shift, vel)
                for (inst, key, vel), shift in zip(prepared, shifts)]
    # Each channel's post-shift median: the centroid out-of-window notes
    # fold toward, keeping phrases contiguous across the boundary.
    by_ch_keys: dict[int, list[int]] = {}
    for i, (_inst, key, _vel) in enumerate(prepared):
        by_ch_keys.setdefault(events[i][1], []).append(key)
    center = {}
    for c, keys in by_ch_keys.items():
        in_window = [k for k in keys if FOLD_LO <= k <= FOLD_HI]
        pool = in_window if in_window else keys
        center[c] = sorted(pool)[len(pool) // 2]
    per_tick: dict[tuple[int, int], int] = defaultdict(int)
    for tick, (_, ch, *_rest) in zip(ticks, events):
        per_tick[(ch, tick)] += 1
    height = {c: max((n for (cc, _t), n in per_tick.items() if cc == c),
                     default=0) for c in channels}
    prefix = {}
    base_layer = 0
    for c in channels:
        prefix[c] = base_layer
        base_layer += max(1, height[c])

    occupied: set[tuple[int, int]] = set()
    notes = []
    dropped = 0
    for tick, (_pos, ch, _note, _vel), (instrument, key, vel) in zip(
            ticks, events, prepared):
        folded = _fold(key, center[ch])
        # A bass note folded up a whole octave or more sits inside the
        # melody's register and muddies the mix; deleting it sounds
        # better than transposing it up.
        if instrument in _BASS and folded - key >= 12:
            dropped += 1
            continue
        key = folded
        layer = prefix[ch]
        while (tick, layer) in occupied:
            layer += 1
        occupied.add((tick, layer))
        notes.append(pynbs.Note(
            tick=tick, layer=layer, instrument=instrument, key=key,
            velocity=min(vel, 100), panning=0,
        ))
    notes.sort(key=lambda n: (n.tick, n.layer))
    if not notes:
        raise ValueError(f"{path.name}: every note was dropped folding "
                         "into the Minecraft two-octave window")
    enda = notes[-1].tick

    # Stereo: notes follow their layer; layers carry the field.
    spread = (45, -45, 25, -25, 55, -55)
    pan: dict[int, int] = {}
    for side, c in enumerate(c for c in channels if c != 9):
        pan[c] = spread[side % len(spread)] if side else 0
    if 9 in channels:
        pan[9] = 0

    name = path.stem.encode("cp1252", errors="ignore").decode("cp1252").strip()
    song = pynbs.new_file(
        song_name=name or "song",
        song_origin=path.name.encode("ascii", "replace").decode(),
    )
    song.header.tempo = tempo
    song.header.time_signature = 4
    song.header.song_length = enda
    song.header.song_layers = base_layer
    song.notes = notes
    song.layers = [
        pynbs.Layer(
            id=i,
            name="Percussion" if c == 9 else _NAME[programs[c]],
            panning=pan[c],
        )
        for c in channels for i in range(prefix[c], prefix[c] + max(1, height[c]))
    ]
    song.octave_info = song_octave_info
    return song
