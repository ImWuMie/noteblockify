"""MC-constrained placement and candidate generation.

A pre NBS is the reference performance. The production decoder changes keys
only by whole octaves. Separate bounded edit candidates are available for
human A/B listening without changing the default production invariant.

The fixed candidate policies are deliberately exposed for preference data:
users can listen to several legal arrangements, label A/B choices, and train
``noteblockify.preference_model`` to rank those arrangements. Until a trained
ranker exists, ``balanced`` is the deterministic fallback.
"""

from __future__ import annotations

import copy
import math
import re
from collections import Counter, defaultdict

import pynbs

WINDOW_LO = 33
WINDOW_HI = 57
PITCH_LO = -1200
PITCH_HI = 1200
_MAX_SHIFT = 16
_CHANNEL_RE = re.compile(r"^ch(\d+):")
_POLICIES = ("balanced", "nearest", "low", "high", "spread")


def _candidates(raw_key: int) -> list[tuple[int, int]]:
    """Return legal (key, octave-shift) choices for one raw key."""
    if WINDOW_LO <= raw_key <= WINDOW_HI:
        return [(raw_key, 0)]
    candidates = [
        (raw_key + 12 * shift, shift)
        for shift in range(-_MAX_SHIFT, _MAX_SHIFT + 1)
        if WINDOW_LO <= raw_key + 12 * shift <= WINDOW_HI
    ]
    if not candidates:
        raise ValueError(f"raw key {raw_key} has no MC-window candidate")
    return candidates


def _all_candidates(raw_key: int) -> list[tuple[int, int]]:
    """Return all legal same-pitch-class keys, including in-window shifts."""
    candidates = [
        (raw_key + 12 * shift, shift)
        for shift in range(-_MAX_SHIFT, _MAX_SHIFT + 1)
        if WINDOW_LO <= raw_key + 12 * shift <= WINDOW_HI
    ]
    if not candidates:
        raise ValueError(f"raw key {raw_key} has no MC-window candidate")
    return candidates


def _voice_id(song: pynbs.File, note: pynbs.Note) -> str:
    if 0 <= note.layer < len(song.layers):
        match = _CHANNEL_RE.match(song.layers[note.layer].name)
        if match:
            return f"ch{match.group(1)}"
    return f"layer{note.layer}"


def _groups(song: pynbs.File) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, note in enumerate(song.notes):
        groups[_voice_id(song, note)].append(index)
    for indices in groups.values():
        indices.sort(key=lambda i: (song.notes[i].tick,
                                    song.notes[i].layer, i))
    return dict(groups)


def _voice_register_targets(song: pynbs.File) -> dict[str, tuple[float, float]]:
    """Assign lower MC targets to lower raw voices.

    A per-voice scorer alone cannot know that a very low piano/strings part
    should stay below the other parts after every voice is folded into 33..57.
    Rank voices by their raw median and add a soft register target. The
    lowest voice targets F#3; higher voices spread upward. This is especially
    important for dense multi-track MIDI where no track is explicitly named
    bass.
    """
    groups = _groups(song)
    medians = []
    for voice, indices in groups.items():
        if voice == "ch9":
            continue
        keys = sorted(song.notes[index].key for index in indices)
        if keys:
            medians.append((keys[len(keys) // 2], voice))
    medians.sort()
    if not medians:
        return {}
    last = max(1, len(medians) - 1)
    targets = {}
    for rank, (median, voice) in enumerate(medians):
        percentile = rank / last
        target = WINDOW_LO + 16.0 * percentile
        # Strongest below the register center; softer for upper voices so
        # melodic material can still follow the learned candidate score.
        weight = 1.40 - 0.75 * percentile
        targets[voice] = (target, weight)
    return targets


def _unary(raw_key: int, key: int, policy: str) -> float:
    """Score an individual legal octave under one candidate policy."""
    if WINDOW_LO <= raw_key <= WINDOW_HI:
        return 0.0 if key == raw_key else math.inf
    distance = abs(key - raw_key) / 12.0
    if policy == "low":
        return 0.05 * distance + (key - WINDOW_LO) / 12.0
    if policy == "high":
        return 0.05 * distance + (WINDOW_HI - key) / 12.0
    return distance


def _contour(raw_previous: int, raw_current: int,
             key_previous: int, key_current: int, policy: str) -> float:
    raw_delta = raw_current - raw_previous
    key_delta = key_current - key_previous
    error = abs(key_delta - raw_delta) / 12.0
    reversal = 0.8 if raw_delta and key_delta and (
        raw_delta > 0) != (key_delta > 0) else 0.0
    excessive_leap = max(0, abs(key_delta) - abs(raw_delta) - 12) / 12.0
    if policy == "nearest":
        return 0.0
    if policy in ("low", "high"):
        return 0.05 * error + 0.10 * reversal
    return 0.42 * error + reversal + 0.10 * excessive_leap


def _solve_voice(song: pynbs.File, indices: list[int],
                 occupied: dict[int, Counter[int]], policy: str) -> dict[int, int]:
    options = [_candidates(song.notes[index].key) for index in indices]
    costs: list[list[float]] = []
    back: list[list[int | None]] = []

    for note, choices in zip((song.notes[i] for i in indices), options):
        collision_weight = 1.20 if policy == "spread" else 0.55
        costs.append([
            _unary(note.key, key, policy) + collision_weight * occupied[note.tick][key]
            for key, _shift in choices
        ])
        back.append([None] * len(choices))

    for position in range(1, len(indices)):
        previous = song.notes[indices[position - 1]]
        current = song.notes[indices[position]]
        for current_choice, (current_key, _shift) in enumerate(
                options[position]):
            best_cost = math.inf
            best_previous = None
            for previous_choice, (previous_key, _previous_shift) in enumerate(
                    options[position - 1]):
                candidate = costs[position - 1][previous_choice] + _contour(
                    previous.key, current.key, previous_key, current_key, policy)
                if candidate < best_cost:
                    best_cost = candidate
                    best_previous = previous_choice
            costs[position][current_choice] += best_cost
            back[position][current_choice] = best_previous

    choice = min(range(len(costs[-1])), key=costs[-1].__getitem__)
    result: dict[int, int] = {}
    for position in range(len(indices) - 1, -1, -1):
        result[indices[position]] = options[position][choice][0]
        previous_choice = back[position][choice]
        if previous_choice is None:
            break
        choice = previous_choice
    return result


def _solve_voice_ranked(song: pynbs.File, indices: list[int],
                        occupied: dict[int, Counter[int]], ranker) -> dict[int, int]:
    """Viterbi decode one voice with learned scores and continuity."""
    from noteblockify.preference_model import _contexts, score_options

    options = [_all_candidates(song.notes[index].key) for index in indices]
    contexts = _contexts(song)
    costs: list[list[float]] = []
    back: list[list[int | None]] = []
    for index, choices in zip(indices, options):
        keys = [key for key, _shift in choices]
        learned = score_options(ranker, song, index, keys, contexts)
        # The legacy ranker was trained while in-window notes had one
        # immutable option. Do not let its out-of-distribution score choose a
        # new octave for those notes; phrase continuity owns that decision.
        if WINDOW_LO <= song.notes[index].key <= WINDOW_HI:
            learned = [0.0] * len(keys)
        collision_weight = 0.30
        costs.append([
            (0.0 if WINDOW_LO <= song.notes[index].key <= WINDOW_HI else -score)
            + 0.12 * abs(key - song.notes[index].key) / 12.0
            + collision_weight * occupied[song.notes[index].tick][key]
            for score, key in zip(learned, keys)
        ])
        back.append([None] * len(keys))

    for position in range(1, len(indices)):
        previous = song.notes[indices[position - 1]]
        current = song.notes[indices[position]]
        for current_choice, (current_key, current_shift) in enumerate(
                options[position]):
            best_cost = math.inf
            best_previous = None
            for previous_choice, (previous_key, _previous_shift) in enumerate(
                    options[position - 1]):
                candidate = (costs[position - 1][previous_choice]
                             + 0.45 * _contour(
                                 previous.key, current.key,
                                 previous_key, current_key, "balanced")
                             + 2.0 * (current_shift != _previous_shift))
                if candidate < best_cost:
                    best_cost = candidate
                    best_previous = previous_choice
            costs[position][current_choice] += best_cost
            back[position][current_choice] = best_previous

    choice = min(range(len(costs[-1])), key=costs[-1].__getitem__)
    result: dict[int, int] = {}
    for position in range(len(indices) - 1, -1, -1):
        result[indices[position]] = options[position][choice][0]
        previous_choice = back[position][choice]
        if previous_choice is None:
            break
        choice = previous_choice
    return result


def _solve_voice_smooth(song: pynbs.File, indices: list[int],
                        occupied: dict[int, Counter[int]]) -> dict[int, int]:
    """Choose legal keys while penalizing octave changes inside a phrase."""
    options = [_all_candidates(song.notes[index].key) for index in indices]
    costs: list[list[float]] = []
    back: list[list[int | None]] = []
    for note, choices in zip((song.notes[i] for i in indices), options):
        costs.append([
            0.12 * abs(key - note.key) / 12.0
            + 0.30 * occupied[note.tick][key]
            for key, _shift in choices
        ])
        back.append([None] * len(choices))
    for position in range(1, len(indices)):
        previous = song.notes[indices[position - 1]]
        current = song.notes[indices[position]]
        for current_choice, (current_key, current_shift) in enumerate(
                options[position]):
            best_cost = math.inf
            best_previous = None
            for previous_choice, (previous_key, previous_shift) in enumerate(
                    options[position - 1]):
                candidate = (costs[position - 1][previous_choice]
                             + 0.45 * _contour(
                                 previous.key, current.key,
                                 previous_key, current_key, "balanced")
                             + 2.0 * (current_shift != previous_shift))
                if candidate < best_cost:
                    best_cost = candidate
                    best_previous = previous_choice
            costs[position][current_choice] += best_cost
            back[position][current_choice] = best_previous
    choice = min(range(len(costs[-1])), key=costs[-1].__getitem__)
    result: dict[int, int] = {}
    for position in range(len(indices) - 1, -1, -1):
        result[indices[position]] = options[position][choice][0]
        previous_choice = back[position][choice]
        if previous_choice is None:
            break
        choice = previous_choice
    return result


def _assign_smooth(song: pynbs.File) -> dict[int, int]:
    groups = _groups(song)
    assignment: dict[int, int] = {}
    for _pass in range(4):
        for voice in sorted(groups):
            occupied: dict[int, Counter[int]] = defaultdict(Counter)
            for index, key in assignment.items():
                if _voice_id(song, song.notes[index]) != voice:
                    occupied[song.notes[index].tick][key] += 1
            assignment.update(_solve_voice_smooth(
                song, groups[voice], occupied))
    return assignment


def _assign_ranked(song: pynbs.File, ranker) -> dict[int, int | None]:
    """Coordinate-descent DP: learned note choices plus cross-voice spread."""
    groups = _groups(song)
    assignment: dict[int, int | None] = {}
    for _pass in range(4):
        for voice in sorted(groups):
            occupied: dict[int, Counter[int]] = defaultdict(Counter)
            for index, key in assignment.items():
                if (_voice_id(song, song.notes[index]) != voice and
                        key is not None):
                    occupied[song.notes[index].tick][key] += 1
            assignment.update(_solve_voice_ranked(
                song, groups[voice], occupied, ranker))
    return assignment


def _assign(song: pynbs.File, policy: str = "balanced") -> dict[int, int]:
    if policy not in _POLICIES:
        raise ValueError(f"unknown placement policy: {policy}")
    groups = _groups(song)
    assignment: dict[int, int] = {}
    for _pass in range(4):
        for voice in sorted(groups):
            occupied: dict[int, Counter[int]] = defaultdict(Counter)
            for index, key in assignment.items():
                if _voice_id(song, song.notes[index]) != voice:
                    occupied[song.notes[index].tick][key] += 1
            assignment.update(_solve_voice(
                song, groups[voice], occupied, policy))
    return assignment


def _copy_with_assignment(song: pynbs.File,
                          assignment: dict[int, int | None]) -> pynbs.File:
    result = copy.deepcopy(song)
    result.notes = [
        pynbs.Note(
            tick=note.tick,
            layer=note.layer,
            instrument=note.instrument,
            key=assignment[index],
            velocity=note.velocity,
            panning=note.panning,
            pitch=note.pitch,
        )
        for index, note in enumerate(song.notes)
        if assignment[index] is not None
    ]
    result.notes.sort(key=lambda note: (note.tick, note.layer))
    return result


def _pitch_representation(raw_key: int) -> tuple[int, int]:
    """Represent a raw key with an in-window key plus NBS fine pitch.

    OpenNBS supports note pitch in cents. This preserves raw keys such as 31
    as ``33 - 200 cents`` instead of forcing an octave jump to 43. Meteor's
    Notebot decoder currently ignores this field, so this mode targets NBS
    players/OpenNBS rather than vanilla physical-note playback.
    """
    target = max(WINDOW_LO + PITCH_LO // 100,
                 min(WINDOW_HI + PITCH_HI // 100, raw_key))
    key = max(WINDOW_LO, min(WINDOW_HI, target))
    pitch = max(PITCH_LO, min(PITCH_HI, (raw_key - key) * 100))
    return key, pitch


def refine_pitch(song: pynbs.File) -> pynbs.File:
    """Keep absolute pitch with NBS fine tuning where the MC window permits."""
    result = copy.deepcopy(song)
    notes = []
    for note in song.notes:
        key, pitch = _pitch_representation(note.key)
        notes.append(pynbs.Note(
            tick=note.tick,
            layer=note.layer,
            instrument=note.instrument,
            key=key,
            velocity=note.velocity,
            panning=note.panning,
            pitch=pitch,
        ))
    result.notes = notes
    result.notes.sort(key=lambda note: (note.tick, note.layer))
    return result


def _edit_suspects(song: pynbs.File,
                   assignment: dict[int, int]) -> set[int]:
    """Select low source notes whose forced fold causes a local octave jump."""
    suspects = set()
    for indices in _groups(song).values():
        for position, index in enumerate(indices):
            raw = song.notes[index].key
            if raw >= WINDOW_LO or assignment[index] - raw < 12:
                continue
            errors = []
            if position:
                previous = indices[position - 1]
                errors.append(abs(
                    (assignment[index] - assignment[previous]) -
                    (raw - song.notes[previous].key)))
            if position + 1 < len(indices):
                following = indices[position + 1]
                errors.append(abs(
                    (assignment[following] - assignment[index]) -
                    (song.notes[following].key - raw)))
            if errors and max(errors) >= 8:
                suspects.add(index)
    return suspects


def _edit_options(song: pynbs.File, index: int,
                  indices: list[int], assignment: dict[int, int]):
    """Build bounded keep/drop/replace actions for one suspect source note."""
    note = song.notes[index]
    options = [("keep", assignment[index])]
    if note.key < WINDOW_LO or note.key > WINDOW_HI:
        options.append(("drop", None))
    position = indices.index(index)
    neighbors = []
    if position:
        neighbors.append(indices[position - 1])
    if position + 1 < len(indices):
        neighbors.append(indices[position + 1])
    keys = set()
    for neighbor in neighbors:
        keys.update(assignment[neighbor] for _key, _shift in
                    _candidates(song.notes[neighbor].key))
    for key in sorted(keys):
        if key != assignment[index]:
            options.append(("replace", key))
    return options


def edit_candidates(song: pynbs.File) -> dict[str, pynbs.File]:
    """Generate bounded clamp/drop/replace/add candidates for A/B labeling."""
    assignment = _assign(song, policy="balanced")
    suspects = _edit_suspects(song, assignment)
    candidates = {"balanced": _copy_with_assignment(song, assignment)}
    if not suspects:
        return candidates
    dropped = dict(assignment)
    replaced = dict(assignment)
    for voice, indices in _groups(song).items():
        for index in suspects.intersection(indices):
            options = _edit_options(song, index, indices, assignment)
            dropped[index] = next(key for action, key in options
                                  if action == "drop")
            replacements = [key for action, key in options
                            if action == "replace"]
            if replacements:
                replaced[index] = replacements[0]
    candidates["drop"] = _copy_with_assignment(song, dropped)
    candidates["replace"] = _copy_with_assignment(song, replaced)
    clamped = {
        index: (WINDOW_LO if note.key < WINDOW_LO else WINDOW_HI
                if note.key > WINDOW_HI else assignment[index])
        for index, note in enumerate(song.notes)
    }
    candidates["clamp"] = _copy_with_assignment(song, clamped)
    candidates["phrase"] = _copy_with_assignment(
        song, _phrase_stable_assignment(song))
    candidates["smooth"] = _copy_with_assignment(
        song, _assign_smooth(song))
    additions = _interpolation_additions(song, assignment)
    if additions:
        candidates["add"] = _copy_with_additions(song, assignment, additions)
    return candidates


def _interpolation_additions(song: pynbs.File,
                             assignment: dict[int, int]) -> list[tuple[int, int, int]]:
    """Add at most one legal passing note per sparse low voice."""
    additions = []
    for indices in _groups(song).values():
        raw_median = sorted(song.notes[i].key for i in indices)[len(indices) // 2]
        if raw_median >= WINDOW_LO:
            continue
        for previous_index, current_index in zip(indices, indices[1:]):
            previous = song.notes[previous_index]
            current = song.notes[current_index]
            gap = current.tick - previous.tick
            if gap < 8:
                continue
            tick = previous.tick + gap // 2
            raw_key = round((previous.key + current.key) / 2)
            key = min((candidate for candidate, _shift in _candidates(raw_key)),
                      key=lambda candidate: abs(candidate - raw_key))
            additions.append((tick, previous.instrument, key))
            break
    return additions


def _phrase_stable_assignment(song: pynbs.File) -> dict[int, int]:
    """Keep a voice in one octave until a large raw register change.

    This preserves the source phrase's register instead of allowing each
    low note to independently jump between legal octaves.
    """
    assignment = {}
    for indices in _groups(song).values():
        current_shift = None
        for index in indices:
            raw = song.notes[index].key
            choices = _candidates(raw)
            if current_shift is None:
                current_shift = min(choices, key=lambda item: abs(item[0] - 45))[1]
            selected = next((item for item in choices
                             if item[1] == current_shift), None)
            if selected is None:
                selected = min(choices, key=lambda item: abs(item[1] - current_shift))
                current_shift = selected[1]
            assignment[index] = selected[0]
    return assignment


def _copy_with_additions(song: pynbs.File,
                          assignment: dict[int, int],
                          additions: list[tuple[int, int, int]]) -> pynbs.File:
    """Copy a placement and put each addition on a unique NBS layer.

    NBS note encoding stores layer deltas within a tick. Two notes cannot
    share one layer at the same tick; doing so makes the parser accumulate a
    bogus layer index and corrupts the file.
    """
    result = _copy_with_assignment(song, assignment)
    first_layer = len(result.layers)
    for offset in range(len(additions)):
        layer_id = first_layer + offset
        result.layers.append(pynbs.Layer(layer_id, "edit:add"))
    for offset, (tick, instrument, key) in enumerate(additions):
        result.notes.append(pynbs.Note(
            tick=tick, layer=first_layer + offset,
            instrument=instrument, key=key,
            velocity=100, panning=0, pitch=0))
    result.notes.sort(key=lambda note: (note.tick, note.layer))
    return result



def refine_candidates(song: pynbs.File) -> dict[str, pynbs.File]:
    """Generate distinct legal candidates for human A/B comparison."""
    candidates = {}
    seen = set()
    for policy in _POLICIES:
        candidate = _copy_with_assignment(song, _assign(song, policy))
        signature = tuple((note.tick, note.layer, note.key)
                         for note in candidate.notes)
        if signature not in seen:
            candidates[policy] = candidate
            seen.add(signature)
    return candidates


def refine(song: pynbs.File, ranker=None) -> pynbs.File:
    """Return the smooth voice-level placement, optionally ranker-guided.

    The fallback is intentional: no untrained/random network is ever used.
    Every candidate already satisfies the MC hard constraints before ranking.
    """
    candidates = refine_candidates(song)
    if ranker is None:
        try:
            from noteblockify.preference_model import load_ranker

            ranker = load_ranker()
        except ImportError:
            ranker = None
    if ranker is None:
        return _copy_with_assignment(song, _assign_smooth(song))
    return _copy_with_assignment(song, _assign_ranked(song, ranker))


def changed_keys(before: pynbs.File, after: pynbs.File) -> int:
    """Count changed source events, including events omitted by the model."""
    slots: dict[tuple[int, int, int, int, int], list[int]] = defaultdict(list)
    for note in after.notes:
        slots[(note.tick, note.layer, note.instrument,
               note.velocity, note.panning)].append(note.key)
    changed = 0
    for note in before.notes:
        slot = (note.tick, note.layer, note.instrument,
                note.velocity, note.panning)
        keys = slots[slot]
        if not keys or keys.pop(0) != note.key:
            changed += 1
    changed += sum(len(keys) for keys in slots.values())
    return changed
