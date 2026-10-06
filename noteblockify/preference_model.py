"""Learned candidate ranking for MC-constrained note placement.

The model scores legal octave candidates from human A/B preferences
collected on complete candidate songs.
"""

from __future__ import annotations

import copy
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch import nn

WINDOW_LO = 33
WINDOW_HI = 57
FEATURE_COUNT = 12
MODEL_PATH = Path(__file__).resolve().parent.parent / "mc_ranker.pt"
_CHANNEL_RE = re.compile(r"^ch(\d+):")


class PlacementRanker(nn.Module):
    """Configurable scalar scorer for one raw-note/legal-key decision."""

    def __init__(self, width: int = 96, depth: int = 2):
        super().__init__()
        if width < 8 or depth < 1:
            raise ValueError("ranker width must be >= 8 and depth >= 1")
        layers = [nn.Linear(FEATURE_COUNT, width),
                  nn.LayerNorm(width), nn.GELU()]
        for _ in range(depth - 1):
            layers.extend((nn.Linear(width, width),
                           nn.LayerNorm(width), nn.GELU()))
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)
        self.width = width
        self.depth = depth

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def parameter_count(model: nn.Module) -> int:
    """Return trainable parameter count for reporting and model selection."""
    return sum(parameter.numel() for parameter in model.parameters()
               if parameter.requires_grad)


def _voice_id(song, note) -> str:
    if 0 <= note.layer < len(song.layers):
        match = _CHANNEL_RE.match(song.layers[note.layer].name)
        if match:
            return f"ch{match.group(1)}"
    return f"layer{note.layer}"


def _contexts(song):
    """Return per-note context shared by training and inference."""
    groups: dict[str, list[int]] = defaultdict(list)
    for index, note in enumerate(song.notes):
        groups[_voice_id(song, note)].append(index)
    length = max(1, max((note.tick for note in song.notes), default=0))
    layer_count = max(1, len(song.layers) - 1)
    result = {}
    for indices in groups.values():
        indices.sort(key=lambda i: (song.notes[i].tick,
                                    song.notes[i].layer, i))
        density = Counter(song.notes[i].tick for i in indices)
        size = max(1, len(indices) - 1)
        for position, index in enumerate(indices):
            note = song.notes[index]
            previous = song.notes[indices[position - 1]].key if position else note.key
            following = (song.notes[indices[position + 1]].key
                         if position + 1 < len(indices) else note.key)
            result[index] = (
                previous, following, position / size,
                len(indices) / max(1, len(song.notes)),
                min(density[note.tick], 16) / 16.0,
                note.layer / layer_count,
                note.tick / length,
            )
    return result


def feature_vector(song, index: int, candidate_key: int,
                   contexts=None) -> list[float]:
    """Build the fixed feature vector for one candidate key."""
    note = song.notes[index]
    previous, following, position, voice_size, density, layer, progress = (
        (contexts or _contexts(song))[index])
    raw = note.key
    shift = (candidate_key - raw) / 12.0
    return [
        raw / 87.0,
        candidate_key / 87.0,
        shift / 8.0,
        note.instrument / 15.0,
        previous / 87.0,
        following / 87.0,
        position,
        voice_size,
        density,
        layer,
        progress,
        float(WINDOW_LO <= raw <= WINDOW_HI),
    ]


def _ordered_notes(song):
    return sorted(range(len(song.notes)),
                  key=lambda i: (song.notes[i].tick, song.notes[i].layer, i))


def assignment_features(pre, candidate) -> list[list[float]]:
    """Features for a candidate NBS, aligned to the pre note order."""
    pre_order = _ordered_notes(pre)
    candidate_order = _ordered_notes(candidate)
    if len(pre_order) != len(candidate_order):
        raise ValueError("candidate note count differs from pre")
    contexts = _contexts(pre)
    rows = []
    for pre_index, candidate_index in zip(pre_order, candidate_order):
        before = pre.notes[pre_index]
        after = candidate.notes[candidate_index]
        immutable = (before.tick, before.layer, before.instrument,
                     before.velocity, before.panning)
        actual = (after.tick, after.layer, after.instrument,
                  after.velocity, after.panning)
        if immutable != actual:
            raise ValueError("candidate changes a non-key note field")
        if (after.key - before.key) % 12:
            raise ValueError("candidate does not preserve pitch class")
        rows.append(feature_vector(pre, pre_index, after.key, contexts))
    return rows


def candidate_score(model: PlacementRanker | None, pre, candidate) -> float:
    """Mean note score used by pairwise preference training."""
    rows = assignment_features(pre, candidate)
    if not rows or model is None:
        return 0.0
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        values = model(torch.tensor(rows, dtype=torch.float32, device=device))
    return float(values.mean())


def score_options(model: PlacementRanker, song, index: int,
                  candidate_keys: list[int | None], contexts=None) -> list[float]:
    """Score every legal key option for one note in one model call.

    This is the same ``feature_vector`` path used by pairwise training. The
    decoder may therefore choose a different legal key at every note instead
    of selecting one pre-built whole-song policy.
    """
    if not candidate_keys:
        return []
    context = contexts or _contexts(song)
    rows = [feature_vector(song, index, key, context)
            for key in candidate_keys]
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        values = model(torch.tensor(rows, dtype=torch.float32, device=device))
    return [float(value) for value in values.tolist()]


def load_ranker(path: Path | None = None) -> PlacementRanker | None:
    """Load trained preferences on CUDA when available."""
    path = Path(path or MODEL_PATH)
    if not path.exists():
        return None
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state = torch.load(path, map_location=device, weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            width = int(state.get("width", 96))
            depth = int(state.get("depth", 2))
            state = state["state_dict"]
        else:
            width, depth = 96, 2
        model = PlacementRanker(width=width, depth=depth).to(device)
        model.load_state_dict(state)
        model.eval()
        return model
    except (OSError, RuntimeError, TypeError, ValueError):
        return None


def _checkpoint_state(path: Path, device: torch.device):
    """Read both legacy raw state dicts and metadata checkpoints."""
    state = torch.load(path, map_location=device, weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        return state["state_dict"]
    return state


def clone_with_keys(song, keys: dict[int, int]):
    """Create a candidate copy while changing keys only."""
    result = copy.deepcopy(song)
    result.notes = [
        type(note)(
            tick=note.tick,
            layer=note.layer,
            instrument=note.instrument,
            key=keys[index],
            velocity=note.velocity,
            panning=note.panning,
            pitch=note.pitch,
        )
        for index, note in enumerate(song.notes)
    ]
    result.notes.sort(key=lambda note: (note.tick, note.layer))
    return result


def append_preference(path: Path, pre: Path, preferred: Path,
                      rejected: Path, note: str = "") -> None:
    """Append one human preference as JSONL.

    Paths are stored exactly as supplied so a feedback file can be moved
    together with its candidate directory. Each row is one A/B judgment.
    """
    for candidate in (pre, preferred, rejected):
        if not candidate.exists():
            raise FileNotFoundError(candidate)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "pre": str(pre),
        "preferred": str(preferred),
        "rejected": str(rejected),
    }
    if note:
        row["note"] = note
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _records(path: Path):
    from pynbs import read

    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                yield (read(row["pre"]), read(row["preferred"]),
                       read(row["rejected"]))
            except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid preference row {line_number} in "
                                 f"{path}: {exc}") from exc


def validate_preferences(path: Path) -> dict:
    """Validate every JSONL row and report unique songs and candidate files."""
    records = list(_records(path))
    if not records:
        raise ValueError(f"no preference rows in {path}")
    songs = {str(pre.header.song_origin or pre.header.song_name)
             for pre, _preferred, _rejected in records}
    for pre, preferred, rejected in records:
        assignment_features(pre, preferred)
        assignment_features(pre, rejected)
    return {"rows": len(records), "songs": len(songs)}


def _fit_examples(examples, output: Path, steps: int,
                   learning_rate: float, init: Path | None = None,
                   width: int = 96, depth: int = 2,
                   validation_examples=None, patience: int = 200) -> dict:
    """Fit pairwise examples and return measured training metrics."""
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PlacementRanker(width=width, depth=depth).to(device)
    if init is not None:
        state = _checkpoint_state(init, device)
        model.load_state_dict(state)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    preferred_parts = [preferred for preferred, _rejected in examples]
    rejected_parts = [rejected for _preferred, rejected in examples]
    preferred_lengths = torch.tensor(
        [len(part) for part in preferred_parts], dtype=torch.long,
        device=device)
    rejected_lengths = torch.tensor(
        [len(part) for part in rejected_parts], dtype=torch.long,
        device=device)
    preferred_ids = torch.repeat_interleave(
        torch.arange(len(examples), device=device), preferred_lengths)
    rejected_ids = torch.repeat_interleave(
        torch.arange(len(examples), device=device), rejected_lengths)
    preferred_features = torch.cat(preferred_parts).to(device)
    rejected_features = torch.cat(rejected_parts).to(device)
    model.train()
    best_state = None
    best_metric = float("-inf")
    stale = 0
    losses = []
    for step in range(steps):
        preferred_values = model(preferred_features)
        rejected_values = model(rejected_features)
        preferred_sums = torch.zeros(len(examples), device=device)
        rejected_sums = torch.zeros(len(examples), device=device)
        preferred_sums.scatter_add_(0, preferred_ids, preferred_values)
        rejected_sums.scatter_add_(0, rejected_ids, rejected_values)
        preferred_means = preferred_sums / preferred_lengths
        rejected_means = rejected_sums / rejected_lengths
        loss = torch.nn.functional.softplus(
            -(preferred_means - rejected_means)).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        if validation_examples:
            model.eval()
            with torch.no_grad():
                validation_correct = 0
                for preferred, rejected in validation_examples:
                    preferred_score = model(preferred.to(device)).mean()
                    rejected_score = model(rejected.to(device)).mean()
                    validation_correct += int(preferred_score > rejected_score)
            validation_metric = validation_correct / len(validation_examples)
            model.train()
            if validation_metric > best_metric:
                best_metric = validation_metric
                best_state = {key: value.detach().cpu().clone()
                              for key, value in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    output.parent.mkdir(parents=True, exist_ok=True)
    cpu_state = {key: value.detach().cpu()
                 for key, value in model.state_dict().items()}
    torch.save({"state_dict": cpu_state, "width": width, "depth": depth},
               output)
    model.eval()
    with torch.no_grad():
        correct = 0
        preferred_values = model(preferred_features)
        rejected_values = model(rejected_features)
        preferred_sums = torch.zeros(len(examples), device=device)
        rejected_sums = torch.zeros(len(examples), device=device)
        preferred_sums.scatter_add_(0, preferred_ids, preferred_values)
        rejected_sums.scatter_add_(0, rejected_ids, rejected_values)
        preferred_means = preferred_sums / preferred_lengths
        rejected_means = rejected_sums / rejected_lengths
        correct = int((preferred_means > rejected_means).sum())
    return {
        "examples": len(examples),
        "steps": steps,
        "final_loss": losses[-1] if losses else None,
        "pair_accuracy": correct / len(examples) if examples else 0.0,
        "device": str(device),
        "width": width,
        "depth": depth,
        "parameters": parameter_count(model),
        "best_validation_accuracy": (best_metric
                                      if validation_examples else None),
    }


def _write_metrics(output: Path, metrics: dict, kind: str) -> None:
    metrics_path = output.with_suffix(".json")
    payload = {"kind": kind, **metrics}
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def train_preferences(data: Path, output: Path = MODEL_PATH,
                      steps: int = 2000, learning_rate: float = 2e-3,
                      init: Path | None = None) -> dict:
    """Fit the ranker from pairwise human A/B choices."""
    records = list(_records(data))
    if not records:
        raise ValueError(f"no preference rows in {data}")
    examples = [(
        torch.tensor(assignment_features(pre, preferred),
                     dtype=torch.float32),
        torch.tensor(assignment_features(pre, rejected),
                     dtype=torch.float32),
    ) for pre, preferred, rejected in records]
    metrics = _fit_examples(examples, output, steps, learning_rate, init)
    _write_metrics(output, metrics, "human_pairwise")
    print(f"wrote {output} from {len(records)} human preferences; "
          f"pair accuracy {metrics['pair_accuracy']:.3f}")
    return metrics


def _proxy_loss(pre, candidate) -> float:
    """Transparent bootstrap objective; never presented as human truth."""
    from noteblockify.mc_model import _voice_id

    ordered = sorted(range(len(pre.notes)),
                     key=lambda i: (pre.notes[i].tick, pre.notes[i].layer, i))
    groups: dict[str, list[int]] = defaultdict(list)
    for index in ordered:
        groups[_voice_id(pre, pre.notes[index])].append(index)
    movement = sum(abs(candidate.notes[i].key - pre.notes[i].key) / 12
                   for i in ordered)
    contour = 0.0
    for indices in groups.values():
        for previous, current in zip(indices, indices[1:]):
            raw_delta = pre.notes[current].key - pre.notes[previous].key
            new_delta = candidate.notes[current].key - candidate.notes[previous].key
            contour += abs(new_delta - raw_delta) / 12
    occupied = Counter((note.tick, note.key) for note in candidate.notes)
    collisions = sum(max(0, count - 1) for count in occupied.values())
    return movement + 0.42 * contour + 0.55 * collisions


def _automatic_candidate_loss(pre, candidate) -> float:
    """MIDI-guided deterministic reward for self-training.

    The raw mapped pre key is the MIDI proxy. Distance is primary; contour
    disruption and same-tick pitch collisions are secondary. This objective
    cannot invent musical truth, but it is a reproducible self-training
    signal that does not reward arbitrary octave jumps.
    """
    ordered = _ordered_notes(pre)
    candidate_by_slot = {
        (note.tick, note.layer): note for note in candidate.notes
    }
    movement = 0.0
    for index in ordered:
        raw = pre.notes[index]
        moved = candidate_by_slot[(raw.tick, raw.layer)]
        movement += abs(moved.key - raw.key) / 12.0
    return _proxy_loss(pre, candidate) + movement


def train_midi_guided(midi_dir: Path, output: Path = MODEL_PATH,
                      steps: int = 800, learning_rate: float = 2e-3,
                      width: int = 96, depth: int = 2) -> dict:
    """Train ranking pairs ordered by the automatic MIDI proxy reward."""
    from noteblockify.mc_model import refine_candidates
    from noteblockify.song import arrange_pre

    examples = []
    songs = sorted(midi_dir.glob("*.mid"))
    for midi in songs:
        pre = arrange_pre(midi)
        candidates = refine_candidates(pre)
        ranked = sorted(candidates.values(),
                        key=lambda c: _automatic_candidate_loss(pre, c))
        if len(ranked) < 2:
            continue
        preferred = ranked[0]
        for rejected in ranked[1:]:
            examples.append((
                torch.tensor(assignment_features(pre, preferred),
                             dtype=torch.float32),
                torch.tensor(assignment_features(pre, rejected),
                             dtype=torch.float32),
            ))
    if not examples:
        raise ValueError(f"no candidate pairs found in {midi_dir}")
    split = max(1, len(examples) // 5)
    validation = examples[-split:]
    training = examples[:-split] or examples
    metrics = _fit_examples(training, output, steps, learning_rate,
                             width=width, depth=depth,
                             validation_examples=validation)
    metrics["training_pairs"] = len(training)
    metrics["validation_pairs"] = len(validation)
    _write_metrics(output, metrics, "midi_guided_proxy")
    print(f"wrote MIDI-guided ranker {output} from {len(songs)} songs and "
          f"{len(examples)} pairs; accuracy "
          f"{metrics['pair_accuracy']:.3f}")
    return metrics


def train_bootstrap(midi_dir: Path, output: Path = MODEL_PATH,
                    steps: int = 800, learning_rate: float = 2e-3) -> dict:
    """Pretrain on the explicit solver objective before human feedback.

    This creates a real neural checkpoint and a usable ranking pipeline, but
    its labels are only a transparent engineering prior. Human A/B training
    should be run later with ``--init`` to turn this into a listening model.
    """
    from noteblockify.mc_model import refine_candidates
    from noteblockify.song import arrange_pre

    examples = []
    songs = sorted(midi_dir.glob("*.mid"))
    for midi in songs:
        pre = arrange_pre(midi)
        candidates = refine_candidates(pre)
        if len(candidates) < 2:
            continue
        ranked = sorted(candidates.values(), key=lambda c: _proxy_loss(pre, c))
        preferred = ranked[0]
        for rejected in ranked[1:]:
            examples.append((
                torch.tensor(assignment_features(pre, preferred),
                             dtype=torch.float32),
                torch.tensor(assignment_features(pre, rejected),
                             dtype=torch.float32),
            ))
    if not examples:
        raise ValueError(f"no distinct candidate pairs found in {midi_dir}")
    metrics = _fit_examples(examples, output, steps, learning_rate)
    _write_metrics(output, metrics, "bootstrap_proxy")
    print(f"wrote bootstrap {output} from {len(songs)} songs and "
          f"{len(examples)} proxy pairs; pair accuracy "
          f"{metrics['pair_accuracy']:.3f}")
    return metrics


def bootstrap_validation(midi_dir: Path, steps: int = 200,
                         metrics_path: Path | None = None) -> dict:
    """Leave-song-out validation for the transparent proxy objective."""
    from noteblockify.mc_model import refine_candidates
    from noteblockify.song import arrange_pre

    songs = sorted(midi_dir.glob("*.mid"))
    if len(songs) < 2:
        raise ValueError("bootstrap validation needs at least two MIDI files")
    train_examples = []
    held_out = []
    for song_index, midi in enumerate(songs):
        pre = arrange_pre(midi)
        candidates = refine_candidates(pre)
        ranked = sorted(candidates.values(), key=lambda c: _proxy_loss(pre, c))
        pairs = [(
            torch.tensor(assignment_features(pre, ranked[0]), dtype=torch.float32),
            torch.tensor(assignment_features(pre, rejected), dtype=torch.float32),
        ) for rejected in ranked[1:]]
        (held_out if song_index == len(songs) - 1 else train_examples).extend(pairs)
    if not train_examples or not held_out:
        raise ValueError("could not build train/held-out candidate pairs")
    checkpoint = midi_dir.parent / ".bootstrap_validation.pt"
    _fit_examples(train_examples, checkpoint, steps, 2e-3)
    model = load_ranker(checkpoint)
    with torch.no_grad():
        correct = 0
        for preferred, rejected in held_out:
            correct += int(model(preferred).mean() > model(rejected).mean())
    checkpoint.unlink(missing_ok=True)
    metrics = {"train_pairs": len(train_examples),
               "held_out_pairs": len(held_out),
               "held_out_pair_accuracy": correct / len(held_out)}
    if metrics_path is not None:
        metrics_path.write_text(json.dumps(metrics, indent=2),
                                encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train the MC placement ranker from human A/B choices")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data", type=Path,
                        help="JSONL human preference file")
    source.add_argument("--midi-dir", type=Path,
                        help="bootstrap from MIDI using the transparent solver prior")
    source.add_argument("--midi-guided", type=Path,
                        help="train from automatic MIDI-guided candidate rewards")
    parser.add_argument("--output", type=Path, default=MODEL_PATH)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--init", type=Path, default=None,
                        help="checkpoint to continue with human preferences")
    parser.add_argument("--validate-bootstrap", action="store_true",
                        help="leave-one-song-out validation of proxy bootstrap")
    parser.add_argument("--validate", action="store_true",
                        help="validate human preference JSONL without training")
    parser.add_argument("--metrics", type=Path, default=None)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--depth", type=int, default=2)
    args = parser.parse_args(argv)
    if args.validate:
        if args.data is None:
            parser.error("--validate requires --data")
        print(json.dumps(validate_preferences(args.data), indent=2))
    elif args.midi_guided is not None:
        train_midi_guided(args.midi_guided, args.output, args.steps,
                          width=args.width, depth=args.depth)
    elif args.midi_dir is not None and args.validate_bootstrap:
        bootstrap_validation(args.midi_dir, args.steps, args.metrics)
    elif args.midi_dir is not None:
        train_bootstrap(args.midi_dir, args.output, args.steps)
    else:
        train_preferences(args.data, args.output, args.steps, init=args.init)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
