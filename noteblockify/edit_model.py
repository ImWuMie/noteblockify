"""Learned ranking for bounded note-edit candidates.

The legacy placement ranker only chooses legal octaves and requires equal note
counts. This module ranks actions that keep, replace, drop, or add a bounded
source-grounded note. Human A/B labels remain the listening signal.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch import nn

FEATURE_COUNT = 16
WINDOW_LO = 33
WINDOW_HI = 57
MODEL_PATH = Path(__file__).resolve().parent.parent / "mc_edit_ranker.pt"
_CHANNEL_RE = re.compile(r"^ch(\d+):")
_ACTIONS = ("keep", "drop", "replace", "add")


class EditRanker(nn.Module):
    """Scalar scorer for one source-note edit action."""

    def __init__(self, width: int = 96, depth: int = 2):
        super().__init__()
        if width < 8 or depth < 1:
            raise ValueError("edit ranker width must be >= 8 and depth >= 1")
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


def _voice_id(song, note) -> str:
    if 0 <= note.layer < len(song.layers):
        match = _CHANNEL_RE.match(song.layers[note.layer].name)
        if match:
            return f"ch{match.group(1)}"
    return f"layer{note.layer}"


def _contexts(song):
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


def action_features(song, index: int, candidate_key: int | None,
                    action: str, contexts=None) -> list[float]:
    """Build features for one keep/drop/replace/add action."""
    if action not in _ACTIONS:
        raise ValueError(f"unknown edit action: {action}")
    note = song.notes[index]
    previous, following, position, voice_size, density, layer, progress = (
        (contexts or _contexts(song))[index])
    raw = note.key
    represented = -1 if candidate_key is None else candidate_key
    shift = 0.0 if candidate_key is None else (represented - raw) / 12.0
    one_hot = [float(action == name) for name in _ACTIONS]
    return [
        raw / 87.0,
        represented / 87.0,
        shift / 8.0,
        note.instrument / 15.0,
        previous / 87.0,
        following / 87.0,
        position,
        voice_size,
        density,
        layer,
        progress,
        float(raw < WINDOW_LO or raw > WINDOW_HI),
        *one_hot,
    ]


def _ordered(song):
    return sorted(range(len(song.notes)),
                  key=lambda i: (song.notes[i].tick, song.notes[i].layer, i))


def _slot(note):
    return (note.tick, note.layer, note.instrument,
            note.velocity, note.panning)


def _action_rows(pre, candidate) -> list[list[float]]:
    slots = defaultdict(list)
    for index in _ordered(candidate):
        slots[_slot(candidate.notes[index])].append(candidate.notes[index])
    contexts = _contexts(pre)
    rows = []
    pre_order = _ordered(pre)
    for index in pre_order:
        source = pre.notes[index]
        matches = slots[_slot(source)]
        target = matches.pop(0) if matches else None
        if target is None:
            rows.append(action_features(pre, index, None, "drop", contexts))
        else:
            action = "keep" if (target.key - source.key) % 12 == 0 else "replace"
            rows.append(action_features(pre, index, target.key, action, contexts))
    extras = [note for values in slots.values() for note in values]
    for extra in extras:
        index = min(pre_order,
                    key=lambda i: (abs(pre.notes[i].tick - extra.tick),
                                   abs(pre.notes[i].layer - extra.layer)))
        rows.append(action_features(pre, index, extra.key, "add", contexts))
    return rows


def candidate_score(model: EditRanker | None, pre, candidate) -> float:
    rows = _action_rows(pre, candidate)
    if model is None or not rows:
        return 0.0
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        values = model(torch.tensor(rows, dtype=torch.float32, device=device))
    return float(values.mean())


def score_options(model: EditRanker, song, index: int,
                  options: list[tuple[str, int | None]], contexts=None) -> list[float]:
    """Score edit options in one model call."""
    if not options:
        return []
    context = contexts or _contexts(song)
    rows = [action_features(song, index, key, action, context)
            for action, key in options]
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        values = model(torch.tensor(rows, dtype=torch.float32, device=device))
    return [float(value) for value in values.tolist()]


def load_ranker(path: Path | None = None) -> EditRanker | None:
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
        model = EditRanker(width=width, depth=depth).to(device)
        model.load_state_dict(state)
        model.eval()
        return model
    except (OSError, RuntimeError, TypeError, ValueError):
        return None


def _records(path: Path):
    import pynbs

    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                yield (pynbs.read(row["pre"]),
                       pynbs.read(row["preferred"]),
                       pynbs.read(row["rejected"]))
            except (KeyError, OSError, TypeError, ValueError,
                    json.JSONDecodeError) as exc:
                raise ValueError(f"invalid edit preference row "
                                 f"{line_number} in {path}: {exc}") from exc


def train_edit_preferences(data: Path, output: Path = MODEL_PATH,
                           steps: int = 1000, learning_rate: float = 2e-3,
                           width: int = 96, depth: int = 2) -> dict:
    """Fit pairwise preferences where candidate note counts may differ."""
    records = list(_records(data))
    if not records:
        raise ValueError(f"no edit preference rows in {data}")
    preferred = [torch.tensor(_action_rows(pre, good), dtype=torch.float32)
                 for pre, good, _bad in records]
    rejected = [torch.tensor(_action_rows(pre, bad), dtype=torch.float32)
                for pre, _good, bad in records]
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EditRanker(width=width, depth=depth).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model.train()
    losses = []
    for _step in range(steps):
        preferred_values = torch.stack([
            model(rows.to(device)).mean() for rows in preferred])
        rejected_values = torch.stack([
            model(rows.to(device)).mean() for rows in rejected])
        loss = torch.nn.functional.softplus(
            -(preferred_values - rejected_values)).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    model.eval()
    with torch.no_grad():
        accuracy = sum(
            candidate_score(model, pre, good) > candidate_score(model, pre, bad)
            for pre, good, bad in records) / len(records)
    output.parent.mkdir(parents=True, exist_ok=True)
    cpu_state = {key: value.detach().cpu()
                 for key, value in model.state_dict().items()}
    torch.save({"state_dict": cpu_state, "width": width, "depth": depth},
               output)
    metrics = {"examples": len(records), "steps": steps,
               "final_loss": losses[-1], "pair_accuracy": accuracy,
               "device": str(device), "width": width, "depth": depth}
    output.with_suffix(".json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"wrote edit ranker {output} from {len(records)} preferences; "
          f"pair accuracy {accuracy:.3f}")
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train the ranker for note-edit candidate songs")
    parser.add_argument("--edit-data", type=Path, required=True,
                        help="JSONL A/B preferences for edit candidates")
    parser.add_argument("--output", type=Path, default=MODEL_PATH)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--depth", type=int, default=2)
    args = parser.parse_args(argv)
    train_edit_preferences(args.edit_data, args.output, args.steps,
                           width=args.width, depth=args.depth)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
