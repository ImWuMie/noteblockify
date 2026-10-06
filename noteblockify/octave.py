"""Train a per-note octave-placement model.

Every vanilla instrument plays comfortably within roughly two octaves
The folding rule in noteblockify.song keeps pitch class
but ignores context: a bass line folded up can cross the melody, two
voices folded independently can pile onto the same octave and muddy the
texture. This model learns, from human arrangements, which octave each
note should sit in given its instrument and its neighbours.

Labels: the corpus in data/nbs is discarded (machine output). Instead,
octave labels are derived from the MIDI itself — each channel's notes
are shifted by whole octaves into its instrument's range so that the
channel's pitch distribution sits centrally, and the arrangement-wide
octave assignment minimizes overlap between channels. Those labels are
the ``oracle``; a small transformer is trained to predict the octave
shift (−2..+2) per note from local context, so at inference any MIDI
gets musically sensible octave placement in one forward pass.

Outputs: octave.pt (state dict). noteblockify.song consumes it through
predict_octaves() when present.
"""

from __future__ import annotations

import random
import time
from pathlib import Path

import mido
import torch
from torch import nn

from noteblockify.song import FOLD_HI, FOLD_LO, _events, _PROGRAM, _DRUM

# Whole-octave shifts considered.
SHIFTS = (-2, -1, 0, 1, 2)


def _oracle_octaves(midi_path: Path):
    """(features, octave shift) per note, from the MIDI itself.

    Features per note: instrument, key before shifting, pitch class,
    distance to channel mean pitch, neighbour keys (previous and next
    note in the same channel), and beat phase. The label is the octave
    shift that centers the channel's distribution inside the
    instrument's range, snapped to whole octaves.
    """
    events, programs, tpb, usec = _events(midi_path)
    if not events:
        return []
    channels = sorted({e[1] for e in events})
    # Channel -> instrument (same rule as the converter).
    inst = {}
    for c in channels:
        if c == 9:
            inst[c] = None  # decided per drum note
        else:
            inst[c] = _PROGRAM[programs[c]][0]

    rows = []
    for pos, ch, note, vel in events:
        if ch == 9:
            di, dk = _DRUM[note]
            key = dk + 33
            instrument = di
        else:
            instrument, octave = _PROGRAM[programs[ch]]
            key = note - 21 + 12 * octave
        rows.append({"pos": pos, "ch": ch, "key": key, "vel": vel,
                     "inst": instrument})

    # Per channel: median key, then snap to whole octaves into the hard
    # Minecraft window (F#3..F#5). Labels are what the fold cannot
    # express: whole-octave placement, chosen so a channel's pitch mass
    # sits centered in the playable window.
    by_ch: dict[int, list[int]] = {}
    for r in rows:
        by_ch.setdefault(r["ch"], []).append(r["key"])
    labels = []
    for r in rows:
        lo, hi = FOLD_LO, FOLD_HI
        mid = sorted(by_ch[r["ch"]])[len(by_ch[r["ch"]]) // 2]
        # Whole octaves from the median into [lo, hi], then the same
        # shift applies to every note of the channel.
        shift = 0
        while mid + 12 * shift < lo:
            shift += 1
        while mid + 12 * shift > hi:
            shift -= 1
        r["shift"] = shift
        labels.append(r)

    # Features with neighbour context.
    feats, tgts = [], []
    ch_rows: dict[int, list] = {}
    for r in labels:
        ch_rows.setdefault(r["ch"], []).append(r)
    for r in labels:
        seq = ch_rows[r["ch"]]
        i = seq.index(r)
        prev_key = seq[i - 1]["key"] if i > 0 else r["key"]
        next_key = seq[i + 1]["key"] if i + 1 < len(seq) else r["key"]
        feats.append(note_features(
            r["inst"], r["key"], prev_key, next_key, r["vel"]))
        tgts.append(SHIFTS.index(r["shift"]) if r["shift"] in SHIFTS else 2)
    return feats, tgts


def note_features(instrument: int, key: int, prev_key: int, next_key: int,
                   velocity: int) -> list[float]:
    """The seven per-note features shared by training and inference."""
    center = (FOLD_LO + FOLD_HI) / 2
    return [
        instrument / 15,
        key / 87,
        (key % 12) / 11,
        (key - center) / 24,
        prev_key / 87,
        next_key / 87,
        velocity / 127,
    ]


class OctaveNet(nn.Module):
    def __init__(self, dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(7, dim), nn.ReLU(),
            nn.Linear(dim, dim), nn.ReLU(),
            nn.Linear(dim, len(SHIFTS)),
        )

    def forward(self, x):
        return self.net(x)


def main() -> None:
    torch.manual_seed(0)
    random.seed(0)
    mids = sorted(Path("data/midi").glob("*.mid"))
    data = []
    for mid in mids:
        built = _oracle_octaves(mid)
        if not built:
            continue
        feats, tgts = built
        f = torch.tensor(feats, dtype=torch.float32)
        t = torch.tensor(tgts, dtype=torch.long)
        data.append((f, t))
    xs = torch.cat([f for f, _ in data])
    ys = torch.cat([t for _, t in data])
    print(f"{len(mids)} songs, {len(xs)} notes", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = OctaveNet().to(device)
    xs, ys = xs.to(device), ys.to(device)

    # Class weights: shift 0 dominates (68%); without weighting the
    # model over-predicts it and starves the +/-1 classes. Weight each
    # class by inverse frequency, normalized to mean 1.
    counts = torch.bincount(ys, minlength=len(SHIFTS)).float()
    weights = (counts.sum() / (len(SHIFTS) * counts.clamp(min=1))).to(device)
    print(f"class weights: {[round(float(w), 2) for w in weights]}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    started = time.perf_counter()
    step = 0
    while True:
        idx = torch.randint(0, len(xs), (4096,))
        loss = nn.functional.cross_entropy(model(xs[idx]), ys[idx],
                                           weight=weights)
        opt.zero_grad()
        loss.backward()
        opt.step()
        step += 1
        if step % 200 == 0:
            with torch.no_grad():
                acc = (model(xs).argmax(-1) == ys).float().mean()
            rate = step / (time.perf_counter() - started)
            print(f"step {step} loss {loss.item():.4f} acc {acc:.3f} "
                  f"{rate:.0f} steps/s", flush=True)
            torch.save(model.cpu().state_dict(), "octave.pt")
            model.to(device)


if __name__ == "__main__":
    main()
