"""Train a per-note octave-placement model.

Every vanilla instrument plays comfortably within roughly two octaves
(_RANGE in noteblockify.song). The folding rule in noteblockify.song keeps pitch class
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

from noteblockify.song import FOLD_HI, FOLD_LO, _RANGE, _events, _PROGRAM, _DRUM

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
            octave = _PROGRAM[programs[ch]][1] if False else 0
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


class OctaveTransformer(nn.Module):
    """Per-note transformer over each channel's note sequence.

    The MLP sees one note at a time; this model attends along the
    channel, so a note's octave placement can depend on the whole
    melodic line it belongs to, not just its two neighbours.
    Same features, same head count; a drop-in upgrade.
    """

    def __init__(self, dim: int = 96, depth: int = 3, heads: int = 4,
                 window: int = 256):
        super().__init__()
        self.window = window
        self.embed = nn.Linear(7, dim)
        self.pos = nn.Embedding(window, dim)
        layer = nn.TransformerEncoderLayer(
            dim, heads, dim * 4, batch_first=True, dropout=0.1)
        self.encoder = nn.TransformerEncoder(layer, depth)
        self.head = nn.Linear(dim, len(SHIFTS))

    def forward(self, x):
        """x: (B, N, 7) padded channel windows; mask out padding."""
        pad = (x[..., :1].abs().sum(-1) == 0) & (torch.arange(
            x.shape[1], device=x.device)[None, :] >= 0)
        keep = ~pad.all(-1, keepdim=True).squeeze(-1)
        # A fully-zero feature row marks padding (instrument 0, key 0,
        # ... all zero); build the mask from any real column.
        mask = (x.abs().sum(-1) != 0)
        h = self.embed(x) + self.pos.weight[: x.shape[1]].expand(
            x.shape[0], -1, -1)
        h = self.encoder(h, src_key_padding_mask=~mask)
        out = self.head(h)
        out = out * mask.unsqueeze(-1)
        return out


def _channel_windows(midi_path: Path):
    """(features, targets) per channel window for the transformer."""
    events, programs, tpb, usec = _events(midi_path)
    if not events:
        return []
    channels = sorted({e[1] for e in events})
    inst = {c: (_PROGRAM[programs[c]][0] if c != 9 else None) for c in channels}
    by_ch: dict[int, list] = {}
    for pos, ch, note, vel in events:
        if ch == 9:
            di, dk = _DRUM[note]
            key = dk + 33
            instrument = di
        else:
            instrument = inst[ch]
            key = note - 21 + 12 * _PROGRAM[programs[ch]][1]
        by_ch.setdefault(ch, []).append((key, instrument, vel))
    windows = []
    for ch, seq in by_ch.items():
        # Label: median snapped to whole octaves into the hard window,
        # same as the oracle.
        mid = sorted(k for k, _i, _v in seq)[len(seq) // 2]
        shift = 0
        while mid + 12 * shift < FOLD_LO:
            shift += 1
        while mid + 12 * shift > FOLD_HI:
            shift -= 1
        label = SHIFTS.index(shift) if shift in SHIFTS else 2
        center = (FOLD_LO + FOLD_HI) / 2
        for start in range(0, len(seq), 256):
            chunk = seq[start: start + 256]
            feats = []
            for i, (key, instrument, vel) in enumerate(chunk):
                prev_key = chunk[i - 1][0] if i > 0 else key
                next_key = chunk[i + 1][0] if i + 1 < len(chunk) else key
                feats.append(note_features(instrument, key, prev_key,
                                           next_key, vel))
            windows.append((torch.tensor(feats, dtype=torch.float32),
                            torch.full((len(chunk),), label, dtype=torch.long)))
    return windows


def train_transformer() -> None:
    """Train OctaveTransformer on channel windows; saves octave_tf.pt."""
    torch.manual_seed(0)
    random.seed(0)
    mids = sorted(Path("data/midi").glob("*.mid"))
    data = []
    for mid in mids:
        data.extend(_channel_windows(mid))
    print(f"{len(mids)} songs, {len(data)} windows", flush=True)
    if not data:
        raise SystemExit("no data")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = OctaveTransformer().to(device)
    ys = torch.cat([t for _f, t in data]).to(device)
    counts = torch.bincount(ys, minlength=len(SHIFTS)).float()
    weights = (counts.sum() / (len(SHIFTS) * counts.clamp(min=1))).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    started = time.perf_counter()
    step = 0
    while True:
        batch = random.sample(data, min(16, len(data)))
        width = max(f.shape[0] for f, _t in batch)
        x = torch.zeros(len(batch), width, 7)
        t = torch.full((len(batch), width), -100, dtype=torch.long)
        for b, (f, tg) in enumerate(batch):
            x[b, : f.shape[0]] = f
            t[b, : tg.shape[0]] = tg
        x, t = x.to(device), t.to(device)
        logits = model(x)
        mask = t != -100
        loss = nn.functional.cross_entropy(
            logits[mask], t[mask], weight=weights)
        opt.zero_grad()
        loss.backward()
        opt.step()
        step += 1
        if step % 100 == 0:
            with torch.no_grad():
                correct = ((logits[mask].argmax(-1)) == t[mask]).float().mean()
            rate = step / (time.perf_counter() - started)
            print(f"step {step} loss {loss.item():.4f} acc {correct:.3f} "
                  f"{rate:.0f} steps/s", flush=True)
            torch.save(model.cpu().state_dict(), "octave_tf.pt")
            model.to(device)


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
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "transformer":
        train_transformer()
    else:
        main()
