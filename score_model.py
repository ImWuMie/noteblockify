"""Score the deployed octave model (octave.pt) on the corpus.

Reports train-set and held-out (leave-5-out, fresh retrain on 18) accuracy,
per-class recall, and the conversion-level score with the model enabled.
"""

import random
import collections
from pathlib import Path

import torch
import torch.nn as nn

from noteblockify.octave import OctaveNet, SHIFTS, _oracle_octaves
from noteblockify.song import arrange
from noteblockify.hear import compare


def per_class(model, xs, ys):
    per, tot = collections.Counter(), collections.Counter()
    with torch.no_grad():
        pred = model(xs).argmax(-1)
    for t, p in zip(ys.tolist(), pred.tolist()):
        tot[t] += 1
        if t == p:
            per[t] += 1
    return per, tot


mids = sorted(Path("data/midi").glob("*.mid"))

# ---- 1) Deployed model, train-set ----
model = OctaveNet()
model.load_state_dict(torch.load("octave.pt", map_location="cpu"))
model.eval()
fs, ts, names = [], [], []
for mid in mids:
    b = _oracle_octaves(mid)
    if not b:
        continue
    fs.append(torch.tensor(b[0], dtype=torch.float32))
    ts.append(torch.tensor(b[1], dtype=torch.long))
    names.append(mid.stem)
xs = torch.cat(fs)
ys = torch.cat(ts)
per, tot = per_class(model, xs, ys)
print("=== deployed octave.pt (weighted MLP, train set) ===")
for s in range(5):
    if tot[s]:
        print(f"  shift {SHIFTS[s]:+d}: recall {per[s]/tot[s]:.3f}  ({tot[s]} notes)")
print(f"  overall {sum(per.values())/sum(tot.values()):.4f}")

# Per-song worst
worst = []
for f, t, name in zip(fs, ts, names):
    with torch.no_grad():
        acc = (model(f).argmax(-1) == t).float().mean().item()
    worst.append((acc, name))
worst.sort()
print("  worst songs:", [(n[:28], round(a, 3)) for a, n in worst[:3]])

# ---- 2) Held-out: fresh 18/5 split ----
random.seed(1)
order = list(range(len(mids)))
random.shuffle(order)
test_idx = set(order[:5])

train_fs = [fs[i] for i in range(len(fs)) if i not in test_idx]
train_ts = [ts[i] for i in range(len(ts)) if i not in test_idx]
txs, tys = torch.cat(train_fs), torch.cat(train_ts)
counts = torch.bincount(tys, minlength=5).float()
weights = counts.sum() / (5 * counts.clamp(min=1))
fresh = OctaveNet()
opt = torch.optim.AdamW(fresh.parameters(), lr=3e-3)
for _ in range(8000):
    idx = torch.randint(0, len(txs), (4096,))
    loss = nn.functional.cross_entropy(fresh(txs[idx]), tys[idx], weight=weights)
    opt.zero_grad()
    loss.backward()
    opt.step()
fresh.eval()
test_xs = torch.cat([fs[i] for i in sorted(test_idx)])
test_ys = torch.cat([ts[i] for i in sorted(test_idx)])
per, tot = per_class(fresh, test_xs, test_ys)
print("=== held-out (18 train / 5 test, fresh model) ===")
for s in range(5):
    if tot[s]:
        print(f"  shift {SHIFTS[s]:+d}: recall {per[s]/tot[s]:.3f}  ({tot[s]} notes)")
print(f"  overall {sum(per.values())/sum(tot.values()):.4f}")

# ---- 3) Conversion-level score with the model in the loop ----
rows = []
for mid in mids:
    song = arrange(mid)
    out = Path("data/auto") / (mid.stem + ".nbs")
    song.save(out)
    rows.append(compare(mid, out).total)
print(f"=== conversion score with model ===")
print(f"  corpus mean {sum(rows)/len(rows):.4f}  min {min(rows):.3f}  max {max(rows):.3f}")
