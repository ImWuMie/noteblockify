"""Record pairwise listening preferences for the learned MC ranker."""

from __future__ import annotations

import argparse
from pathlib import Path

from noteblockify.preference_model import append_preference


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Append a human A/B choice to a JSONL preference file")
    parser.add_argument("--data", type=Path, default=Path("data/preferences.jsonl"))
    parser.add_argument("--pre", type=Path, required=True)
    parser.add_argument("--preferred", type=Path, required=True)
    parser.add_argument("--rejected", type=Path, required=True)
    parser.add_argument("--note", default="")
    args = parser.parse_args(argv)
    append_preference(args.data, args.pre, args.preferred, args.rejected,
                      args.note)
    print(f"recorded preference in {args.data}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
