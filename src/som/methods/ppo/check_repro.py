"""Check that a fresh baseline evaluation reproduces the committed results, seed by seed.

    python -m som.methods.ppo.check_repro runs/repro          # compares with results/

Every CSV under the given folder is matched to results/<policy>/<scenario>.csv and compared row
by row on every metric (env_version and wall time excluded), with a small relative tolerance
for floating-point differences between machines. Exit code 1 if anything differs.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SKIP = {"env_version", "wall_s"}


def same(a: str, b: str) -> bool:
    if a == b:
        return True
    try:
        x, y = float(a), float(b)
    except ValueError:
        return False
    if math.isnan(x) and math.isnan(y):
        return True
    return math.isclose(x, y, rel_tol=1e-6, abs_tol=1e-6)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("fresh", type=Path)
    ap.add_argument("--reference", type=Path, default=ROOT / "results")
    args = ap.parse_args()
    files = sorted(args.fresh.glob("*/*.csv"))
    if not files:
        raise SystemExit(f"no csv files under {args.fresh}")
    episodes = differ = missing = 0
    for f in files:
        ref = args.reference / f.parent.name / f.name
        if not ref.exists():
            print(f"no reference for {f.parent.name}/{f.name}")
            missing += 1
            continue
        a = list(csv.DictReader(f.open(encoding="utf-8")))
        b = {r["seed"]: r for r in csv.DictReader(ref.open(encoding="utf-8"))}
        for row in a:
            episodes += 1
            other = b.get(row["seed"])
            keys = [k for k in row if k not in SKIP]
            bad = [k for k in keys if other is None or not same(row[k], other.get(k, ""))]
            if bad:
                differ += 1
                if differ <= 20:
                    print(f"DIFF {f.parent.name}/{f.name} seed {row['seed']}: {', '.join(bad)}")
    print(f"{len(files)} files, {episodes} episodes compared: {differ} differ, {missing} missing")
    sys.exit(1 if differ or missing else 0)


if __name__ == "__main__":
    main()
