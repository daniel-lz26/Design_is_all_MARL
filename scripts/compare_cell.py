"""Build the cell comparison from every results/<policy>/*.csv (spec §11.1, §12).

    python scripts/compare_cell.py [--results results]

Writes results/summary.md (per scenario: every controller with mean ± 95% CI over its seeds,
the best family-A rule marked, paired differences vs the best rule and vs the best auction
with bootstrap p-values, and the nominal -> disrupt robustness table), results/summary.csv and
results/heatmap_on_time.png. A new method shows up as soon as its results folder exists.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from som.eval.compare import mean_ci95

ROOT = Path(__file__).resolve().parents[1]
PAIRS = [("cell_10", "cell_10_disrupt"), ("cell_random_10", "cell_random_10_disrupt")]
BOOT = 10_000


def load(results: Path) -> dict[str, dict[str, dict[int, dict[str, str]]]]:
    """scenario tag -> policy -> seed -> row"""
    data: dict[str, dict[str, dict[int, dict[str, str]]]] = defaultdict(dict)
    for path in sorted(results.glob("*/*.csv")):
        with path.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if rows:
            data[path.stem][rows[0]["policy"]] = {int(r["seed"]): r for r in rows}
    return data


def col(rows: dict[int, dict[str, str]], key: str) -> list[float]:
    return [float(r[key]) for r in rows.values() if r[key] not in ("", "nan")]


def paired(
    a: dict[int, dict[str, str]], b: dict[int, dict[str, str]], key: str, rng: np.random.Generator
) -> tuple[float, float, float]:
    """Mean of a - b over shared seeds, its 95% CI half-width and a two-sided bootstrap p."""
    seeds = sorted(set(a) & set(b))
    d = np.array([float(a[s][key]) - float(b[s][key]) for s in seeds])
    if len(d) < 2:
        return math.nan, math.nan, math.nan
    m, ci = mean_ci95(list(d))
    if np.all(d == d[0]):
        return m, ci, 0.0 if d[0] != 0 else 1.0
    boot = rng.choice(d - d.mean(), size=(BOOT, len(d))).mean(axis=1)
    p = float(np.mean(np.abs(boot) >= abs(d.mean())))
    return m, ci, p


def family(rows: dict[int, dict[str, str]]) -> str:
    return next(iter(rows.values()))["family"]


def best(policies: dict[str, dict[int, dict[str, str]]], fam: str) -> str | None:
    pool = [p for p, rows in policies.items() if family(rows) == fam]
    if not pool:
        return None
    return max(
        pool,
        key=lambda p: (
            np.mean(col(policies[p], "on_time_pct")),
            -np.mean(col(policies[p], "mean_tardiness")),
        ),
    )


def fmt(xs: list[float], digits: int = 1) -> str:
    if not xs:
        return "n/a"
    m, ci = mean_ci95(xs)
    return f"{m:.{digits}f} ± {ci:.{digits}f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--results", type=Path, default=ROOT / "results")
    args = ap.parse_args()
    data = load(args.results)
    rng = np.random.default_rng(0)
    md = [
        "# Cell results summary",
        "",
        "Mean ± 95% CI over each controller's seeds. Families: A = dispatching rules, "
        "B = auction, C = central scheduler, L = learned. Only the best family-A rule is "
        "listed per scenario; `diff vs …` are paired by seed (bootstrap p).",
        "",
    ]
    flat = []
    for tag in sorted(data):
        pols = data[tag]
        best_a, best_b = best(pols, "A"), best(pols, "B")
        md += [
            f"## {tag}",
            "",
            "| Controller | Fam | n | On-time % | Mean tardiness (s) | "
            "Makespan (s) | Blocked trips | Recovery (s) | vs best rule | vs best auction |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        shown = [p for p in pols if family(pols[p]) != "A" or p == best_a]
        for p in sorted(shown, key=lambda p: (family(pols[p]), p)):
            rows = pols[p]
            cells = []
            for ref in (best_a, best_b):
                if ref is None or ref == p or family(rows) == "A":
                    cells.append("")
                    continue
                m, ci, pv = paired(rows, pols[ref], "on_time_pct", rng)
                cells.append(f"{m:+.1f} ± {ci:.1f} (p={pv:.3f})")
            name = f"**{p}** (best A)" if p == best_a else p
            md.append(
                f"| {name} | {family(rows)} | {len(rows)} | {fmt(col(rows, 'on_time_pct'))} | "
                f"{fmt(col(rows, 'mean_tardiness'))} | {fmt(col(rows, 'makespan'), 0)} | "
                f"{fmt(col(rows, 'blocked_trips'))} | {fmt(col(rows, 'recovery_time'))} | "
                f"{cells[0]} | {cells[1]} |"
            )
        for p, rows in pols.items():
            flat.append(
                {
                    "scenario": tag,
                    "policy": p,
                    "family": family(rows),
                    "n": len(rows),
                    "on_time_pct": np.mean(col(rows, "on_time_pct")),
                    "mean_tardiness": np.mean(col(rows, "mean_tardiness")),
                    "makespan": np.mean(col(rows, "makespan")),
                }
            )
        md.append("")

    md += [
        "## Robustness (on-time % nominal -> disrupt)",
        "",
        "| Controller | " + " | ".join(f"{a} -> {b}" for a, b in PAIRS) + " |",
        "|---|" + "---|" * len(PAIRS),
    ]
    names = sorted({p for tag in data for p in data[tag]})
    for p in names:
        cells = []
        for a, b in PAIRS:
            if p in data.get(a, {}) and p in data.get(b, {}):
                x, y = (
                    np.mean(col(data[a][p], "on_time_pct")),
                    np.mean(col(data[b][p], "on_time_pct")),
                )
                cells.append(f"{x:.1f} -> {y:.1f} ({y - x:+.1f})")
            else:
                cells.append("n/a")
        if any(c != "n/a" for c in cells):
            md.append(f"| {p} | " + " | ".join(cells) + " |")
    out = args.results
    (out / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    if flat:
        with (out / "summary.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(flat[0]))
            w.writeheader()
            w.writerows(flat)
        heatmap(data, out / "heatmap_on_time.png")
    print(f"wrote {out / 'summary.md'} ({len(data)} scenarios, {len(names)} controllers)")


def heatmap(data: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tags = sorted(data)
    rows = []
    for tag in tags:
        a = best(data[tag], "A")
        rows += [p for p in data[tag] if family(data[tag][p]) != "A"] + ([a] if a else [])
    names = sorted(set(rows) - {None}, key=lambda p: (p.split("/")[0] in ("AUCTION",), p))
    best_label = "best rule (A)"
    grid = np.full((len(names) + 1, len(tags)), np.nan)
    for j, tag in enumerate(tags):
        a = best(data[tag], "A")
        if a:
            grid[0, j] = np.mean(col(data[tag][a], "on_time_pct"))
        for i, p in enumerate(names, start=1):
            if p in data[tag] and family(data[tag][p]) != "A":
                grid[i, j] = np.mean(col(data[tag][p], "on_time_pct"))
    labels = [best_label, *names]
    keep = [i for i in range(len(labels)) if not np.all(np.isnan(grid[i]))]
    grid, labels = grid[keep], [labels[i] for i in keep]
    fig, ax = plt.subplots(figsize=(1.1 * len(tags) + 3, 0.45 * len(labels) + 1.5))
    im = ax.imshow(grid, cmap="viridis", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(tags)), tags, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if not np.isnan(grid[i, j]):
                ax.text(
                    j,
                    i,
                    f"{grid[i, j]:.0f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if grid[i, j] < 50 else "black",
                )
    fig.colorbar(im, ax=ax, label="on-time %")
    ax.set_title("On-time % by controller and scenario (test seeds)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
