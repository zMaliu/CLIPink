import argparse
import csv
import json
from pathlib import Path


SUMMARY_NAME = "summary.json"
CONFIG_NAME = "config.json"
FINAL_IMAGE_NAME = "final_highres.png"


def _iter_runs(runs_root: Path):
    for summary_path in runs_root.rglob(SUMMARY_NAME):
        run_dir = summary_path.parent
        parts = run_dir.relative_to(runs_root).parts
        if len(parts) < 3 or parts[0] == "_reports":
            continue
        experiment, image, seed = parts[0], parts[1], parts[2]
        yield experiment, image, seed, run_dir


def _read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _to_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _mean(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _std(values):
    vals = [v for v in values if v is not None]
    if len(vals) <= 1:
        return 0.0 if vals else None
    mu = _mean(vals)
    return (sum((v - mu) ** 2 for v in vals) / len(vals)) ** 0.5


def collect(runs_root: Path):
    rows = []
    for experiment, image, seed, run_dir in _iter_runs(runs_root):
        summary = _read_json(run_dir / SUMMARY_NAME)
        config_path = run_dir / CONFIG_NAME
        config = _read_json(config_path) if config_path.exists() else {}
        row = {
            "experiment": experiment,
            "image": image,
            "seed": seed,
            "run_dir": str(run_dir),
            "final_image_path": str(run_dir / FINAL_IMAGE_NAME),
            "loss": _to_float(summary.get("loss")),
            "clip_loss": _to_float(summary.get("clip_loss")),
            "sparse_loss": _to_float(summary.get("sparse_loss")),
            "ink_loss": _to_float(summary.get("ink_loss")),
            "l2_loss": _to_float(summary.get("l2_loss")),
            "edge_loss": _to_float(summary.get("edge_loss")),
            "active_count": _to_float(summary.get("active_count")),
            "active_ratio": _to_float(summary.get("active_ratio")),
            "ink_mass": _to_float(summary.get("ink_mass")),
            "whitespace_ratio": _to_float(summary.get("whitespace_ratio")),
            "n_strokes": config.get("n_strokes"),
            "render_profile": config.get("render_profile"),
            "enable_gate": config.get("enable_gate"),
            "layered_init": config.get("layered_init"),
        }
        rows.append(row)
    return rows


def summarize(rows):
    grouped = {}
    for row in rows:
        key = (row["experiment"], row["image"])
        grouped.setdefault(key, []).append(row)
    summary_rows = []
    metric_keys = [
        "loss",
        "clip_loss",
        "sparse_loss",
        "ink_loss",
        "l2_loss",
        "edge_loss",
        "active_count",
        "active_ratio",
        "ink_mass",
        "whitespace_ratio",
    ]
    for (experiment, image), items in sorted(grouped.items()):
        row = {
            "experiment": experiment,
            "image": image,
            "num_seeds": len(items),
        }
        for key in metric_keys:
            vals = [_to_float(item.get(key)) for item in items]
            row[f"{key}_mean"] = _mean(vals)
            row[f"{key}_std"] = _std(vals)
        summary_rows.append(row)
    return summary_rows


def _write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    keys = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Collect run summaries under runs/ into CSV reports.")
    parser.add_argument("--runs_root", type=str, default="runs")
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    out_dir = Path(args.out_dir) if args.out_dir is not None else runs_root / "_reports"
    rows = collect(runs_root)
    summary_rows = summarize(rows)
    _write_csv(out_dir / "results.csv", rows)
    _write_csv(out_dir / "results_summary.csv", summary_rows)
    print(f"[collect] wrote {out_dir / 'results.csv'}")
    print(f"[collect] wrote {out_dir / 'results_summary.csv'}")


if __name__ == "__main__":
    main()
