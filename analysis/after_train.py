import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps, ImageDraw

FINAL_IMAGE_NAME = "final_highres.png"


DEFAULT_EXPERIMENTS = [
    "full",
    "no_sparse",
    "no_gate",
    "no_ink",
    "no_l2",
    "no_edge",
    "softline",
]


def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _to_float(x: Optional[str]) -> Optional[float]:
    if x is None or x == "":
        return None
    try:
        return float(x)
    except Exception:
        return None


def _parse_bool(x: Optional[str]) -> Optional[bool]:
    if x is None:
        return None
    s = str(x).strip().lower()
    if s in {"true", "1"}:
        return True
    if s in {"false", "0"}:
        return False
    return None


def _seed_to_int(seed_name: str) -> int:
    # expected format: seed_0
    if seed_name.startswith("seed_"):
        return int(seed_name.split("_", 1)[1])
    return int(seed_name)


def _resolve_final_image_path(
    raw_path: str,
    runs_root: Path,
    reports_dir: Path,
    experiment: str,
    image_name: str,
    seed_name: str,
):
    candidates: List[Path] = []
    rel = Path(str(raw_path))

    # 1) absolute path
    if rel.is_absolute():
        candidates.append(rel)
    else:
        # 2) project cwd + relative path from csv
        candidates.append(Path.cwd() / rel)
        # 3) runs_root.parent + relative path from csv
        candidates.append(runs_root.parent / rel)
        # extra-safe fallbacks
        candidates.append(runs_root / rel)
        candidates.append(reports_dir / rel)

    # 4) canonical fallback from run layout
    candidates.append(runs_root / experiment / image_name / seed_name / FINAL_IMAGE_NAME)

    seen = set()
    deduped: List[Path] = []
    for p in candidates:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(p)

    for p in deduped:
        if p.exists():
            return p, deduped
    return None, deduped


def _pick_seed_row(rows: List[Dict[str, str]], exp: str, image: str, seed: int) -> Optional[Dict[str, str]]:
    candidates = [r for r in rows if r.get("experiment") == exp and r.get("image") == image]
    if not candidates:
        return None
    exact = [r for r in candidates if _seed_to_int(r.get("seed", "0")) == seed]
    if exact:
        return exact[0]
    # fallback: choose row with minimal clip_loss
    best = sorted(candidates, key=lambda r: (_to_float(r.get("clip_loss")) is None, _to_float(r.get("clip_loss")) or 1e9))[0]
    return best


def _pick_summary_row(rows: List[Dict[str, str]], exp: str, image: str) -> Optional[Dict[str, str]]:
    candidates = [r for r in rows if r.get("experiment") == exp and r.get("image") == image]
    return candidates[0] if candidates else None


def _open_image(path: Path, tile_size: int) -> Image.Image:
    img = Image.open(path).convert("RGB")
    img = ImageOps.fit(img, (tile_size, tile_size), method=Image.Resampling.BICUBIC)
    return img


def _draw_label(img: Image.Image, text: str) -> Image.Image:
    canvas = img.copy()
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, canvas.width, 24), fill=(255, 255, 255))
    draw.text((6, 4), text, fill=(0, 0, 0))
    return canvas


def build_main_panel(
    results_rows: List[Dict[str, str]],
    image_name: str,
    experiments: List[str],
    seed: int,
    runs_root: Path,
    reports_dir: Path,
    source_image_path: Optional[Path],
    out_path: Path,
    tile_size: int = 320,
):
    tiles: List[Image.Image] = []
    labels: List[str] = []

    if source_image_path is not None and source_image_path.exists():
        src = _open_image(source_image_path, tile_size)
        tiles.append(_draw_label(src, "input"))
        labels.append("input")

    for exp in experiments:
        row = _pick_seed_row(results_rows, exp, image_name, seed)
        if row is None:
            blank = Image.new("RGB", (tile_size, tile_size), (245, 245, 245))
            tiles.append(_draw_label(blank, f"{exp} (missing)"))
            labels.append(exp)
            continue
        resolved_path, tried = _resolve_final_image_path(
            raw_path=row.get("final_image_path", ""),
            runs_root=runs_root,
            reports_dir=reports_dir,
            experiment=exp,
            image_name=image_name,
            seed_name=row.get("seed", f"seed_{seed}"),
        )
        if resolved_path is not None:
            img_path = resolved_path
            img = _open_image(img_path, tile_size)
            tiles.append(_draw_label(img, exp))
        else:
            blank = Image.new("RGB", (tile_size, tile_size), (245, 245, 245))
            tiles.append(_draw_label(blank, f"{exp} (missing img)"))
            short = [str(p) for p in tried[:4]]
            print(
                f"[after_train][missing] exp={exp} image={image_name} seed={row.get('seed')} "
                f"raw={row.get('final_image_path')} tried={short}",
                flush=True,
            )
        labels.append(exp)

    n = len(tiles)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    panel = Image.new("RGB", (cols * tile_size, rows * tile_size), (255, 255, 255))
    for i, tile in enumerate(tiles):
        r = i // cols
        c = i % cols
        panel.paste(tile, (c * tile_size, r * tile_size))

    _ensure_dir(out_path.parent)
    panel.save(out_path)


def _metric_value(summary_row: Dict[str, str], key: str) -> float:
    return _to_float(summary_row.get(f"{key}_mean")) or 0.0


def build_metric_plot(
    summary_rows: List[Dict[str, str]],
    image_name: str,
    experiments: List[str],
    out_path: Path,
):
    metrics = ["clip_loss", "edge_loss", "l2_loss", "active_ratio", "ink_mass", "whitespace_ratio"]
    data = []
    labels = []
    for exp in experiments:
        row = _pick_summary_row(summary_rows, exp, image_name)
        if row is None:
            continue
        labels.append(exp)
        data.append([_metric_value(row, m) for m in metrics])

    if not data:
        raise ValueError(f"No summary rows found for image={image_name}")

    arr = np.array(data, dtype=np.float32)
    # normalize per-metric for comparable bars
    min_v = arr.min(axis=0, keepdims=True)
    max_v = arr.max(axis=0, keepdims=True)
    norm = (arr - min_v) / (max_v - min_v + 1e-8)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    axes = axes.flatten()
    x = np.arange(len(labels))
    for i, m in enumerate(metrics):
        ax = axes[i]
        ax.bar(x, norm[:, i], color="#4C78A8")
        ax.set_title(m)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylim(0.0, 1.0)
    fig.suptitle(f"Ablation Metrics (normalized) - {image_name}")
    fig.tight_layout()
    _ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_metric_table(
    summary_rows: List[Dict[str, str]],
    image_name: str,
    experiments: List[str],
    out_path: Path,
):
    keys = [
        "experiment",
        "num_seeds",
        "clip_loss_mean",
        "edge_loss_mean",
        "l2_loss_mean",
        "active_ratio_mean",
        "ink_mass_mean",
        "whitespace_ratio_mean",
    ]
    rows = []
    for exp in experiments:
        row = _pick_summary_row(summary_rows, exp, image_name)
        if row is None:
            continue
        rows.append({k: row.get(k, "") for k in keys})

    _ensure_dir(out_path.parent)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Post-train report generator: main comparison panel + metric plots/tables.")
    parser.add_argument("--runs_root", type=str, default="runs")
    parser.add_argument("--reports_dir", type=str, default=None)
    parser.add_argument("--image", type=str, required=True, help="Image stem used in runs (e.g., camel)")
    parser.add_argument("--seed", type=int, default=0, help="Preferred seed for qualitative panel")
    parser.add_argument(
        "--experiments",
        type=str,
        default=",".join(DEFAULT_EXPERIMENTS),
        help="Comma-separated experiment names for panel/table order",
    )
    parser.add_argument("--target_image", type=str, default=None, help="Optional original input image path")
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    reports_dir = Path(args.reports_dir) if args.reports_dir else (runs_root / "_reports")
    results_csv = reports_dir / "results.csv"
    summary_csv = reports_dir / "results_summary.csv"
    if not results_csv.exists() or not summary_csv.exists():
        raise FileNotFoundError(
            f"Missing report CSVs under {reports_dir}. Run: python analysis/collect_results.py --runs_root {runs_root}"
        )

    results_rows = _read_csv(results_csv)
    summary_rows = _read_csv(summary_csv)
    experiments = [x.strip() for x in args.experiments.split(",") if x.strip()]

    post_dir = reports_dir / "post"
    _ensure_dir(post_dir)

    source_image_path = Path(args.target_image) if args.target_image else None
    if source_image_path is not None and not source_image_path.is_absolute():
        source_image_path = Path.cwd() / source_image_path

    panel_out = post_dir / f"main_compare_{args.image}.png"
    metric_plot_out = post_dir / f"metrics_{args.image}.png"
    metric_table_out = post_dir / f"metrics_{args.image}.csv"
    meta_out = post_dir / f"meta_{args.image}.json"

    build_main_panel(
        results_rows=results_rows,
        image_name=args.image,
        experiments=experiments,
        seed=int(args.seed),
        runs_root=runs_root,
        reports_dir=reports_dir,
        source_image_path=source_image_path,
        out_path=panel_out,
    )
    build_metric_plot(
        summary_rows=summary_rows,
        image_name=args.image,
        experiments=experiments,
        out_path=metric_plot_out,
    )
    build_metric_table(
        summary_rows=summary_rows,
        image_name=args.image,
        experiments=experiments,
        out_path=metric_table_out,
    )

    payload = {
        "image": args.image,
        "seed": int(args.seed),
        "experiments": experiments,
        "outputs": {
            "main_compare": str(panel_out),
            "metrics_plot": str(metric_plot_out),
            "metrics_table": str(metric_table_out),
        },
    }
    with open(meta_out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"[after_train] main compare: {panel_out}")
    print(f"[after_train] metrics plot: {metric_plot_out}")
    print(f"[after_train] metrics table: {metric_table_out}")


if __name__ == "__main__":
    main()
