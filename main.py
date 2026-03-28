import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from inkproj.core.config import default_config_path, load_train_config


def _list_images(path: Path):
    if path.is_file():
        return [path]
    exts = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"]
    files = []
    for ext in exts:
        files.extend(path.glob(ext))
    return sorted(files)


def _load_yaml_or_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        if path.suffix.lower() == ".json":
            return json.load(f)
        return yaml.safe_load(f)


def _normalize_suite_experiments(payload):
    if payload is None:
        return []
    if isinstance(payload, list):
        items = payload
    else:
        items = payload.get("experiments", [])
    normalized = []
    for item in items:
        if isinstance(item, str):
            normalized.append({"name": Path(item).stem, "config": item})
        elif isinstance(item, dict):
            cfg_path = item.get("config")
            if not cfg_path:
                raise ValueError("Each suite experiment entry must define 'config'.")
            normalized.append({"name": item.get("name", Path(cfg_path).stem), "config": cfg_path})
        else:
            raise TypeError("Suite experiments must be strings or objects.")
    return normalized


def _run_is_complete(run_dir: Path) -> bool:
    return (run_dir / "summary.json").exists() and (run_dir / "final_highres.png").exists()


def _build_parser():
    parser = argparse.ArgumentParser(
        description="inkproj unified entrypoint for single-image and directory batch training."
    )
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train", help="Train on a single target image.")
    train_parser.add_argument("--config", type=str, default=default_config_path(str(PROJECT_ROOT)))
    train_parser.add_argument("--target", type=str, default=None)
    train_parser.add_argument("--out_dir", type=str, default=None)
    train_parser.add_argument("--seed", type=int, default=None)
    train_parser.add_argument("--iters", type=int, default=None)
    train_parser.add_argument("--batch", type=int, default=None)
    train_parser.add_argument("--lr", type=float, default=None)
    train_parser.add_argument("--n_strokes", type=int, default=None)
    train_parser.add_argument("--steps", type=int, default=None)
    train_parser.add_argument("--width", type=int, default=None)
    train_parser.add_argument("--w_clip", type=float, default=None)
    train_parser.add_argument("--w_sparse", type=float, default=None)
    train_parser.add_argument("--w_ink", type=float, default=None)
    train_parser.add_argument("--w_l2", type=float, default=None)
    train_parser.add_argument("--w_edge", type=float, default=None)
    train_parser.add_argument("--save_every", type=int, default=None)
    train_parser.add_argument("--render_scale", type=int, default=None)
    train_parser.add_argument("--render_step_chunk", type=int, default=None)
    train_parser.add_argument("--render_diffusion_scale", type=float, default=None)
    train_parser.add_argument("--enable_highres", type=int, choices=[0, 1], default=None)
    train_parser.add_argument("--highres_batch", type=int, default=None)
    train_parser.add_argument("--cpu", action="store_true")

    batch_parser = subparsers.add_parser("batch", help="Train on all images under a target directory.")
    batch_parser.add_argument("--config", type=str, default=default_config_path(str(PROJECT_ROOT)))
    batch_parser.add_argument("--target_dir", type=str, required=True)
    batch_parser.add_argument("--runs_root", type=str, default="runs")
    batch_parser.add_argument("--seeds", type=str, default="0")
    batch_parser.add_argument("--iters", type=int, default=None)
    batch_parser.add_argument("--batch", type=int, default=None)
    batch_parser.add_argument("--lr", type=float, default=None)
    batch_parser.add_argument("--n_strokes", type=int, default=None)
    batch_parser.add_argument("--steps", type=int, default=None)
    batch_parser.add_argument("--width", type=int, default=None)
    batch_parser.add_argument("--w_clip", type=float, default=None)
    batch_parser.add_argument("--w_sparse", type=float, default=None)
    batch_parser.add_argument("--w_ink", type=float, default=None)
    batch_parser.add_argument("--w_l2", type=float, default=None)
    batch_parser.add_argument("--w_edge", type=float, default=None)
    batch_parser.add_argument("--save_every", type=int, default=None)
    batch_parser.add_argument("--render_scale", type=int, default=None)
    batch_parser.add_argument("--render_step_chunk", type=int, default=None)
    batch_parser.add_argument("--render_diffusion_scale", type=float, default=None)
    batch_parser.add_argument("--enable_highres", type=int, choices=[0, 1], default=None)
    batch_parser.add_argument("--highres_batch", type=int, default=None)
    batch_parser.add_argument("--cpu", action="store_true")

    suite_parser = subparsers.add_parser("suite", help="Run a named experiment suite serially.")
    suite_parser.add_argument("--suite", type=str, required=True)
    suite_parser.add_argument("--target_dir", type=str, required=True)
    suite_parser.add_argument("--runs_root", type=str, default="runs")
    suite_parser.add_argument("--seeds", type=str, default="0")
    suite_parser.add_argument("--force", action="store_true")
    suite_parser.add_argument("--cpu", action="store_true")

    render_parser = subparsers.add_parser("render", help="Render final high-resolution image from a completed run.")
    render_parser.add_argument("--run_dir", type=str, required=True)
    render_parser.add_argument("--config", type=str, default=None)
    render_parser.add_argument("--highres_batch", type=int, default=None)
    render_parser.add_argument("--cpu", action="store_true")

    return parser


def _run_single(args):
    from inkproj.pipelines import run_train

    cfg = load_train_config(
        args.config,
        target=args.target,
        out_dir=args.out_dir,
        seed=args.seed,
        iters=args.iters,
        batch=args.batch,
        lr=args.lr,
        n_strokes=args.n_strokes,
        steps=args.steps,
        width=args.width,
        w_clip=args.w_clip,
        w_sparse=args.w_sparse,
        w_ink=args.w_ink,
        w_l2=args.w_l2,
        w_edge=args.w_edge,
        save_every=args.save_every,
        render_scale=args.render_scale,
        render_step_chunk=args.render_step_chunk,
        render_diffusion_scale=args.render_diffusion_scale,
        enable_highres=bool(args.enable_highres) if args.enable_highres is not None else None,
        highres_batch=args.highres_batch,
        cpu=True if args.cpu else None,
    )
    summary = run_train(cfg)
    print(summary)


def _run_batch(args):
    from inkproj.pipelines import run_train

    target_dir = Path(args.target_dir)
    images = _list_images(target_dir)
    if not images:
        raise FileNotFoundError(f"No images found in {target_dir}")
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip() != ""]
    base_cfg = load_train_config(
        args.config,
        iters=args.iters,
        batch=args.batch,
        lr=args.lr,
        n_strokes=args.n_strokes,
        steps=args.steps,
        width=args.width,
        w_clip=args.w_clip,
        w_sparse=args.w_sparse,
        w_ink=args.w_ink,
        w_l2=args.w_l2,
        w_edge=args.w_edge,
        save_every=args.save_every,
        render_scale=args.render_scale,
        render_step_chunk=args.render_step_chunk,
        render_diffusion_scale=args.render_diffusion_scale,
        enable_highres=bool(args.enable_highres) if args.enable_highres is not None else None,
        highres_batch=args.highres_batch,
        cpu=True if args.cpu else None,
    )

    for img in images:
        for seed in seeds:
            out_dir = Path(args.runs_root) / Path(args.config).stem / img.stem / f"seed_{seed}"
            out_dir.mkdir(parents=True, exist_ok=True)
            cfg = replace(base_cfg, target=str(img), out_dir=str(out_dir), seed=int(seed))
            summary = run_train(cfg)
            print(f"[{img.name}][seed={seed}] {summary}")


def _run_suite(args):
    from inkproj.pipelines import run_train

    suite_path = Path(args.suite)
    if not suite_path.is_absolute():
        suite_path = PROJECT_ROOT / suite_path
    suite_payload = _load_yaml_or_json(suite_path) or {}
    experiments = _normalize_suite_experiments(suite_payload)
    if not experiments:
        raise ValueError(f"No experiments defined in suite: {suite_path}")

    target_dir = Path(args.target_dir)
    if not target_dir.is_absolute():
        target_dir = PROJECT_ROOT / target_dir
    images = _list_images(target_dir)
    if not images:
        raise FileNotFoundError(f"No images found in {target_dir}")
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip() != ""]

    suite_name = suite_path.stem
    runs_root = Path(args.runs_root)
    if not runs_root.is_absolute():
        runs_root = PROJECT_ROOT / runs_root
    suite_report_dir = runs_root / "_reports"
    suite_report_dir.mkdir(parents=True, exist_ok=True)
    suite_summary = {
        "suite": suite_name,
        "suite_path": str(suite_path),
        "runs_root": str(runs_root),
        "images": [img.name for img in images],
        "seeds": seeds,
        "tasks": [],
    }

    total_tasks = len(experiments) * len(images) * len(seeds)
    task_index = 0
    for exp in experiments:
        config_path = Path(exp["config"])
        if not config_path.is_absolute():
            config_path = PROJECT_ROOT / config_path
        exp_name = exp["name"]
        for img in images:
            for seed in seeds:
                task_index += 1
                out_dir = runs_root / exp_name / img.stem / f"seed_{seed}"
                out_dir.mkdir(parents=True, exist_ok=True)
                task_meta = {
                    "experiment": exp_name,
                    "config": str(config_path),
                    "image": img.name,
                    "seed": seed,
                    "out_dir": str(out_dir),
                    "status": "pending",
                }
                if (not args.force) and _run_is_complete(out_dir):
                    task_meta["status"] = "skipped"
                    print(f"[suite {task_index}/{total_tasks}] skip exp={exp_name} image={img.name} seed={seed}", flush=True)
                    suite_summary["tasks"].append(task_meta)
                    continue
                print(f"[suite {task_index}/{total_tasks}] start exp={exp_name} image={img.name} seed={seed}", flush=True)
                try:
                    cfg = load_train_config(
                        str(config_path),
                        target=str(img),
                        out_dir=str(out_dir),
                        seed=int(seed),
                        cpu=True if args.cpu else None,
                    )
                    summary = run_train(cfg)
                    task_meta["status"] = "completed"
                    task_meta["summary"] = summary
                    print(f"[suite {task_index}/{total_tasks}] done exp={exp_name} image={img.name} seed={seed}", flush=True)
                except Exception as exc:
                    task_meta["status"] = "failed"
                    task_meta["error"] = repr(exc)
                    print(
                        f"[suite {task_index}/{total_tasks}] failed exp={exp_name} image={img.name} seed={seed} error={exc}",
                        flush=True,
                    )
                suite_summary["tasks"].append(task_meta)

    report_path = suite_report_dir / f"{suite_name}_summary.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(suite_summary, f, indent=2, ensure_ascii=False)
    print(f"[suite] summary saved to {report_path}", flush=True)


def _run_render(args):
    from inkproj.pipelines import run_render_final

    run_dir = Path(args.run_dir)
    config_path = args.config if args.config is not None else str(run_dir / "config.json")
    cfg = load_train_config(
        config_path,
        highres_batch=args.highres_batch,
        cpu=True if args.cpu else None,
    )
    summary = run_render_final(cfg, str(run_dir))
    print(summary)


def main():
    parser = _build_parser()
    argv = sys.argv[1:]

    # Backward-friendly behavior: if no subcommand is given, default to single-image training.
    if argv and argv[0] not in {"train", "batch", "suite", "render", "-h", "--help"}:
        argv = ["train"] + argv
    elif not argv:
        parser.print_help()
        return

    args = parser.parse_args(argv)
    if args.command == "batch":
        _run_batch(args)
        return
    if args.command == "suite":
        _run_suite(args)
        return
    if args.command == "render":
        _run_render(args)
        return
    _run_single(args)


if __name__ == "__main__":
    main()
