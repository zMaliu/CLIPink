import argparse
import sys
from dataclasses import replace
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from inkproj.core.config import default_config_path, load_train_config


def _list_images(folder: Path):
    exts = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"]
    files = []
    for ext in exts:
        files.extend(folder.glob(ext))
    return sorted(files)


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
    if argv and argv[0] not in {"train", "batch", "render", "-h", "--help"}:
        argv = ["train"] + argv
    elif not argv:
        parser.print_help()
        return

    args = parser.parse_args(argv)
    if args.command == "batch":
        _run_batch(args)
        return
    if args.command == "render":
        _run_render(args)
        return
    _run_single(args)


if __name__ == "__main__":
    main()
