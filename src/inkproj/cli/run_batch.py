import argparse
from dataclasses import replace
from pathlib import Path

from inkproj.core.config import default_config_path, load_train_config
from inkproj.pipelines import run_train


def _list_images(folder: Path):
    exts = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"]
    files = []
    for ext in exts:
        files.extend(folder.glob(ext))
    return sorted(files)


def main():
    project_root = Path(__file__).resolve().parents[3]
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=str(default_config_path(str(project_root))))
    ap.add_argument("--target_dir", type=str, required=True)
    ap.add_argument("--runs_root", type=str, default="runs")
    ap.add_argument("--seeds", type=str, default="0")
    args = ap.parse_args()

    target_dir = Path(args.target_dir)
    images = _list_images(target_dir)
    if not images:
        raise FileNotFoundError(f"No images found in {target_dir}")
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip() != ""]
    base_cfg = load_train_config(args.config)

    for img in images:
        for seed in seeds:
            out_dir = Path(args.runs_root) / Path(args.config).stem / img.stem / f"seed_{seed}"
            out_dir.mkdir(parents=True, exist_ok=True)
            cfg = replace(base_cfg, target=str(img), out_dir=str(out_dir), seed=int(seed))
            run_train(cfg)


if __name__ == "__main__":
    main()
