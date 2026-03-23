import argparse
import os

from inkproj.core.config import default_config_path, load_train_config
from inkproj.pipelines import run_train


def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=default_config_path(project_root))
    ap.add_argument("--target", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    cfg = load_train_config(
        args.config,
        target=args.target,
        out_dir=args.out_dir,
        seed=args.seed,
        iters=args.iters,
        batch=args.batch,
        cpu=True if args.cpu else None,
    )
    summary = run_train(cfg)
    print(summary)


if __name__ == "__main__":
    main()
