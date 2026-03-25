# inkproj

CLIP-guided sparse differentiable ink stroke rendering project.

## Installation

```bash
pip install -r requirements.txt
```

## Single-image Training

```bash
python main.py train --config configs/main_weighted.yaml
```

Override target, output directory, seed, iterations, or batch size:

```bash
python main.py train \
  --config configs/main_weighted.yaml \
  --target target_images/camel.png \
  --out_dir runs/main_weighted/camel/seed_0 \
  --seed 0 \
  --iters 650 \
  --batch 1
```

## Batch Training

Train on all supported images under a directory:

```bash
python main.py batch \
  --config configs/main_weighted.yaml \
  --target_dir target_images \
  --runs_root runs/main_weighted_batch \
  --seeds 0,1,2
```

## Outputs

Each run writes:

- `config.json`
- `metrics.csv`
- `summary.json`
- `iter_*.png`
- `final_highres.png`
- `params_final.npy`
- `gates_final.npy`

## Directory Layout

- `src/inkproj/core`: configuration, parameter sampling, losses, metrics, I/O
- `src/inkproj/model`: renderer implementation
- `src/inkproj/pipelines`: training flow
- `src/inkproj/third_party`: embedded third-party CLIP dependency
- `configs`: experiment configurations
- `runs`: saved experiment outputs
- `target_images`: sample images for sanity checks and demos
