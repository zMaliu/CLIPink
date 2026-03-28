# inkproj

CLIP-guided sparse differentiable ink stroke rendering project.

## Installation

```bash
pip install -r requirements.txt
```

## Entry Points

Single-image training:

```bash
python main.py train --config configs/experiments/full.yaml --target target_images/camel.png --out_dir runs/full/camel/seed_0 --seed 0
```

One experiment on all images and seeds:

```bash
python main.py batch --config configs/experiments/full.yaml --target_dir target_images --runs_root runs --seeds 0,1,2
```

Whole ablation suite in one overnight run:

```bash
python main.py suite --suite configs/suites/ablation_core.yaml --target_dir target_images --runs_root runs --seeds 0,1,2
```

Collect aggregated CSV reports:

```bash
python analysis/collect_results.py --runs_root runs
```

## Config Layout

- `configs/experiments`: one YAML per experiment variant
- `configs/suites`: lists of experiment configs for serial overnight execution

Core experiment variants included:

- `full`
- `no_sparse`
- `no_ink`
- `no_l2`
- `no_edge`
- `no_gate`
- `no_layered_init`
- `softline`
- `strokes_48`
- `strokes_96`

## Output Layout

All suite and batch runs use:

```text
runs/<experiment>/<image>/seed_<k>/
```

Each run writes:

- `config.json`
- `metrics.csv`
- `summary.json`
- `iter_*.png`
- `final_highres.png`
- `params_final.npy`
- `gates_final.npy`

Aggregated reports are written to:

```text
runs/_reports/results.csv
runs/_reports/results_summary.csv
```

## Directory Layout

- `src/inkproj/core`: configuration, parameter sampling, losses, metrics, I/O
- `src/inkproj/model`: renderer implementation
- `src/inkproj/pipelines`: training flow
- `src/inkproj/third_party`: embedded CLIP dependency
- `configs`: experiment and suite definitions
- `analysis`: result aggregation utilities
- `runs`: saved experiment outputs
- `target_images`: sample images for sanity checks and demos
