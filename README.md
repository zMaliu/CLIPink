# inkproj

CLIP-guided sparse differentiable ink stroke rendering project.

## 安装

```bash
cd new
pip install -e .
```

或使用：

```bash
pip install -r requirements.txt
```

## 单次训练

```bash
ink-train
```

可覆盖参数：

```bash
ink-train --config configs/main_weighted.yaml --target <image_path> --out_dir <runs_dir> --seed 0
```

## 批量实验

```bash
ink-batch --config configs/main_weighted.yaml --target_dir <target_dir> --runs_root runs --seeds 0,1,2
```

## 输出结构

每次运行会在目标目录保存：

- `config.json`
- `metrics.csv`
- `summary.json`
- `iter_*.png`
- `final_highres.png`
- `params_final.npy`
- `gates_final.npy`

## 目录说明

- `src/inkproj/model`: 墨迹渲染器
- `src/inkproj/core`: 参数、损失、合成、指标与IO
- `src/inkproj/pipelines`: 训练流程
- `src/inkproj/third_party/clip`: 本地CLIP最小依赖
- `scripts`: 运行入口
- `configs`: 实验配置
- `analysis`: 分析脚本目录

## 权重说明

默认使用 CLIP `ViT-B/32`。权重查找顺序为：

- 环境变量 `INKPROJ_CLIP_MODEL_PATH` 指向的文件或目录
- 项目目录 `weights/ViT-B-32.pt`
- 项目根目录 `ViT-B-32.pt`
- 用户缓存目录 `~/.cache/clip/ViT-B-32.pt`

若都未命中，代码会自动下载到缓存目录。

`ViT-B-32.pt` 已被 `.gitignore` 规则覆盖，默认不会上传到 GitHub。
