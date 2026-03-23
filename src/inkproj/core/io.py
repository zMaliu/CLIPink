import csv
import json
import os

import numpy as np
from PIL import Image


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def to_u8(img):
    x = img.detach().to("cpu").clamp(0.0, 1.0)
    if x.dim() == 4:
        x = x[0]
    if int(x.shape[0]) == 3:
        x = x.permute(1, 2, 0)
    return (x.numpy() * 255.0).astype(np.uint8)


def save_image(path: str, arr):
    ensure_dir(os.path.dirname(path))
    Image.fromarray(arr).save(path)


def save_json(path: str, payload):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def save_metrics_csv(path: str, rows):
    ensure_dir(os.path.dirname(path))
    if len(rows) == 0:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    keys = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
