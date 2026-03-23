import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def sample_params(n: int, seed: int, target_img=None):
    rng = random.Random(int(seed))
    n_base = int(n * 0.7)
    p = torch.zeros((n, 19), dtype=torch.float32)
    if target_img is not None:
        img_np = np.array(target_img.convert("L"))
        h, w = img_np.shape
        probs = (255 - img_np).flatten().astype(float)
        probs = probs / (probs.sum() + 1e-8)
        indices = np.random.choice(h * w, size=n, p=probs)
        ys, xs = np.unravel_index(indices, (h, w))
        init_x = xs / float(w)
        init_y = ys / float(h)
        target_rgb = np.array(target_img.convert("RGB"))
    for i in range(n):
        is_detail = i >= n_base
        if target_img is not None:
            x0, y0 = init_x[i], init_y[i]
            x2 = x0 + rng.uniform(-0.1, 0.1)
            y2 = y0 + rng.uniform(-0.1, 0.1)
            py = int(np.clip(y0 * h, 0, h - 1))
            px = int(np.clip(x0 * w, 0, w - 1))
            r, g, b = target_rgb[py, px] / 255.0
            if is_detail:
                r = np.clip(r * 0.6, 0, 1)
                g = np.clip(g * 0.6, 0, 1)
                b = np.clip(b * 0.6, 0, 1)
            else:
                r = np.clip(r * 0.9 + 0.1, 0, 1)
                g = np.clip(g * 0.9 + 0.1, 0, 1)
                b = np.clip(b * 0.9 + 0.1, 0, 1)
            r = np.clip(r + rng.uniform(-0.05, 0.05), 0, 1)
            g = np.clip(g + rng.uniform(-0.05, 0.05), 0, 1)
            b = np.clip(b + rng.uniform(-0.05, 0.05), 0, 1)
        else:
            x0, y0 = rng.random(), rng.random()
            x2, y2 = rng.random(), rng.random()
            r, g, b = rng.random() * 0.5, rng.random() * 0.5, rng.random() * 0.5
        mx = min(1.0, max(0.0, 0.5 * (x0 + x2) + rng.uniform(-0.05, 0.05)))
        my = min(1.0, max(0.0, 0.5 * (y0 + y2) + rng.uniform(-0.05, 0.05)))
        theta = rng.uniform(-1.0, 1.0)
        if is_detail:
            p_start = rng.uniform(1.2, 1.8)
            p_end = rng.uniform(0.8, 1.2)
            bw = rng.uniform(0.01, 0.04)
            aspect = rng.uniform(3.0, 6.0)
            ink = rng.uniform(1.8, 2.5)
            diff = rng.uniform(0.0, 0.02)
        else:
            p_start = rng.uniform(1.0, 1.5)
            p_end = rng.uniform(0.5, 1.0)
            bw = rng.uniform(0.06, 0.15)
            aspect = rng.uniform(1.0, 2.5)
            ink = rng.uniform(0.8, 1.4)
            diff = rng.uniform(0.02, 0.1)
        p_m1 = rng.uniform(0.8, 1.4)
        p_m2 = rng.uniform(0.8, 1.4)
        gamma = rng.uniform(0.8, 1.2)
        p[i] = torch.tensor([x0, y0, mx, my, x2, y2, theta, p_start, p_end, bw, aspect, ink, diff, p_m1, p_m2, gamma, r, g, b])
    return p
