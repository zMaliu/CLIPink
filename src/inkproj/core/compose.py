import torch


def compose_canvas(params, gates, width, steps, batch_size, render_fn):
    n = int(params.shape[0])
    canvas = torch.ones(1, 3, width, width, device=params.device)
    for s in range(0, n, batch_size):
        e = min(n, s + batch_size)
        strokes = render_fn(params[s:e], width=int(width), steps=int(steps))
        alpha = strokes[:, 3:4].clamp(0.0, 0.999)
        alpha = (alpha * gates[s:e]).clamp(0.0, 0.999)
        rgb = strokes[:, 0:3]
        mix_factor = 1.0 - alpha * (1.0 - rgb)
        for b in range(mix_factor.shape[0]):
            canvas = canvas * mix_factor[b:b + 1]
    return canvas
