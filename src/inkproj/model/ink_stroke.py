import math
import os

import torch
import torch.nn.functional as F


def _bezier_cubic(p0, p1, p2, p3, t):
    u = 1.0 - t
    return (u * u * u) * p0 + 3.0 * (u * u * t) * p1 + 3.0 * (u * t * t) * p2 + (t * t * t) * p3


def _gauss_kernel_1d(device, sigma):
    sigma = float(max(1e-4, float(sigma)))
    k = int(max(3, int(2 * round(1.5 * sigma) + 1)))
    if (k % 2) == 0:
        k += 1
    ax = torch.arange(k, device=device, dtype=torch.float32) - (k - 1) / 2.0
    kk = torch.exp(-(ax * ax) / (2.0 * (sigma * sigma)))
    kk = kk / (kk.sum() + 1e-8)
    return kk, k


def render_strokes(params, width=128, steps=None):
    if not torch.is_tensor(params):
        params = torch.tensor(params, dtype=torch.float32)
    p = params.to(dtype=torch.float32)
    if p.dim() == 1:
        p = p.view(1, -1)
    
    # Check dimensions to support both grayscale (16) and color (19)
    n_dims = int(p.shape[1])
    if n_dims < 16:
        raise ValueError("params must have at least 16 dims")

    x0, y0, x1, y1, x2, y2 = [p[:, i].clamp(0.0, 1.0) for i in range(6)]
    theta = p[:, 6]
    pressure_start = p[:, 7].clamp(0.0, 1.5)
    pressure_end = p[:, 8].clamp(0.0, 1.5)
    brush_width = p[:, 9].clamp(0.001, 1.0)
    aspect = p[:, 10].clamp(0.2, 3.0)
    ink_density = p[:, 11].clamp(0.0, 2.0)
    diffusion_sigma = p[:, 12].clamp(0.0, 10.0)
    pressure_mid1 = p[:, 13].clamp(0.0, 1.5)
    pressure_mid2 = p[:, 14].clamp(0.0, 1.5)
    width_gamma = p[:, 15].clamp(0.2, 3.0)
    
    # Handle color parameters if present (indices 16, 17, 18 for R, G, B)
    if n_dims >= 19:
        color_r = p[:, 16].clamp(0.0, 1.0)
        color_g = p[:, 17].clamp(0.0, 1.0)
        color_b = p[:, 18].clamp(0.0, 1.0)
    else:
        # Default to black ink if no color provided
        n = int(p.shape[0])
        device = p.device
        color_r = torch.zeros(n, device=device)
        color_g = torch.zeros(n, device=device)
        color_b = torch.zeros(n, device=device)

    device = p.device
    n = int(p.shape[0])
    width = int(width)

    render_scale = int(os.environ.get("STROKE_RENDER_SCALE", "2") or 2)
    render_scale = max(1, render_scale)
    render_width = int(width * render_scale)

    x0p = x0 * (render_width - 1)
    y0p = y0 * (render_width - 1)
    x1p = x1 * (render_width - 1)
    y1p = y1 * (render_width - 1)
    x2p = x2 * (render_width - 1)
    y2p = y2 * (render_width - 1)

    stroke_length = torch.sqrt((x2p - x0p) ** 2 + (y2p - y0p) ** 2)
    if steps is None:
        max_steps = int(torch.clamp(stroke_length.max(), 200, 900).item())
        steps = max_steps
    steps = int(max(32, steps))

    t = torch.linspace(0.0, 1.0, steps, device=device, dtype=torch.float32).view(1, steps)
    u = 1.0 - t

    xf = (u * u) * x0p.view(n, 1) + (2.0 * u * t) * x1p.view(n, 1) + (t * t) * x2p.view(n, 1)
    yf = (u * u) * y0p.view(n, 1) + (2.0 * u * t) * y1p.view(n, 1) + (t * t) * y2p.view(n, 1)

    pressure = _bezier_cubic(pressure_start.view(n, 1), pressure_mid1.view(n, 1), pressure_mid2.view(n, 1), pressure_end.view(n, 1), t)
    pressure = pressure.clamp(0.0, 1.5)

    current_width = brush_width.view(n, 1) * (pressure.clamp_min(1e-6) ** width_gamma.view(n, 1)) * render_width * 0.5
    ink_intensity = (0.8 * pressure + 0.2) * ink_density.view(n, 1)
    ink_intensity = ink_intensity.clamp(0.0, 2.0)

    major_axis = (current_width * aspect.view(n, 1)).clamp(1.0, float(render_width))
    minor_axis = (current_width / aspect.view(n, 1).clamp_min(0.2)).clamp(1.0, float(render_width))

    angle_rad = theta.view(n, 1) * math.pi
    cos_angle = torch.cos(angle_rad)
    sin_angle = torch.sin(angle_rad)

    yy, xx = torch.meshgrid(
        torch.arange(render_width, device=device, dtype=torch.float32),
        torch.arange(render_width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    xx = xx.reshape(1, 1, -1)
    yy = yy.reshape(1, 1, -1)

    log_keep = torch.zeros((n, xx.shape[-1]), device=device, dtype=torch.float32)
    step_chunk = int(os.environ.get("STROKE_RENDER_STEP_CHUNK", "64") or 64)
    step_chunk = max(1, step_chunk)

    for s in range(0, steps, step_chunk):
        e = min(steps, s + step_chunk)
        xf_s = xf[:, s:e]
        yf_s = yf[:, s:e]
        pressure_s = pressure[:, s:e]
        major_s = major_axis[:, s:e]
        minor_s = minor_axis[:, s:e]
        ink_s = ink_intensity[:, s:e]

        dx = xx - xf_s.view(n, e - s, 1)
        dy = yy - yf_s.view(n, e - s, 1)
        dxr = cos_angle.view(n, 1, 1) * dx + sin_angle.view(n, 1, 1) * dy
        dyr = -sin_angle.view(n, 1, 1) * dx + cos_angle.view(n, 1, 1) * dy

        q = 0.5 * ((dxr / major_s.view(n, e - s, 1)) ** 2 + (dyr / minor_s.view(n, e - s, 1)) ** 2)
        g = torch.exp(-q)

        noise_freq_x = 0.05
        noise_freq_y = 0.5
        coord_x = dxr * noise_freq_x
        coord_y = dyr * noise_freq_y
        noise = torch.sin(coord_x * 20.0) * torch.cos(coord_y * 20.0)
        noise = (noise + 1.0) * 0.5
        random_offset = torch.sin(xf_s.view(n, e - s, 1) * 0.1) * torch.cos(yf_s.view(n, e - s, 1) * 0.1)
        noise = (noise + random_offset * 0.2).clamp(0.0, 1.0)

        dryness = 1.0 - pressure_s.clamp(0.0, 1.0)
        dryness = dryness.view(n, e - s, 1)
        noise = (noise - 0.5) * 2.0
        noise = torch.sigmoid(noise * 5.0)
        noise_mask = 1.0 - (dryness * (1.0 - noise) * 0.95)

        g = (g * ink_s.view(n, e - s, 1) * noise_mask).clamp(0.0, 0.99)
        log_keep = log_keep + torch.log1p(-g).sum(dim=1)

    A = 1.0 - torch.exp(log_keep)
    A = A.view(n, render_width, render_width).clamp(0.0, 1.0)

    # --- Reduced Diffusion for Sharper Edges ---
    # Only apply minimal anti-aliasing unless explicitly requested
    diffusion_scale = float(os.environ.get("STROKE_RENDER_DIFFUSION_SCALE", "0.1") or 0.1) # Default reduced from 1.0
    diffusion_min = float(os.environ.get("STROKE_RENDER_DIFFUSION_MIN", "0.0") or 0.0)
    diffusion_max = float(os.environ.get("STROKE_RENDER_DIFFUSION_MAX", "10.0") or 10.0)
    diffusion_sigma = (diffusion_sigma * diffusion_scale).clamp(diffusion_min, diffusion_max)

    if float(diffusion_sigma.max().detach().item()) > 1e-6:
        sig = diffusion_sigma
        kk, k = _gauss_kernel_1d(device, float(sig.max().detach().item()))
        A_ = A.unsqueeze(0)
        A_ = F.conv2d(A_, kk.view(1, 1, 1, k).repeat(n, 1, 1, 1), padding=(0, k // 2), groups=n)
        A_ = F.conv2d(A_, kk.view(1, 1, k, 1).repeat(n, 1, 1, 1), padding=(k // 2, 0), groups=n)
        A = A_.squeeze(0).clamp(0.0, 1.0)

    # --- RGBA Output ---
    # Resize alpha mask to target width
    A = A.unsqueeze(1)
    out_alpha = F.interpolate(A, size=(width, width), mode="area").clamp(0.0, 1.0)
    
    # Prepare color channels
    # color_r shape: [n] -> [n, 1, 1, 1] -> [n, 1, w, w]
    c_r = color_r.view(n, 1, 1, 1).expand(n, 1, width, width)
    c_g = color_g.view(n, 1, 1, 1).expand(n, 1, width, width)
    c_b = color_b.view(n, 1, 1, 1).expand(n, 1, width, width)
    
    # Combine RGB with Alpha: Output is [n, 4, w, w] (RGBA)
    # R, G, B channels are premultiplied or just carried along. 
    # For composition, we usually want the color + alpha.
    # Here we return RGBA, where RGB is the ink color.
    out = torch.cat([c_r, c_g, c_b, out_alpha], dim=1)
    
    return out
