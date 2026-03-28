import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from inkproj.core.compose import compose_canvas
from inkproj.core.config import TrainConfig, config_to_dict
from inkproj.core.io import ensure_dir, save_image, save_json, save_metrics_csv, to_u8
from inkproj.core.losses import compute_clip_loss, compute_regularizers
from inkproj.core.metrics import compute_metrics
from inkproj.core.params import sample_params, set_seed
from inkproj.model import render_strokes
from inkproj.third_party.clip import clip


def _build_target(target, clip_preprocess, device):
    preprocess_target = transforms.Compose([
        clip_preprocess.transforms[0],
        clip_preprocess.transforms[1],
        transforms.ToTensor(),
        clip_preprocess.transforms[-1],
    ])
    return preprocess_target(target).unsqueeze(0).to(device)


def run_train(cfg: TrainConfig):
    set_seed(int(cfg.seed))
    device = torch.device("cpu" if cfg.cpu or (not torch.cuda.is_available()) else "cuda")
    ensure_dir(cfg.out_dir)
    os.environ["STROKE_RENDER_PROFILE"] = str(cfg.render_profile)
    os.environ["STROKE_RENDER_SCALE"] = str(int(cfg.render_scale))
    os.environ["STROKE_RENDER_STEP_CHUNK"] = str(int(cfg.render_step_chunk))
    os.environ["STROKE_RENDER_DIFFUSION_SCALE"] = str(float(cfg.render_diffusion_scale))
    os.environ["STROKE_RENDER_DIFFUSION_MIN"] = str(float(cfg.render_diffusion_min))
    os.environ["STROKE_RENDER_DIFFUSION_MAX"] = str(float(cfg.render_diffusion_max))
    print(
        f"[train] device={device} target={cfg.target} out_dir={cfg.out_dir} "
        f"n_strokes={cfg.n_strokes} width={cfg.width} steps={cfg.steps} iters={cfg.iters} "
        f"batch={cfg.batch} render_scale={cfg.render_scale} step_chunk={cfg.render_step_chunk} "
        f"diffusion={cfg.render_diffusion_scale} profile={cfg.render_profile} "
        f"gate={cfg.enable_gate} layered_init={cfg.layered_init} highres={cfg.enable_highres}",
        flush=True,
    )
    model, clip_preprocess = clip.load("ViT-B/32", device=device, jit=False)
    model.eval()
    target = Image.open(cfg.target).convert("RGB")
    target_t = _build_target(target, clip_preprocess, device)
    target_t.requires_grad_(True)
    target_feat = model.encode_image(target_t)
    target_feat = target_feat / target_feat.norm(dim=-1, keepdim=True)
    target_feat = target_feat.detach()
    params_init = sample_params(
        int(cfg.n_strokes),
        int(cfg.seed),
        target_img=target,
        layered_init=bool(cfg.layered_init),
    ).to(device)
    params_clamped = params_init.clamp(1e-4, 1 - 1e-4)
    z_params = torch.log(params_clamped / (1 - params_clamped))
    z_params = z_params.clone().detach().requires_grad_(True)
    gate_logits = torch.full((int(cfg.n_strokes),), -1.5, device=device, dtype=torch.float32)
    if bool(cfg.enable_gate):
        gate_logits = gate_logits.requires_grad_(True)
        opt = torch.optim.Adam([z_params, gate_logits], lr=float(cfg.lr))
    else:
        opt = torch.optim.Adam([z_params], lr=float(cfg.lr))
    metrics_log = []

    for it in range(int(cfg.iters) + 1):
        opt.zero_grad(set_to_none=True)
        params = torch.sigmoid(z_params)
        if bool(cfg.enable_gate):
            gates = torch.sigmoid(gate_logits).view(-1, 1, 1, 1)
        else:
            gates = torch.ones((int(cfg.n_strokes), 1, 1, 1), device=device, dtype=torch.float32)
        canvas = compose_canvas(params, gates, cfg.width, cfg.steps, max(1, int(cfg.batch)), render_strokes)
        clip_loss = compute_clip_loss(model, clip_preprocess, canvas, target_feat)
        sparse_loss, ink_loss, l2_loss, edge_loss = compute_regularizers(
            canvas, gates, target_t, cfg.width, cfg.w_l2, cfg.w_edge
        )
        loss = (
            float(cfg.w_clip) * clip_loss
            + float(cfg.w_sparse) * sparse_loss
            + float(cfg.w_ink) * ink_loss
            + float(cfg.w_l2) * l2_loss
            + float(cfg.w_edge) * edge_loss
        )
        loss.backward()
        opt.step()
        with torch.no_grad():
            z_params[:, 16:19].clamp_(-10.0, 10.0)

        row = {
            "iter": int(it),
            "loss": float(loss.item()),
            "clip_loss": float(clip_loss.item()),
            "sparse_loss": float(sparse_loss.item()),
            "ink_loss": float(ink_loss.item()),
            "l2_loss": float(l2_loss.item()),
            "edge_loss": float(edge_loss.item()),
        }
        row.update(compute_metrics(canvas, gates, tau_active=cfg.tau_active, tau_white=cfg.tau_white))
        metrics_log.append(row)

        if it % int(cfg.save_every) == 0 or it == int(cfg.iters):
            print(
                f"[iter {it:04d}] loss={row['loss']:.4f} clip={row['clip_loss']:.4f} "
                f"l2={row['l2_loss']:.4f} edge={row['edge_loss']:.4f} "
                f"sparse={row['sparse_loss']:.4f} ink={row['ink_loss']:.4f} "
                f"active={row['active_ratio']:.4f} white={row['whitespace_ratio']:.4f}",
                flush=True,
            )
            save_image(os.path.join(cfg.out_dir, f"iter_{it:04d}.png"), to_u8(canvas))

    with torch.no_grad():
        params = torch.sigmoid(z_params)
        np.save(os.path.join(cfg.out_dir, "params_final.npy"), params.detach().cpu().numpy())
        final_gates = torch.sigmoid(gate_logits) if bool(cfg.enable_gate) else torch.ones(int(cfg.n_strokes), device=device)
        np.save(os.path.join(cfg.out_dir, "gates_final.npy"), final_gates.detach().cpu().numpy())
        if bool(cfg.enable_highres):
            print("Rendering high-resolution final image on GPU (Parallel)...", flush=True)
            prev_render_scale = os.environ.get("STROKE_RENDER_SCALE")
            os.environ["STROKE_RENDER_SCALE"] = str(int(cfg.highres_render_scale))
            high_res_w = int(cfg.width) * 2
            high_res_steps = max(1, int(float(cfg.steps) * float(cfg.highres_steps_scale)))
            canvas_high = torch.ones(1, 3, high_res_w, high_res_w, device=device)
            n = int(params.shape[0])
            bs = max(1, int(cfg.highres_batch))
            gates_final = final_gates
            for s in range(0, n, bs):
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                e = min(n, s + bs)
                strokes = render_strokes(params[s:e], width=high_res_w, steps=high_res_steps)
                alpha = strokes[:, 3:4].clamp(0.0, 0.999)
                rgb = strokes[:, 0:3]
                gates_view = gates_final[s:e].view(-1, 1, 1, 1)
                alpha = (alpha * gates_view).clamp(0.0, 0.999)
                mix_factor = (1.0 - alpha * (1.0 - rgb))
                for i in range(mix_factor.shape[0]):
                    canvas_high = canvas_high * mix_factor[i:i + 1]
            save_image(os.path.join(cfg.out_dir, "final_highres.png"), to_u8(canvas_high))
            print(f"High-res image saved to {os.path.join(cfg.out_dir, 'final_highres.png')}", flush=True)
            if prev_render_scale is None:
                os.environ.pop("STROKE_RENDER_SCALE", None)
            else:
                os.environ["STROKE_RENDER_SCALE"] = prev_render_scale
        else:
            print("[highres] skipped", flush=True)

    save_json(os.path.join(cfg.out_dir, "config.json"), config_to_dict(cfg))
    save_metrics_csv(os.path.join(cfg.out_dir, "metrics.csv"), metrics_log)
    summary = metrics_log[-1] if len(metrics_log) > 0 else {}
    save_json(os.path.join(cfg.out_dir, "summary.json"), summary)
    return summary


def run_render_final(cfg: TrainConfig, run_dir: str):
    device = torch.device("cpu" if cfg.cpu or (not torch.cuda.is_available()) else "cuda")
    params_path = os.path.join(run_dir, "params_final.npy")
    gates_path = os.path.join(run_dir, "gates_final.npy")
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"Missing params file: {params_path}")
    if not os.path.exists(gates_path):
        raise FileNotFoundError(f"Missing gates file: {gates_path}")

    params = torch.from_numpy(np.load(params_path)).to(device=device, dtype=torch.float32)
    gates = torch.from_numpy(np.load(gates_path)).to(device=device, dtype=torch.float32).view(-1, 1, 1, 1)

    os.environ["STROKE_RENDER_PROFILE"] = str(cfg.render_profile)
    os.environ["STROKE_RENDER_SCALE"] = str(int(cfg.highres_render_scale))
    os.environ["STROKE_RENDER_STEP_CHUNK"] = str(int(cfg.render_step_chunk))
    os.environ["STROKE_RENDER_DIFFUSION_SCALE"] = str(float(cfg.render_diffusion_scale))
    os.environ["STROKE_RENDER_DIFFUSION_MIN"] = str(float(cfg.render_diffusion_min))
    os.environ["STROKE_RENDER_DIFFUSION_MAX"] = str(float(cfg.render_diffusion_max))

    high_res_w = int(cfg.width) * 2
    high_res_steps = max(1, int(float(cfg.steps) * float(cfg.highres_steps_scale)))
    print(
        f"[render] device={device} run_dir={run_dir} width={high_res_w} "
        f"steps={high_res_steps} batch={cfg.highres_batch} render_scale={cfg.highres_render_scale}",
        flush=True,
    )
    with torch.no_grad():
        canvas_high = torch.ones(1, 3, high_res_w, high_res_w, device=device)
        n = int(params.shape[0])
        bs = max(1, int(cfg.highres_batch))
        gates_final = gates.view(-1)
        for s in range(0, n, bs):
            if device.type == "cuda":
                torch.cuda.empty_cache()
            e = min(n, s + bs)
            strokes = render_strokes(params[s:e], width=high_res_w, steps=high_res_steps)
            alpha = strokes[:, 3:4].clamp(0.0, 0.999)
            rgb = strokes[:, 0:3]
            gates_view = gates_final[s:e].view(-1, 1, 1, 1)
            alpha = (alpha * gates_view).clamp(0.0, 0.999)
            mix_factor = (1.0 - alpha * (1.0 - rgb))
            for i in range(mix_factor.shape[0]):
                canvas_high = canvas_high * mix_factor[i:i + 1]
    out_path = os.path.join(run_dir, "final_highres.png")
    save_image(out_path, to_u8(canvas_high))
    print(f"[render] saved {out_path}", flush=True)
    return {"out_path": out_path}
