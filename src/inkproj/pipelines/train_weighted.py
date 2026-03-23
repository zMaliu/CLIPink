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


def _build_targets(target, clip_preprocess, device):
    preprocess_clip_target = transforms.Compose([
        clip_preprocess.transforms[0],
        clip_preprocess.transforms[1],
        transforms.ToTensor(),
        clip_preprocess.transforms[-1],
    ])
    preprocess_rgb_target = transforms.Compose([
        clip_preprocess.transforms[0],
        clip_preprocess.transforms[1],
        transforms.ToTensor(),
    ])
    target_clip_t = preprocess_clip_target(target).unsqueeze(0).to(device)
    target_rgb_t = preprocess_rgb_target(target).unsqueeze(0).to(device)
    return target_clip_t, target_rgb_t


def run_train(cfg: TrainConfig):
    set_seed(int(cfg.seed))
    device = torch.device("cpu" if cfg.cpu or (not torch.cuda.is_available()) else "cuda")
    ensure_dir(cfg.out_dir)
    model, clip_preprocess = clip.load("ViT-B/32", device=device, jit=False)
    model.eval()
    target = Image.open(cfg.target).convert("RGB")
    target_clip_t, target_rgb_t = _build_targets(target, clip_preprocess, device)
    target_feat = model.encode_image(target_clip_t)
    target_feat = target_feat / target_feat.norm(dim=-1, keepdim=True)
    target_feat = target_feat.detach()
    params_init = sample_params(int(cfg.n_strokes), int(cfg.seed), target_img=target).to(device)
    params_var = params_init.clone().detach().requires_grad_(True)
    gate_logits = torch.full((int(cfg.n_strokes),), -1.5, device=device, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([params_var, gate_logits], lr=float(cfg.lr))
    metrics_log = []

    for it in range(int(cfg.iters) + 1):
        opt.zero_grad(set_to_none=True)
        gates = torch.sigmoid(gate_logits).view(-1, 1, 1, 1)
        canvas = compose_canvas(params_var, gates, cfg.width, cfg.steps, max(1, int(cfg.batch)), render_strokes)
        clip_loss = compute_clip_loss(model, clip_preprocess, canvas, target_feat)
        sparse_loss, ink_loss, l2_loss, edge_loss = compute_regularizers(
            canvas, gates, target_rgb_t, cfg.width, cfg.w_l2, cfg.w_edge
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
            params_var[:, 16:19].clamp_(0.0, 1.0)

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
            save_image(os.path.join(cfg.out_dir, f"iter_{it:04d}.png"), to_u8(canvas))

    with torch.no_grad():
        np.save(os.path.join(cfg.out_dir, "params_final.npy"), params_var.detach().cpu().numpy())
        np.save(os.path.join(cfg.out_dir, "gates_final.npy"), torch.sigmoid(gate_logits).detach().cpu().numpy())
        high_res_w = int(cfg.width) * 2
        gates = torch.sigmoid(gate_logits).view(-1, 1, 1, 1)
        canvas_high = compose_canvas(params_var, gates, high_res_w, int(cfg.steps) * 2, max(1, int(cfg.batch)), render_strokes)
        save_image(os.path.join(cfg.out_dir, "final_highres.png"), to_u8(canvas_high))

    save_json(os.path.join(cfg.out_dir, "config.json"), config_to_dict(cfg))
    save_metrics_csv(os.path.join(cfg.out_dir, "metrics.csv"), metrics_log)
    summary = metrics_log[-1] if len(metrics_log) > 0 else {}
    save_json(os.path.join(cfg.out_dir, "summary.json"), summary)
    return summary
