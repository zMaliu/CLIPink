import torch
import torch.nn.functional as F


def sobel_edges(img):
    if img.dim() == 4 and img.shape[1] == 3:
        gray = 0.2989 * img[:, 0:1] + 0.5870 * img[:, 1:2] + 0.1140 * img[:, 2:3]
    else:
        gray = img
    device = gray.device
    dtype = gray.dtype
    kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=device, dtype=dtype).view(1, 1, 3, 3)
    ky = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], device=device, dtype=dtype).view(1, 1, 3, 3)
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-6)


def compute_clip_loss(model, clip_preprocess, canvas, target_feat):
    clip_img = canvas
    if int(clip_img.shape[-1]) != 224 or int(clip_img.shape[-2]) != 224:
        clip_img = F.interpolate(clip_img, size=(224, 224), mode="bilinear", align_corners=False)
    clip_in = clip_preprocess.transforms[-1](clip_img.squeeze(0)).unsqueeze(0)
    feat = model.encode_image(clip_in)
    feat = feat / feat.norm(dim=-1, keepdim=True)
    return 1.0 - (feat * target_feat).sum(dim=-1).mean()


def compute_regularizers(canvas, gates, target_rgb_t, width: int, w_l2: float, w_edge: float):
    sparse_loss = gates.mean()
    ink_loss = (1.0 - canvas).mean()
    t_resized = F.interpolate(target_rgb_t, size=(width, width), mode="bilinear")
    l2_loss = torch.tensor(0.0, device=canvas.device)
    edge_loss = torch.tensor(0.0, device=canvas.device)
    if w_l2 > 0:
        l2_loss = F.mse_loss(canvas, t_resized)
    if w_edge > 0:
        edge_loss = F.mse_loss(sobel_edges(canvas), sobel_edges(t_resized))
    return sparse_loss, ink_loss, l2_loss, edge_loss
