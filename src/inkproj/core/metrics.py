def compute_metrics(canvas, gates, tau_active=0.5, tau_white=0.95):
    gates_vec = gates.view(-1)
    active_count = int((gates_vec > float(tau_active)).sum().item())
    active_ratio = float(active_count / max(1, int(gates_vec.numel())))
    ink_mass = float((1.0 - canvas).mean().item())
    white_ratio = float((canvas.mean(dim=1, keepdim=True) > float(tau_white)).float().mean().item())
    return {
        "active_count": active_count,
        "active_ratio": active_ratio,
        "ink_mass": ink_mass,
        "whitespace_ratio": white_ratio,
    }
