import torch
import torch.nn.functional as F

from .geometry import coords_grid, generate_window_grid, normalize_coords


def global_correlation_softmax(feature0, feature1):
    """
    Compute the global softmax correlation between two feature maps.

    Args:
        feature0 (torch.Tensor): Feature map 1 of shape [B, C, H, W].
        feature1 (torch.Tensor): Feature map 2 of shape [B, C, H, W].

    Returns:
        flow (torch.Tensor): Predicted global flow [B, 2, H, W].
        prob (torch.Tensor): Matching probability [B, H*W, H*W].
    """
    b, c, h, w = feature0.shape

    # Flatten and permute features for matrix multiplication
    feature0 = feature0.view(b, c, -1).permute(0, 2, 1)  # [B, H*W, C]
    feature1 = feature1.view(b, c, -1)                   # [B, C, H*W]

    # Compute global correlation
    correlation = torch.matmul(feature0, feature1).view(b, h, w, h, w) / (c ** 0.5)  # [B, H, W, H, W]

    # Initialize coordinate grid
    init_grid = coords_grid(b, h, w).to(correlation.device)  # [B, 2, H, W]
    grid = init_grid.view(b, 2, -1).permute(0, 2, 1)          # [B, H*W, 2]
    correlation = correlation.view(b, h * w, h * w)           # [B, H*W, H*W]

    # Softmax over correlation
    prob = F.softmax(correlation, dim=-1)                    # [B, H*W, H*W]

    # Compute correspondences via weighted sum of grid locations
    correspondence = torch.matmul(prob, grid).view(b, h, w, 2).permute(0, 3, 1, 2)  # [B, 2, H, W]

    # Compute flow as difference between correspondences and original grid
    flow = correspondence.contiguous() - init_grid.contiguous()

    return flow, prob


def local_correlation_softmax(feature0, feature1, local_radius, padding_mode='zeros'):
    """
    Compute the local softmax correlation between two feature maps within a window.

    Args:
        feature0 (torch.Tensor): Feature map 1 [B, C, H, W].
        feature1 (torch.Tensor): Feature map 2 [B, C, H, W].
        local_radius (int): Radius of the local window.
        padding_mode (str): Padding mode for grid sampling (default: 'zeros').

    Returns:
        flow (torch.Tensor): Predicted local flow [B, 2, H, W].
        match_prob (torch.Tensor): Matching probability within local window [B, H*W, (2R+1)^2].
    """
    b, c, h, w = feature0.size()

    # Initialize coordinate grid
    coords_init = coords_grid(b, h, w).to(feature0.device)  # [B, 2, H, W]
    coords = coords_init.view(b, 2, -1).permute(0, 2, 1)     # [B, H*W, 2]

    local_h = 2 * local_radius + 1
    local_w = 2 * local_radius + 1

    # Generate window grid for sampling
    window_grid = generate_window_grid(
        -local_radius, local_radius,
        -local_radius, local_radius,
        local_h, local_w,
        device=feature0.device
    )  # [2R+1, 2R+1, 2]
    window_grid = window_grid.reshape(-1, 2).repeat(b, 1, 1, 1)  # [B, 1, (2R+1)^2, 2]

    # Sample coordinates in the local window
    sample_coords = coords.unsqueeze(-2) + window_grid  # [B, H*W, (2R+1)^2, 2]

    sample_coords_softmax = sample_coords

    # Mask out invalid coordinates that are outside the image
    valid_x = (sample_coords[:, :, :, 0] >= 0) & (sample_coords[:, :, :, 0] < w)  # [B, H*W, (2R+1)^2]
    valid_y = (sample_coords[:, :, :, 1] >= 0) & (sample_coords[:, :, :, 1] < h)  # [B, H*W, (2R+1)^2]
    valid = valid_x & valid_y  # [B, H*W, (2R+1)^2]

    # Normalize coordinates to [-1, 1] for grid_sample
    sample_coords_norm = normalize_coords(sample_coords, h, w)

    # Sample local features from feature1
    window_feature = F.grid_sample(
        feature1,
        sample_coords_norm,
        padding_mode=padding_mode,
        align_corners=True
    ).permute(0, 2, 1, 3)  # [B, H*W, C, (2R+1)^2]

    # Reshape feature0 for correlation computation
    feature0_view = feature0.permute(0, 2, 3, 1).view(b, h * w, 1, c)  # [B, H*W, 1, C]

    # Compute local correlation
    corr = torch.matmul(feature0_view, window_feature).view(b, h * w, -1) / (c ** 0.5)  # [B, H*W, (2R+1)^2]

    # Mask invalid locations
    corr[~valid] = -1e9

    # Softmax over local window
    prob = F.softmax(corr, dim=-1)  # [B, H*W, (2R+1)^2]

    # Compute correspondences via weighted sum
    correspondence = torch.matmul(
        prob.unsqueeze(-2), sample_coords_softmax
    ).squeeze(-2).view(b, h, w, 2).permute(0, 3, 1, 2)  # [B, 2, H, W]

    # Compute local flow
    flow = correspondence - coords_init
    match_prob = prob

    return flow, match_prob
