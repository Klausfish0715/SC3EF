import torch
import torch.nn.functional as F


def coords_grid(b, h, w, homogeneous=False, device=None):
    """
    Generate a coordinate grid for an image.

    Args:
        b (int): Batch size.
        h (int): Height of the image.
        w (int): Width of the image.
        homogeneous (bool): If True, add a homogeneous coordinate (1) channel.
        device (torch.device, optional): Target device.

    Returns:
        torch.Tensor: Coordinate grid of shape [B, 2, H, W] or [B, 3, H, W].
    """
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')  # [H, W]

    stacks = [x, y]
    if homogeneous:
        ones = torch.ones_like(x)
        stacks.append(ones)

    grid = torch.stack(stacks, dim=0).float()  # [2, H, W] or [3, H, W]
    grid = grid[None].repeat(b, 1, 1, 1)        # [B, 2, H, W] or [B, 3, H, W]

    if device is not None:
        grid = grid.to(device)

    return grid


def generate_window_grid(h_min, h_max, w_min, w_max, len_h, len_w, device=None):
    """
    Generate a local window coordinate grid.

    Args:
        h_min (float): Minimum height value.
        h_max (float): Maximum height value.
        w_min (float): Minimum width value.
        w_max (float): Maximum width value.
        len_h (int): Number of steps along the height.
        len_w (int): Number of steps along the width.
        device (torch.device): Device on which the grid is created.

    Returns:
        torch.Tensor: Window grid of shape [H, W, 2].
    """
    assert device is not None

    x, y = torch.meshgrid(
        torch.linspace(w_min, w_max, len_w, device=device),
        torch.linspace(h_min, h_max, len_h, device=device),
        indexing='ij'
    )
    grid = torch.stack((x, y), dim=-1).transpose(0, 1).float()  # [H, W, 2]

    return grid


def normalize_coords(coords, h, w):
    """
    Normalize coordinates from pixel space to [-1, 1] range.

    Args:
        coords (torch.Tensor): Coordinate tensor [B, H, W, 2].
        h (int): Image height.
        w (int): Image width.

    Returns:
        torch.Tensor: Normalized coordinates [B, H, W, 2].
    """
    c = torch.tensor([(w - 1) / 2., (h - 1) / 2.], device=coords.device)
    return (coords - c) / c


def bilinear_sample(img, sample_coords, mode='bilinear', padding_mode='zeros', return_mask=False):
    """
    Perform bilinear sampling on the image using sampling coordinates.

    Args:
        img (torch.Tensor): Input image [B, C, H, W].
        sample_coords (torch.Tensor): Sampling coordinates [B, 2, H, W] or [B, H, W, 2].
        mode (str): Sampling mode ('bilinear' or 'nearest').
        padding_mode (str): Padding mode for out-of-boundary samples.
        return_mask (bool): Whether to return valid mask.

    Returns:
        torch.Tensor: Sampled image.
        (optional) torch.Tensor: Valid mask indicating in-bound samples.
    """
    if sample_coords.size(1) != 2:
        sample_coords = sample_coords.permute(0, 3, 1, 2)

    b, _, h, w = sample_coords.shape

    # Normalize coordinates to [-1, 1]
    x_grid = 2 * sample_coords[:, 0] / (w - 1) - 1
    y_grid = 2 * sample_coords[:, 1] / (h - 1) - 1
    grid = torch.stack([x_grid, y_grid], dim=-1)  # [B, H, W, 2]

    img = F.grid_sample(img, grid, mode=mode, padding_mode=padding_mode, align_corners=True)

    if return_mask:
        mask = (x_grid >= -1) & (x_grid <= 1) & (y_grid >= -1) & (y_grid <= 1)  # [B, H, W]
        return img, mask

    return img


def flow_warp(feature, flow, mask=False, padding_mode='zeros'):
    """
    Warp a feature map according to the given optical flow.

    Args:
        feature (torch.Tensor): Feature map [B, C, H, W].
        flow (torch.Tensor): Optical flow [B, 2, H, W].
        mask (bool): Whether to return valid mask.
        padding_mode (str): Padding mode for out-of-boundary samples.

    Returns:
        torch.Tensor: Warped feature map.
    """
    b, c, h, w = feature.size()
    assert flow.size(1) == 2

    grid = coords_grid(b, h, w, device=flow.device) + flow  # [B, 2, H, W]

    return bilinear_sample(feature, grid, padding_mode=padding_mode, return_mask=mask)


def forward_backward_consistency_check(fwd_flow, bwd_flow, alpha=0.01, beta=0.5):
    """
    Perform forward-backward consistency check between flows.

    Args:
        fwd_flow (torch.Tensor): Forward flow [B, 2, H, W].
        bwd_flow (torch.Tensor): Backward flow [B, 2, H, W].
        alpha (float): Alpha term for thresholding.
        beta (float): Beta term for thresholding.

    Returns:
        tuple: (forward occlusion map, backward occlusion map), both [B, H, W].
    """
    assert fwd_flow.dim() == 4 and bwd_flow.dim() == 4
    assert fwd_flow.size(1) == 2 and bwd_flow.size(1) == 2

    # Compute flow magnitudes
    flow_mag = torch.norm(fwd_flow, dim=1) + torch.norm(bwd_flow, dim=1)  # [B, H, W]

    # Warp flows
    warped_bwd_flow = flow_warp(bwd_flow, fwd_flow)
    warped_fwd_flow = flow_warp(fwd_flow, bwd_flow)

    # Compute consistency errors
    diff_fwd = torch.norm(fwd_flow + warped_bwd_flow, dim=1)
    diff_bwd = torch.norm(bwd_flow + warped_fwd_flow, dim=1)

    # Consistency threshold
    threshold = alpha * flow_mag + beta

    # Identify occlusions
    fwd_occ = (diff_fwd > threshold).float()
    bwd_occ = (diff_bwd > threshold).float()

    return fwd_occ, bwd_occ
