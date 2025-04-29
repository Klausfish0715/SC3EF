from os import path as osp
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from utils.flow_viz import save_vis_flow_tofile


def save_flow(args, net, val_loader):
    """
    Save the predicted optical flow and the warped RGB images.
    
    Args:
        args: arguments including model name and metric path.
        net: trained network model.
        val_loader: validation dataloader.
    """
    for _, mini_batch in enumerate(val_loader):
        im_name = mini_batch['im_name']
        input_rgb = mini_batch['input_rgb'][-1]
        input_tir = mini_batch['input_tir'][-1]

        # Load low- and high-frequency features for RGB and TIR modalities
        source_rgb_lf = mini_batch['source_rgb_lf'].cuda()
        source_tir_lf = mini_batch['source_tir_lf'].cuda()
        source_rgb_hf = mini_batch['source_rgb_hf'].cuda()
        source_tir_hf = mini_batch['source_tir_hf'].cuda()

        bs, _, _, _ = source_rgb_lf.shape

        # Model forward pass
        results_dict = net(source_rgb_lf, source_rgb_hf, source_tir_lf, source_tir_hf)
        flow_est = results_dict['flow_preds'][-1].permute(0, 2, 3, 1).cuda()

        # Warp the RGB input image based on the estimated flow
        w_rgb_img = (F.grid_sample(
            input_rgb.cpu(), flow_est.cpu(),
            align_corners=True
        ).squeeze().permute(1, 2, 0).numpy() * 255.).astype(np.uint8)

        save_warp_rgb = str(args.metric_path) + str(args.model_name) + '/' + str(im_name)[2:-2]
        save_flow_est = str(args.metric_path) + str(args.model_name) + '/' + str(im_name)[2:-6] + '_flow.png'
        save_flow_gt = str(args.metric_path) + str(args.model_name) + '/' + str(im_name)[2:-6] + '_flow_gt.png'

        print(save_flow_est)

        # Save the visualized estimated flow
        save_vis_flow_tofile(flow_est.squeeze().cpu().numpy(), save_flow_est)

        # Save the warped RGB image
        warp_rgb = cv2.cvtColor(w_rgb_img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(save_warp_rgb, warp_rgb)


def epe(input_flow, target_flow):
    """
    Compute End-Point-Error (EPE) between the estimated and ground-truth flow.

    Args:
        input_flow (torch.Tensor): predicted flow [B, H, W, 2]
        target_flow (torch.Tensor): ground-truth flow [B, H, W, 2]

    Returns:
        torch.Tensor: averaged EPE value.
    """
    return torch.norm(target_flow - input_flow, p=2, dim=1).mean()


def calculate_epe_rgbt(args, net, val_loader, img_size=256):
    """
    Calculate the averaged EPE over the entire RGB-T validation dataset.

    Args:
        args: arguments including paths and model name.
        net: trained model.
        val_loader: validation dataloader.
        img_size: image resolution (default=256).

    Returns:
        list: averaged EPE values for each image.
    """
    aepe_array = []
    name_array = []
    n_registered_pxs = 0

    for k, mini_batch in enumerate(val_loader):
        name = mini_batch['im_name']
        source_rgb_lf = mini_batch['source_rgb_lf'].cuda()
        source_tir_lf = mini_batch['source_tir_lf'].cuda()
        source_rgb_hf = mini_batch['source_rgb_hf'].cuda()
        source_tir_hf = mini_batch['source_tir_hf'].cuda()

        # Forward pass
        results_dict = net(source_rgb_lf, source_rgb_hf, source_tir_lf, source_tir_hf)
        flow_est = results_dict['flow_preds'][-1].permute(0, 2, 3, 1).cuda()
        flow_target = mini_batch['flow_gt'][-1].permute(0, 2, 3, 1).cuda()

        # Apply valid mask (flow values in [-1, 1])
        mask_x_gt = flow_target[:, :, :, 0].ge(-1) & flow_target[:, :, :, 0].le(1)
        mask_y_gt = flow_target[:, :, :, 1].ge(-1) & flow_target[:, :, :, 1].le(1)
        mask_gt = mask_x_gt & mask_y_gt
        mask_gt = torch.cat((mask_gt.unsqueeze(3), mask_gt.unsqueeze(3)), dim=3)

        # Unnormalize the flows from [-1,1] to [0, img_size-1]
        for i in range(flow_est.shape[0]):
            flow_target[i] = (flow_target[i] + 1) * (img_size - 1) / 2
            flow_est[i] = (flow_est[i] + 1) * (img_size - 1) / 2

        # Extract valid pixels
        flow_target_x = flow_target[:, :, :, 0]
        flow_target_y = flow_target[:, :, :, 1]
        flow_est_x = flow_est[:, :, :, 0]
        flow_est_y = flow_est[:, :, :, 1]

        flow_target = torch.cat((
            flow_target_x[mask_gt[:, :, :, 0]].unsqueeze(1),
            flow_target_y[mask_gt[:, :, :, 1]].unsqueeze(1)
        ), dim=1)
        flow_est = torch.cat((
            flow_est_x[mask_gt[:, :, :, 0]].unsqueeze(1),
            flow_est_y[mask_gt[:, :, :, 1]].unsqueeze(1)
        ), dim=1)

        # Calculate EPE
        aepe = epe(flow_est, flow_target)
        aepe_array.append(aepe.item())
        name_array.append(str(name))
        n_registered_pxs += flow_target.shape[0]

    clm = ["epe"]
    df = pd.DataFrame(data=aepe_array, index=name_array, columns=clm)
    out_dir = osp.join(args.metric_path, args.model_name + '_aepe.csv')
    df.to_csv(out_dir)
    return aepe_array


def correct_correspondences(input_flow, target_flow, alpha, img_size=256):
    """
    Calculate the number of correct correspondences within a given threshold (PCK computation).

    Args:
        input_flow (torch.Tensor): predicted flow.
        target_flow (torch.Tensor): ground-truth flow.
        alpha (float): threshold ratio.
        img_size (int): image size.

    Returns:
        int: number of correct correspondences.
    """
    input_flow = input_flow.unsqueeze(0)
    target_flow = target_flow.unsqueeze(0)
    dist = torch.norm(target_flow - input_flow, p=2, dim=0).unsqueeze(0)
    pck_threshold = alpha * img_size
    mask = dist.le(pck_threshold)

    return len(dist[mask.detach()])


def calculate_pck_rgbt(args, net, val_loader, alpha=1, img_size=256):
    """
    Calculate the averaged PCK over the entire RGB-T validation dataset.

    Args:
        args: arguments including paths and model name.
        net: trained model.
        val_loader: validation dataloader.
        alpha (float): threshold ratio (default=1).
        img_size (int): image resolution (default=256).

    Returns:
        tuple: (list of image names, list of corresponding PCK values)
    """
    name_array = []
    pck_array = []

    for i, mini_batch in enumerate(val_loader):
        name = mini_batch['im_name']
        source_rgb_lf = mini_batch['source_rgb_lf'].cuda()
        source_tir_lf = mini_batch['source_tir_lf'].cuda()
        source_rgb_hf = mini_batch['source_rgb_hf'].cuda()
        source_tir_hf = mini_batch['source_tir_hf'].cuda()

        # Forward pass
        results_dict = net(source_rgb_lf, source_rgb_hf, source_tir_lf, source_tir_hf)
        flow_est = results_dict['flow_preds'][-1].permute(0, 2, 3, 1).cuda()
        flow_target = mini_batch['flow_gt'][-1].permute(0, 2, 3, 1).cuda()

        bs, ch_g, h_g, w_g = flow_target.shape

        # Apply valid mask
        mask_x_gt = flow_target[:, :, :, 0].ge(-1) & flow_target[:, :, :, 0].le(1)
        mask_y_gt = flow_target[:, :, :, 1].ge(-1) & flow_target[:, :, :, 1].le(1)
        mask_gt = mask_x_gt & mask_y_gt
        mask_gt = torch.cat((mask_gt.unsqueeze(3), mask_gt.unsqueeze(3)), dim=3)

        # Unnormalize the flows
        for i in range(bs):
            flow_target[i] = (flow_target[i] + 1) * (img_size - 1) / 2
            flow_est[i] = (flow_est[i] + 1) * (img_size - 1) / 2

        flow_target = flow_target.contiguous().view(1, bs * h_g * w_g * ch_g)
        flow_est = flow_est.contiguous().view(1, bs * h_g * w_g * ch_g)

        flow_target_m = flow_target[mask_gt.contiguous().view(1, bs * h_g * w_g * ch_g)]
        flow_est_m = flow_est[mask_gt.contiguous().view(1, bs * h_g * w_g * ch_g)]

        corr = len(flow_target_m)
        c_corr = correct_correspondences(flow_est_m, flow_target_m, alpha=alpha, img_size=img_size)

        pck = c_corr / (corr + 1e-6)
        pck_array.append(pck)
        name_array.append(str(name))

    return name_array, pck_array
