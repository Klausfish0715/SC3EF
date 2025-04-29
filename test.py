import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'  # Set CUDA visible devices
from os import path as osp

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Import custom modules
from data.dataset_simu_KAIST_freq_32 import simu_KAIST_32
from evaluate_pyr import calculate_epe_rgbt, calculate_pck_rgbt, save_flow
from gmflow.p2t_tiny_convnext_2dec_v7_secrosshf import PyramidPoolingTransformer


def get_args_parser():
    """
    Argument parser for evaluation and flow visualization.
    """
    parser = argparse.ArgumentParser()

    # Paths
    parser.add_argument('--image-val-path', type=str,
                        default='/xxxxx/test_occluded',
                        help='Path to the validation dataset (e.g., MFNet)')
    parser.add_argument('--flow-path', type=str,
                        default='./results/xxxxx/',
                        help='Directory to save the generated flows')
    parser.add_argument('--metric-path', type=str,
                        default='./results/',
                        help='Directory to save evaluation metrics')
    parser.add_argument('--model-name', type=str,
                        default='xxxxx',
                        help='Name identifier for saving results')

    # Dataset
    parser.add_argument('--checkpoint-dir', type=str,
                        default='/xxx/xxx/step_382500.pth',
                        help='Path to the model checkpoint')
    parser.add_argument('--stage', type=str,
                        default='kaist',
                        help='Training stage identifier')
    parser.add_argument('--image-size', type=int, nargs='+',
                        default=[256, 256],
                        help='Input image size')
    parser.add_argument('--val-dataset', type=str, nargs='+',
                        default=['kaist'],
                        help='Validation dataset list')

    # Model and training options
    parser.add_argument('--resume', type=str,
                        default=None,
                        help='Path to resume training or fine-tuning')
    parser.add_argument('--strict-resume', action='store_true',
                        help='Strictly enforce checkpoint loading')
    parser.add_argument('--local-rank', type=int, default=0,
                        help='Local rank for distributed evaluation')

    return parser


def main(args):
    """
    Main evaluation pipeline for computing metrics or saving flows.
    """
    # Set random seeds for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True

    # Create flow output directory if it doesn't exist
    if not os.path.exists(args.flow_path):
        os.makedirs(args.flow_path)

    # Initialize model
    model = PyramidPoolingTransformer().cuda()

    # Load model checkpoint
    checkpoint = torch.load(args.checkpoint_dir, map_location='cuda')
    weights = checkpoint['model'] if 'model' in checkpoint else checkpoint
    model.load_state_dict(weights, strict=args.strict_resume)
    model.eval()

    args.tag = 'test'

    # Prepare validation dataloader
    test_dataset = simu_KAIST_32(args)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=1
    )

    args.metric = 'flow'  # Set the metric (aepe, pck, or flow visualization)

    with torch.no_grad():
        if args.metric == 'aepe':
            epe_arr = calculate_epe_rgbt(args, model, test_dataloader)
            print(np.mean(epe_arr))

        if args.metric == 'pck':
            threshold_range = np.asarray([1.0 / 256, 3.0 / 256, 5.0 / 256])
            pck_array = []
            thr_array = []
            im_size = 256

            for t_id, threshold in enumerate(threshold_range):
                thresh_str = 'pck - ' + str(int(threshold * im_size)) + 'px'
                name, pck = calculate_pck_rgbt(
                    args, model, test_dataloader,
                    alpha=threshold, img_size=im_size
                )
                print(t_id, threshold)
                pck_array.append(pck)
                thr_array.append(thresh_str)

            # Save PCK results to CSV
            pck_array = list(map(list, zip(*pck_array)))
            df = pd.DataFrame(data=pck_array, index=name, columns=thr_array)
            out_dir = osp.join(args.metric_path, args.model_name + '_pck.csv')
            df.to_csv(out_dir)

        if args.metric == 'flow':
            save_flow(args, model, test_dataloader)


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    # Set local rank if not set by environment
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    main(args)
