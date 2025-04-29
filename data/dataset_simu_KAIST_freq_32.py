import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


def center_crop(img, size):
    """
    Get the center crop of the input image
    Args:
        img: input image [BxCxHxW]
        size: size of the center crop (tuple)
    Output:
        img_pad: center crop
        x, y: coordinates of the crop
    """

    if not isinstance(size, tuple):
        size = (size, size)

    img = img.copy()
    w, h = img.shape[1::-1]

    pad_w = 0
    pad_h = 0
    if w < size[0]:
        pad_w = np.uint16((size[0] - w) / 2)
    if h < size[1]:
        pad_h = np.uint16((size[1] - h) / 2)
    img_pad = cv2.copyMakeBorder(img,
                                 pad_h,
                                 pad_h,
                                 pad_w,
                                 pad_w,
                                 cv2.BORDER_CONSTANT,
                                 value=[0, 0, 0])
    w, h = img_pad.shape[1::-1]

    x1 = w // 2 - size[0] // 2
    y1 = h // 2 - size[1] // 2

    img_pad = img_pad[y1:y1 + size[1], x1:x1 + size[0], :]

    return img_pad, x1, y1


class simu_KAIST_32(Dataset):

    def __init__(self, args):
        super().__init__()

        if args.tag == 'train':
            self.data_source = args.image_val_path
            self.rgb_dir = os.listdir(
                os.path.join(self.data_source, 'rgb_aug_hf'))
            self.input_rgb_path = self.data_source + '/rgb_aug/'
            self.input_tir_path = self.data_source + '/nir_aug/'
            self.rgb_hf_path = self.data_source + '/rgb_aug_hf/'
            self.rgb_lf_path = self.data_source + '/rgb_aug_lf/'
            self.tir_hf_path = self.data_source + '/nir_aug_hf/'
            self.tir_lf_path = self.data_source + '/nir_aug_lf/'
            self.flow_path = self.data_source + '/flow_label_aug/'

        elif args.tag == 'val':
            self.data_source = args.image_val_path
            self.rgb_dir = os.listdir(
                os.path.join(self.data_source, 'rgb_aug_hf'))
            self.input_rgb_path = self.data_source + '/rgb_aug/'
            self.input_tir_path = self.data_source + '/nir_aug/'
            self.rgb_hf_path = self.data_source + '/rgb_aug_hf/'
            self.rgb_lf_path = self.data_source + '/rgb_aug_lf/'
            self.tir_hf_path = self.data_source + '/nir_aug_hf/'
            self.tir_lf_path = self.data_source + '/nir_aug_lf/'
            self.flow_path = self.data_source + '/flow_label_aug/'

        elif args.tag == 'test':
            self.data_source = args.image_val_path
            self.rgb_dir = os.listdir(
                os.path.join(self.data_source, 'rgb_aug_hf'))
            self.input_rgb_path = self.data_source + '/rgb_aug/'
            self.input_tir_path = self.data_source + '/nir_aug/'
            self.rgb_hf_path = self.data_source + '/rgb_aug_hf/'
            self.rgb_lf_path = self.data_source + '/rgb_aug_lf/'
            self.tir_hf_path = self.data_source + '/nir_aug_hf/'
            self.tir_lf_path = self.data_source + '/nir_aug_lf/'
            self.flow_path = self.data_source + '/flow_label_aug/'

        self.rgb_list = [i for i in self.rgb_dir]

    def readFlow(self, fn):
        """ Read .flo file in Middlebury format"""
        # Code adapted from:
        # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

        # WARNING: this will work on little-endian architectures (eg Intel x86) only!
        # print 'fn = %s'%(fn)
        with open(fn, 'rb') as f:
            magic = np.fromfile(f, np.float32, count=1)
            if 2022.516 != magic:
                print('Magic number incorrect. Invalid .flo file')
                return None
            else:
                w = np.fromfile(f, np.int32, count=1)
                h = np.fromfile(f, np.int32, count=1)
                # print 'Reading %d x %d flo file\n' % (w, h)
                data = np.fromfile(f, np.float32, count=2 * int(w) * int(h))
                # Reshape testdata into 3D array (columns, rows, bands)
                # The reshape here is for visualization, the original code is (w,h,2)
                return np.resize(data, (int(h), int(w), 2))

    def __len__(self):
        return len(self.rgb_list)

    def __getitem__(self, idx):

        # input_rgb = self.input_rgb_path + self.rgb_list[idx][:17] + '.png'
        # input_tir = self.input_tir_path + self.rgb_list[idx][:17] + '.png'

        # rgb_hf_path = self.rgb_hf_path + self.rgb_list[idx]
        # rgb_lf_path = self.rgb_lf_path + self.rgb_list[idx]

        # tir_hf_path = self.tir_hf_path + self.rgb_list[idx]
        # tir_lf_path = self.tir_lf_path + self.rgb_list[idx]

        # flow_path = self.flow_path + self.rgb_list[idx][:-4] + '.flo'

        input_rgb = self.input_rgb_path + self.rgb_list[idx]
        input_tir = self.input_tir_path + self.rgb_list[idx]
        
        rgb_hf_path = self.rgb_hf_path + self.rgb_list[idx]
        rgb_lf_path = self.rgb_lf_path + self.rgb_list[idx]

        tir_hf_path = self.tir_hf_path + self.rgb_list[idx]
        tir_lf_path = self.tir_lf_path + self.rgb_list[idx]

        flow_path = self.flow_path + self.rgb_list[idx][:-4] + '.flo'

        input_rgb = np.array(Image.open(input_rgb)).astype(np.float32) / 255.0
        input_tir = np.array(Image.open(input_tir)).astype(np.float32) / 255.0

        input_rgb_hf = np.array(Image.open(rgb_hf_path)).astype(
            np.float32) / 255.0
        input_rgb_lf = np.array(Image.open(rgb_lf_path)).astype(
            np.float32) / 255.0

        input_tir_hf = np.array(Image.open(tir_hf_path)).astype(
            np.float32) / 255.0
        input_tir_lf = np.array(Image.open(tir_lf_path)).astype(
            np.float32) / 255.0

        # to tensor
        input_rgb_pyr = []
        input_tir_pyr = []
        w, h, c = input_rgb.shape

        input_rgb_hf_32 = torch.from_numpy(
            cv2.resize(input_rgb_hf,
                       (int(w / 32), int(h / 32)))).permute(2, 0, 1)
        input_rgb_lf_32 = torch.from_numpy(
            cv2.resize(input_rgb_lf,
                       (int(w / 32), int(h / 32)))).permute(2, 0, 1)
        input_tir_hf_32 = torch.from_numpy(
            cv2.resize(input_tir_hf,
                       (int(w / 32), int(h / 32)))).permute(2, 0, 1)
        input_tir_lf_32 = torch.from_numpy(
            cv2.resize(input_tir_lf,
                       (int(w / 32), int(h / 32)))).permute(2, 0, 1)

        rgb_1_32 = torch.from_numpy(
            cv2.resize(input_rgb,
                       (int(w / 32), int(h / 32)))).permute(2, 0, 1)
        rgb_1_16 = torch.from_numpy(
            cv2.resize(input_rgb,
                       (int(w / 16), int(h / 16)))).permute(2, 0, 1)
        rgb_1_8 = torch.from_numpy(
            cv2.resize(input_rgb, (int(w / 8), int(h / 8)))).permute(2, 0, 1)
        rgb_1_4 = torch.from_numpy(
            cv2.resize(input_rgb, (int(w / 4), int(h / 4)))).permute(2, 0, 1)
        rgb_1_2 = torch.from_numpy(
            cv2.resize(input_rgb, (int(w / 2), int(h / 2)))).permute(2, 0, 1)
        rgb_1_1 = torch.from_numpy(input_rgb).permute(2, 0, 1)

        input_rgb_pyr.append(rgb_1_32)
        input_rgb_pyr.append(rgb_1_16)
        input_rgb_pyr.append(rgb_1_8)
        input_rgb_pyr.append(rgb_1_4)
        input_rgb_pyr.append(rgb_1_2)
        input_rgb_pyr.append(rgb_1_1)

        tir_1_32 = torch.from_numpy(
            cv2.resize(input_tir,
                       (int(w / 32), int(h / 32)))).permute(2, 0, 1)
        tir_1_16 = torch.from_numpy(
            cv2.resize(input_tir,
                       (int(w / 16), int(h / 16)))).permute(2, 0, 1)
        tir_1_8 = torch.from_numpy(
            cv2.resize(input_tir, (int(w / 8), int(h / 8)))).permute(2, 0, 1)
        tir_1_4 = torch.from_numpy(
            cv2.resize(input_tir, (int(w / 4), int(h / 4)))).permute(2, 0, 1)
        tir_1_2 = torch.from_numpy(
            cv2.resize(input_tir, (int(w / 2), int(h / 2)))).permute(2, 0, 1)
        tir_1_1 = torch.from_numpy(input_tir).permute(2, 0, 1)

        input_tir_pyr.append(tir_1_32)
        input_tir_pyr.append(tir_1_16)
        input_tir_pyr.append(tir_1_8)
        input_tir_pyr.append(tir_1_4)
        input_tir_pyr.append(tir_1_2)
        input_tir_pyr.append(tir_1_1)

        rgb_lf = torch.from_numpy(input_rgb_lf).permute(2, 0, 1)
        tir_lf = torch.from_numpy(input_tir_lf).permute(2, 0, 1)
        rgb_hf = torch.from_numpy(input_rgb_hf).permute(2, 0, 1)
        tir_hf = torch.from_numpy(input_tir_hf).permute(2, 0, 1)

        flow_pyr = []
        valid_pyr = []

        flow_1_1 = self.readFlow(flow_path)
        flow = torch.from_numpy(flow_1_1).permute(2, 0, 1)

        flow_1_2 = torch.from_numpy(
            cv2.resize(flow_1_1, (int(w / 2), int(h / 2)))).permute(2, 0, 1)
        flow_1_4 = torch.from_numpy(
            cv2.resize(flow_1_1, (int(w / 4), int(h / 4)))).permute(2, 0, 1)
        flow_1_8 = torch.from_numpy(
            cv2.resize(flow_1_1, (int(w / 8), int(h / 8)))).permute(2, 0, 1)
        flow_1_16 = torch.from_numpy(
            cv2.resize(flow_1_1, (int(w / 16), int(h / 16)))).permute(2, 0, 1)
        flow_1_32 = torch.from_numpy(
            cv2.resize(flow_1_1, (int(w / 32), int(h / 32)))).permute(2, 0, 1)

        flow_pyr.append(flow_1_32)
        flow_pyr.append(flow_1_16)
        flow_pyr.append(flow_1_8)
        flow_pyr.append(flow_1_4)
        flow_pyr.append(flow_1_2)
        flow_pyr.append(flow)

        valid_1_32 = (flow_1_32[0].abs() < 1000) & (flow_1_32[1].abs() < 1000)
        valid_1_16 = (flow_1_16[0].abs() < 1000) & (flow_1_16[1].abs() < 1000)
        valid_1_8 = (flow_1_8[0].abs() < 1000) & (flow_1_8[1].abs() < 1000)
        valid_1_4 = (flow_1_4[0].abs() < 1000) & (flow_1_4[1].abs() < 1000)
        valid_1_2 = (flow_1_2[0].abs() < 1000) & (flow_1_2[1].abs() < 1000)
        valid_1_1 = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)
        valid_pyr.append(valid_1_32)
        valid_pyr.append(valid_1_16)
        valid_pyr.append(valid_1_8)
        valid_pyr.append(valid_1_4)
        valid_pyr.append(valid_1_2)
        valid_pyr.append(valid_1_1)

        return {
            'im_name': self.rgb_list[idx],
            'input_rgb': input_rgb_pyr,
            'input_tir': input_tir_pyr,
            'source_rgb_lf': rgb_lf,
            'source_rgb_lf_32': input_rgb_lf_32,
            'source_tir_lf': tir_lf,
            'source_tir_lf_32': input_tir_lf_32,
            'source_rgb_hf': rgb_hf,
            'source_rgb_hf_32': input_rgb_hf_32,
            'source_tir_hf': tir_hf,
            'source_tir_hf_32': input_tir_hf_32,
            'flow_gt': flow_pyr,
            'valid': valid_pyr
        }


class simu_KAIST_32_orginput(Dataset):

    def __init__(self, args):
        super().__init__()

        if args.tag == 'train':
            self.data_source = args.image_train_path
            self.rgb_dir = os.listdir(
                os.path.join(self.data_source, 'EH_train_rgb'))
            self.input_rgb_path = self.data_source + '/EH_train_rgb/'
            self.input_tir_path = self.data_source + '/EH_train_tir/'
            self.rgb_hf_path = self.data_source + '/EH_train_rgb/'
            self.rgb_lf_path = self.data_source + '/EH_train_rgb/'
            self.tir_hf_path = self.data_source + '/EH_train_tir/'
            self.tir_lf_path = self.data_source + '/EH_train_tir/'
            self.flow_path = self.data_source + '/train_flow_label/'

        elif args.tag == 'test':
            self.data_source = args.image_val_path
            self.rgb_dir = os.listdir(
                os.path.join(self.data_source, 'EH_test_rgb_clean'))
            self.input_rgb_path = self.data_source + '/EH_test_rgb_clean/'
            self.input_tir_path = self.data_source + '/EH_test_tir_clean/'
            self.rgb_hf_path = self.data_source + '/EH_test_rgb_clean/'
            self.rgb_lf_path = self.data_source + '/EH_test_rgb_clean/'
            self.tir_hf_path = self.data_source + '/EH_test_tir_clean/'
            self.tir_lf_path = self.data_source + '/EH_test_tir_clean/'
            self.flow_path = self.data_source + '/test_flow_gt/'

        # elif args.tag == 'test':
        #     self.data_source = args.image_val_path
        #     self.rgb_dir = os.listdir(
        #         os.path.join(self.data_source, 'test_rgb_hf_aug'))
        #     self.input_rgb_path = self.data_source + '/EH_test_rgb/'
        #     self.input_tir_path = self.data_source + '/EH_test_tir/'
        #     self.rgb_hf_path = self.data_source + '/test_rgb_hf_aug/'
        #     self.rgb_lf_path = self.data_source + '/test_rgb_lf_aug/'
        #     self.tir_hf_path = self.data_source + '/test_tir_hf_aug/'
        #     self.tir_lf_path = self.data_source + '/test_tir_lf_aug/'
        #     self.flow_path = self.data_source + '/test_flow_label_aug/'

        self.rgb_list = [i for i in self.rgb_dir]

    def readFlow(self, fn):
        """ Read .flo file in Middlebury format"""
        # Code adapted from:
        # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

        # WARNING: this will work on little-endian architectures (eg Intel x86) only!
        # print 'fn = %s'%(fn)
        with open(fn, 'rb') as f:
            magic = np.fromfile(f, np.float32, count=1)
            if 2022.516 != magic:
                print('Magic number incorrect. Invalid .flo file')
                return None
            else:
                w = np.fromfile(f, np.int32, count=1)
                h = np.fromfile(f, np.int32, count=1)
                # print 'Reading %d x %d flo file\n' % (w, h)
                data = np.fromfile(f, np.float32, count=2 * int(w) * int(h))
                # Reshape testdata into 3D array (columns, rows, bands)
                # The reshape here is for visualization, the original code is (w,h,2)
                return np.resize(data, (int(h), int(w), 2))

    def __len__(self):
        return len(self.rgb_list)

    def __getitem__(self, idx):

        input_rgb = self.input_rgb_path + self.rgb_list[idx]
        input_tir = self.input_tir_path + self.rgb_list[idx]

        rgb_hf_path = self.rgb_hf_path + self.rgb_list[idx]
        rgb_lf_path = self.rgb_lf_path + self.rgb_list[idx]

        tir_hf_path = self.tir_hf_path + self.rgb_list[idx]
        tir_lf_path = self.tir_lf_path + self.rgb_list[idx]

        flow_path = self.flow_path + self.rgb_list[idx][:-4] + '.flo'

        input_rgb = np.array(Image.open(input_rgb)).astype(np.float32) / 255.0
        input_tir = np.array(Image.open(input_tir)).astype(np.float32) / 255.0

        input_rgb_hf = np.array(Image.open(rgb_hf_path)).astype(
            np.float32) / 255.0
        input_rgb_lf = np.array(Image.open(rgb_lf_path)).astype(
            np.float32) / 255.0

        input_tir_hf = np.array(Image.open(tir_hf_path)).astype(
            np.float32) / 255.0
        input_tir_lf = np.array(Image.open(tir_lf_path)).astype(
            np.float32) / 255.0

        # to tensor
        input_rgb_pyr = []
        input_tir_pyr = []
        w, h, c = input_rgb.shape

        rgb_1_32 = torch.from_numpy(
            cv2.resize(input_rgb,
                       (int(w / 32), int(h / 32)))).permute(2, 0, 1)
        rgb_1_16 = torch.from_numpy(
            cv2.resize(input_rgb,
                       (int(w / 16), int(h / 16)))).permute(2, 0, 1)
        rgb_1_8 = torch.from_numpy(
            cv2.resize(input_rgb, (int(w / 8), int(h / 8)))).permute(2, 0, 1)
        rgb_1_4 = torch.from_numpy(
            cv2.resize(input_rgb, (int(w / 4), int(h / 4)))).permute(2, 0, 1)
        rgb_1_2 = torch.from_numpy(
            cv2.resize(input_rgb, (int(w / 2), int(h / 2)))).permute(2, 0, 1)
        rgb_1_1 = torch.from_numpy(input_rgb).permute(2, 0, 1)

        input_rgb_pyr.append(rgb_1_32)
        input_rgb_pyr.append(rgb_1_16)
        input_rgb_pyr.append(rgb_1_8)
        input_rgb_pyr.append(rgb_1_4)
        input_rgb_pyr.append(rgb_1_2)
        input_rgb_pyr.append(rgb_1_1)

        tir_1_32 = torch.from_numpy(
            cv2.resize(input_tir,
                       (int(w / 32), int(h / 32)))).permute(2, 0, 1)
        tir_1_16 = torch.from_numpy(
            cv2.resize(input_tir,
                       (int(w / 16), int(h / 16)))).permute(2, 0, 1)
        tir_1_8 = torch.from_numpy(
            cv2.resize(input_tir, (int(w / 8), int(h / 8)))).permute(2, 0, 1)
        tir_1_4 = torch.from_numpy(
            cv2.resize(input_tir, (int(w / 4), int(h / 4)))).permute(2, 0, 1)
        tir_1_2 = torch.from_numpy(
            cv2.resize(input_tir, (int(w / 2), int(h / 2)))).permute(2, 0, 1)
        tir_1_1 = torch.from_numpy(input_tir).permute(2, 0, 1)

        input_tir_pyr.append(tir_1_32)
        input_tir_pyr.append(tir_1_16)
        input_tir_pyr.append(tir_1_8)
        input_tir_pyr.append(tir_1_4)
        input_tir_pyr.append(tir_1_2)
        input_tir_pyr.append(tir_1_1)

        rgb_lf = torch.from_numpy(input_rgb_lf).permute(2, 0, 1)
        tir_lf = torch.from_numpy(input_tir_lf).permute(2, 0, 1)
        rgb_hf = torch.from_numpy(input_rgb_hf).permute(2, 0, 1)
        tir_hf = torch.from_numpy(input_tir_hf).permute(2, 0, 1)

        flow_pyr = []
        valid_pyr = []

        flow_1_1 = self.readFlow(flow_path)
        flow = torch.from_numpy(flow_1_1).permute(2, 0, 1)

        flow_1_2 = torch.from_numpy(
            cv2.resize(flow_1_1, (int(w / 2), int(h / 2)))).permute(2, 0, 1)
        flow_1_4 = torch.from_numpy(
            cv2.resize(flow_1_1, (int(w / 4), int(h / 4)))).permute(2, 0, 1)
        flow_1_8 = torch.from_numpy(
            cv2.resize(flow_1_1, (int(w / 8), int(h / 8)))).permute(2, 0, 1)
        flow_1_16 = torch.from_numpy(
            cv2.resize(flow_1_1, (int(w / 16), int(h / 16)))).permute(2, 0, 1)
        flow_1_32 = torch.from_numpy(
            cv2.resize(flow_1_1, (int(w / 32), int(h / 32)))).permute(2, 0, 1)

        flow_pyr.append(flow_1_32)
        flow_pyr.append(flow_1_16)
        flow_pyr.append(flow_1_8)
        flow_pyr.append(flow_1_4)
        flow_pyr.append(flow_1_2)
        flow_pyr.append(flow)

        valid_1_32 = (flow_1_32[0].abs() < 1000) & (flow_1_32[1].abs() < 1000)
        valid_1_16 = (flow_1_16[0].abs() < 1000) & (flow_1_16[1].abs() < 1000)
        valid_1_8 = (flow_1_8[0].abs() < 1000) & (flow_1_8[1].abs() < 1000)
        valid_1_4 = (flow_1_4[0].abs() < 1000) & (flow_1_4[1].abs() < 1000)
        valid_1_2 = (flow_1_2[0].abs() < 1000) & (flow_1_2[1].abs() < 1000)
        valid_1_1 = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)
        valid_pyr.append(valid_1_32)
        valid_pyr.append(valid_1_16)
        valid_pyr.append(valid_1_8)
        valid_pyr.append(valid_1_4)
        valid_pyr.append(valid_1_2)
        valid_pyr.append(valid_1_1)

        return {
            'im_name': self.rgb_list[idx],
            'input_rgb': input_rgb_pyr,
            'input_tir': input_tir_pyr,
            'source_rgb_lf': rgb_lf,
            'source_tir_lf': tir_lf,
            'source_rgb_hf': rgb_hf,
            'source_tir_hf': tir_hf,
            'flow_gt': flow_pyr,
            'valid': valid_pyr
        }


class UAV(Dataset):

    def __init__(self, args):
        super().__init__()

        if args.tag == 'train':
            self.data_source = args.image_train_path
            self.rgb_dir = os.listdir(
                os.path.join(self.data_source, 'EH_train_rgb'))
            self.rgb_path = self.data_source + '/EH_train_rgb/'
            self.tir_path = self.data_source + '/EH_train_tir/'
            self.flow_path = self.data_source + '/train_flow_label/'

        elif args.tag == 'test':
            self.data_source = args.image_val_path
            self.rgb_dir = os.listdir(os.path.join(self.data_source,
                                                   'M30_vis'))
            self.input_rgb_path = self.data_source + '/M30_vis/'
            self.input_tir_path = self.data_source + '/M30_tir/'
            self.rgb_hf_path = self.data_source + '/M30_vis_hf/'
            self.rgb_lf_path = self.data_source + '/M30_vis_lf/'
            self.tir_hf_path = self.data_source + '/M30_tir_hf/'
            self.tir_lf_path = self.data_source + '/M30_tir_lf/'
            # self.flow_path = self.data_source + '/test_flow_gt/'

        self.rgb_list = [i for i in self.rgb_dir]

    def readFlow(self, fn):
        """ Read .flo file in Middlebury format"""
        # Code adapted from:
        # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

        # WARNING: this will work on little-endian architectures (eg Intel x86) only!
        # print 'fn = %s'%(fn)
        with open(fn, 'rb') as f:
            magic = np.fromfile(f, np.float32, count=1)
            if 2022.516 != magic:
                print('Magic number incorrect. Invalid .flo file')
                return None
            else:
                w = np.fromfile(f, np.int32, count=1)
                h = np.fromfile(f, np.int32, count=1)
                # print 'Reading %d x %d flo file\n' % (w, h)
                data = np.fromfile(f, np.float32, count=2 * int(w) * int(h))
                # Reshape testdata into 3D array (columns, rows, bands)
                # The reshape here is for visualization, the original code is (w,h,2)
                return np.resize(data, (int(h), int(w), 2))

    def __len__(self):
        return len(self.rgb_list)

    def __getitem__(self, idx):

        input_rgb = self.input_rgb_path + self.rgb_list[idx]
        input_tir = self.input_tir_path + self.rgb_list[idx]

        rgb_hf_path = self.rgb_hf_path + self.rgb_list[idx]
        rgb_lf_path = self.rgb_lf_path + self.rgb_list[idx]

        tir_hf_path = self.tir_hf_path + self.rgb_list[idx]
        tir_lf_path = self.tir_lf_path + self.rgb_list[idx]

        # flow_path = self.flow_path + self.rgb_list[idx][:-4] + '.flo'

        input_rgb = np.array(Image.open(input_rgb)).astype(np.float32) / 255.0
        input_tir = np.array(Image.open(input_tir)).astype(np.float32) / 255.0

        input_rgb_hf = np.array(Image.open(rgb_hf_path)).astype(
            np.float32) / 255.0
        input_rgb_lf = np.array(Image.open(rgb_lf_path)).astype(
            np.float32) / 255.0

        input_tir_hf = np.array(Image.open(tir_hf_path)).astype(
            np.float32) / 255.0
        input_tir_lf = np.array(Image.open(tir_lf_path)).astype(
            np.float32) / 255.0

        rgb = torch.from_numpy(input_rgb).permute(2, 0, 1)
        tir = torch.from_numpy(input_tir).permute(2, 0, 1)

        rgb_hf = torch.from_numpy(input_rgb_hf).permute(2, 0, 1)
        rgb_lf = torch.from_numpy(input_rgb_lf).permute(2, 0, 1)
        tir_hf = torch.from_numpy(input_tir_hf).permute(2, 0, 1)
        tir_lf = torch.from_numpy(input_tir_lf).permute(2, 0, 1)

        return {
            'im_name': self.rgb_list[idx],
            'input_rgb': rgb,
            'input_tir': tir,
            'source_rgb_lf': rgb_lf,
            'source_tir_lf': tir_lf,
            'source_rgb_hf': rgb_hf,
            'source_tir_hf': tir_hf
        }