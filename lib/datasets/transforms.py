"""Transform input data.

Images are resized with Pillow which has a different coordinate convention:
https://pillow.readthedocs.io/en/3.3.x/handbook/concepts.html#coordinate-system

> The Python Imaging Library uses a Cartesian pixel coordinate system,
  with (0,0) in the upper left corner. Note that the coordinates refer to
  the implied pixel corners; the centre of a pixel addressed as (0, 0)
  actually lies at (0.5, 0.5).
"""

import os
import math
import numpy as np
import cv2 as cv
import torch
from torchvision import transforms
from lib.datasets.heatmap import putGaussianMaps, putGaussianMaps_ellipsoid, show_heatmap_3D
from lib.datasets.datasets import MyDatasets
import matplotlib.pyplot as plt


def resize_pro(origin_img, mnt, dst_size):
    mnt_x = mnt[0]
    # print(mnt_x)
    mnt_y = mnt[1]
    (h, w) = origin_img.shape[:2]
    d0 = max(h, w)
    scale = dst_size[0] * 1.0 / d0
    # print(sample['label'])
    if h > w:
        tran_img = cv.copyMakeBorder(origin_img, 0, 0, int(np.floor((h - w) / 2.0)), int(np.ceil((h - w) / 2.0)),
                                     cv.BORDER_REPLICATE)
        mnt_x = [int((int(i) + np.floor((h - w) / 2.0))) * scale for i in mnt_x]
        mnt_y = [int(int(i) * scale) for i in mnt_y]
    else:
        tran_img = cv.copyMakeBorder(origin_img, int(np.floor((w - h) / 2.0)), int(np.ceil((w - h) / 2.0)), 0, 0,
                                     cv.BORDER_REPLICATE)
        mnt_x = [int(int(i) * scale) for i in mnt_x]
        mnt_y = [int((int(i) + np.floor((w - h) / 2.0)) * scale) for i in mnt_y]
    # print(tran_img.shape)
    resize_img = cv.resize(tran_img, (dst_size[0], dst_size[1]))
    mnt_x_ = []
    mnt_y_ = []
    mnt_label_true = []
    # rgb_img_resize = cv.cvtColor(resize_img, cv.COLOR_GRAY2RGB)
    scale = dst_size[0] * 1.0 / d0
    for k in range(0, len(mnt_x)):
        x_ = int(mnt_x[k] * scale)
        y_ = int(mnt_y[k] * scale)
        mnt_temp_ = []
        mnt_temp_.append(x_)
        mnt_temp_.append(y_)
        mnt_x_.append(x_)
        mnt_y_.append(y_)
        mnt_label_true.append(mnt_temp_)
    #     x2_ = np.round(math.sin(int(mnt_theta[k]) * math.pi / 180.0) * 15)
    #     y2_ = np.round(math.cos(int(mnt_theta[k]) * math.pi / 180.0) * 15)
    #     cv.arrowedLine(rgb_img_resize, (x_, y_),
    #                    (int(mnt_x[k] * scale + x2_), int(mnt_y[k] * scale + y2_)), (0, 0, 255), 2, 0, 0, 0.4)
    # cv.imshow("rgb_img_resize", rgb_img_resize)
    # cv.waitKey(0)
    # label_new = []
    # str_x = [str(x) for x in mnt_x_]
    # label_new.append(" ".join(str_x))
    # str_y = [str(x) for x in mnt_y_]
    # label_new.append(" ".join(str_y))
    # str_theta = [str(x) for x in mnt_theta]
    # label_new.append(" ".join(str_theta))
    heatmaps = np.zeros((int(dst_size[0] / 4), int(dst_size[1] / 4)))
    for k in range(len(mnt_x_)):
        heatmaps = putGaussianMaps((mnt_x_[k], mnt_y_[k]), heatmaps, 1.0,
                                   int(dst_size[0] / 4), int(dst_size[1] / 4), 4, 20)
    return resize_img, heatmaps, mnt_label_true


def resize_to_square(origin_img, dst_size):
    (h, w) = origin_img.shape[:2]
    d0 = max(h, w)
    if h > w:
        tran_img = cv.copyMakeBorder(origin_img, 0, 0, int(np.floor((h - w) / 2.0)), int(np.ceil((h - w) / 2.0)),
                                     cv.BORDER_REPLICATE)
    else:
        tran_img = cv.copyMakeBorder(origin_img, int(np.floor((w - h) / 2.0)), int(np.ceil((w - h) / 2.0)), 0, 0,
                                     cv.BORDER_REPLICATE)
    # print(tran_img.shape)
    resize_img = cv.resize(tran_img, (dst_size[0], dst_size[1]))
    return resize_img


def resize_heatmap(heatmap, dst_size):
    resize_heatmap = cv.resize(heatmap, (dst_size[0], dst_size[1]))
    return resize_heatmap


def inverse_resize(origin_image, resize_image, origin_size):
    (h, w) = origin_size
    if h > w:
        temp_img = cv.resize(resize_image, (h, h))
        w_delta = int(np.floor((h - w) / 2.0))
        origin_img = temp_img[:, w_delta: h - w_delta]
    else:
        temp_img = cv.resize(resize_image, (w, w))
        h_delta = int(np.floor((w - h) / 2.0))
        origin_img = temp_img[h_delta: w - h_delta, :]
    plt.figure()
    print(origin_img.shape)
    plt.subplot(1, 2, 1)
    plt.imshow(cv.cvtColor(origin_image, cv.COLOR_GRAY2RGB))
    plt.subplot(1, 2, 2)
    plt.imshow(cv.cvtColor(origin_img, cv.COLOR_GRAY2RGB))
    plt.show()
    return origin_img


def show_image_label(show_image, mnt_label, is_heatmaps=False):
    if is_heatmaps:
        show_image_rgb = cv.cvtColor(show_image, cv.COLOR_GRAY2RGB)
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(show_image_rgb)
        plt.subplot(1, 2, 2)
        plt.imshow(mnt_label)
        plt.show()
    else:
        show_image = resize_to_square(show_image, (384, 384))
        show_image_rgb = cv.cvtColor(show_image, cv.COLOR_GRAY2RGB)
        plt.figure()
        plt.imshow(show_image_rgb)
        plt.plot(mnt_label[0], mnt_label[1], '.r')
        plt.show()


def broad_repair(image, w, h):
    board = image[1][1]
    for i in range(h):
        image[i][w - 1] = board
        image[i][0] = board
    for j in range(w):
        image[h - 1][j] = board
        image[0][j] = board
    return image


class Resize(object):
    def __init__(self, output_size: tuple, theta_stride: int, dimension_choice: str):
        self.output_size = output_size
        self.theta_stride = theta_stride
        self.dimension_choice = dimension_choice

    def __call__(self, sample):
        # 图像
        origin_img = sample['image']
        mnt_x = sample['label'][0]
        # print(mnt_x)
        mnt_y = sample['label'][1]
        mnt_theta = sample['label'][2]
        mnt_theta = [int(i) for i in mnt_theta]
        (h, w) = origin_img.shape[:2]
        origin_img = broad_repair(origin_img, w, h)
        d0 = max(h, w)
        # print(sample['label'])
        if h > w:
            tran_img = cv.copyMakeBorder(origin_img, 0, 0, int(np.floor((h - w) / 2.0)), int(np.ceil((h - w) / 2.0)),
                                         cv.BORDER_REPLICATE)
            mnt_x = [int(i) + int(np.floor((h - w) / 2.0)) for i in mnt_x]
            mnt_y = [int(i) for i in mnt_y]
        else:
            tran_img = cv.copyMakeBorder(origin_img, int(np.floor((w - h) / 2.0)), int(np.ceil((w - h) / 2.0)), 0, 0,
                                         cv.BORDER_REPLICATE)
            mnt_x = [int(i) for i in mnt_x]
            mnt_y = [int(i) + int(np.floor((w - h) / 2.0)) for i in mnt_y]
        # print(tran_img.shape)
        resize_img = cv.resize(tran_img, (self.output_size[0], self.output_size[1]))
        mnt_x_ = []
        mnt_y_ = []
        # rgb_img_resize = cv.cvtColor(resize_img, cv.COLOR_GRAY2RGB)
        scale = self.output_size[0] * 1.0 / d0
        mnt_label_true = []
        for k in range(0, len(mnt_x)):
            x_ = int(mnt_x[k] * scale)
            y_ = int(mnt_y[k] * scale)
            mnt_x_.append(x_)
            mnt_y_.append(y_)
            mnt_temp_ = []
            mnt_temp_.append(x_)
            mnt_temp_.append(y_)
            mnt_temp_.append(mnt_theta[k])
            mnt_label_true.append(mnt_temp_)
        #     x2_ = np.round(math.sin(int(mnt_theta[k]) * math.pi / 180.0) * 15)
        #     y2_ = np.round(math.cos(int(mnt_theta[k]) * math.pi / 180.0) * 15)
        #     cv.arrowedLine(rgb_img_resize, (x_, y_),
        #                    (int(mnt_x[k] * scale + x2_), int(mnt_y[k] * scale + y2_)), (0, 0, 255), 2, 0, 0, 0.4)
        # cv.imshow("rgb_img_resize", rgb_img_resize)
        # cv.waitKey(0)
        # label_new = []
        # str_x = [str(x) for x in mnt_x_]
        # label_new.append(" ".join(str_x))
        # str_y = [str(x) for x in mnt_y_]
        # label_new.append(" ".join(str_y))
        # str_theta = [str(x) for x in mnt_theta]
        # label_new.append(" ".join(str_theta))
        if self.dimension_choice == '3D':
            N = int(360 / self.theta_stride)
            heatmaps = np.zeros((N, int(self.output_size[0] / 4), int(self.output_size[1] / 4)))
            # heatmaps = np.zeros((int(self.output_size[0] / 4), int(self.output_size[1] / 4)))
            for k in range(len(mnt_x_)):
                heatmaps = putGaussianMaps_ellipsoid((mnt_x_[k], mnt_y_[k], mnt_theta[k]), heatmaps, 1.0,
                                           int(self.output_size[0] / 4), int(self.output_size[1] / 4), 4, self.theta_stride)
                # heatmaps = putGaussianMaps((mnt_x_[k], mnt_y_[k]), heatmaps, 1.0,
                #                            int(self.output_size[0] / 4), int(self.output_size[1] / 4), 4, 20)
            print("heatmap_shape: ", heatmaps.shape)
            # show_image_label(resize_img, heatmaps, is_heatmaps=True)
            # show_heatmap_3D(heatmaps, N)
        else:
            heatmaps = np.zeros((int(self.output_size[0] / 4), int(self.output_size[1] / 4)))
            for k in range(len(mnt_x_)):
                # heatmaps = putGaussianMaps((mnt_x_[k], mnt_y_[k], mnt_theta[k]), heatmaps, 1.0,
                #                            int(self.output_size[0] / 4), int(self.output_size[1] / 4), 4, 20)
                heatmaps = putGaussianMaps((mnt_x_[k], mnt_y_[k]), heatmaps, 1.0,
                                           int(self.output_size[0] / 4), int(self.output_size[1] / 4), 4, self.theta_stride)
            # print("heatmap_shape: ", heatmaps.shape)
            # show_image_label(resize_img, heatmaps, is_heatmaps=True)
        return {'image': resize_img, 'label': heatmaps, 'label_list': mnt_label_true}


class ResizeVali(object):
    def __init__(self, output_size: tuple, theta_stride: int, dimension_choice: str):
        self.output_size = output_size
        self.theta_stride = theta_stride
        self.dimension_choice = dimension_choice

    def __call__(self, sample):
        # 图像
        origin_img = sample['image']
        mnt_x = sample['label'][0]
        # print(mnt_x)
        mnt_y = sample['label'][1]
        mnt_theta = sample['label'][2]
        mnt_theta = [int(i) for i in mnt_theta]
        (h, w) = origin_img.shape[:2]
        origin_img = broad_repair(origin_img, w, h)
        d0 = max(h, w)
        # print(sample['label'])
        if h > w:
            tran_img = cv.copyMakeBorder(origin_img, 0, 0, int(np.floor((h - w) / 2.0)), int(np.ceil((h - w) / 2.0)),
                                         cv.BORDER_REPLICATE)
            mnt_x = [int(i) + int(np.floor((h - w) / 2.0)) for i in mnt_x]
            mnt_y = [int(i) for i in mnt_y]
        else:
            tran_img = cv.copyMakeBorder(origin_img, int(np.floor((w - h) / 2.0)), int(np.ceil((w - h) / 2.0)), 0, 0,
                                         cv.BORDER_REPLICATE)
            mnt_x = [int(i) for i in mnt_x]
            mnt_y = [int(i) + int(np.floor((w - h) / 2.0)) for i in mnt_y]
        # print(tran_img.shape)
        resize_img = cv.resize(tran_img, (self.output_size[0], self.output_size[1]))
        mnt_x_ = []
        mnt_y_ = []
        mnt_label_true = []
        # rgb_img_resize = cv.cvtColor(resize_img, cv.COLOR_GRAY2RGB)
        scale = self.output_size[0] * 1.0 / d0
        for k in range(0, len(mnt_x)):
            x_ = int(mnt_x[k] * scale)
            y_ = int(mnt_y[k] * scale)
            mnt_temp_ = []
            mnt_temp_.append(x_)
            mnt_temp_.append(y_)
            mnt_temp_.append(mnt_theta[k])
            mnt_x_.append(x_)
            mnt_y_.append(y_)
            mnt_label_true.append(mnt_temp_)
        if self.dimension_choice == '3D':
            N = int(360 / self.theta_stride)
            heatmaps = np.zeros((N, int(self.output_size[0] / 4), int(self.output_size[1] / 4)))
            for k in range(len(mnt_x_)):
                heatmaps = putGaussianMaps((mnt_x_[k], mnt_y_[k], mnt_theta[k]), heatmaps, 1.0,
                                           int(self.output_size[0] / 4), int(self.output_size[1] / 4), 4, self.theta_stride)
                # heatmaps = putGaussianMaps((mnt_x_[k], mnt_y_[k]), heatmaps, 1.0,
                #                            int(self.output_size[0]), int(self.output_size[1]), 1, 0)
            # print("heatmap_shape: ", heatmaps.shape)
            # show_image_label(resize_img, heatmaps, is_heatmaps=True)
        else:
            heatmaps = np.zeros((int(self.output_size[0] / 4), int(self.output_size[1] / 4)))
            for k in range(len(mnt_x_)):
                # heatmaps = putGaussianMaps((mnt_x_[k], mnt_y_[k], mnt_theta[k]), heatmaps, 1.0,
                #                            int(self.output_size[0] / 4), int(self.output_size[1] / 4), 4, 20)
                heatmaps = putGaussianMaps((mnt_x_[k], mnt_y_[k]), heatmaps, 1.0,
                                           int(self.output_size[0]), int(self.output_size[1]), 1, 0)
            # print("heatmap_shape: ", heatmaps.shape)
            # show_image_label(resize_img, heatmaps, is_heatmaps=True)
        return {'image': resize_img, 'label': heatmaps, 'label_list': mnt_label_true}


class ToTensorVali(object):

    def __call__(self, sample):
        image = sample['image']
        tran = transforms.ToTensor()  # 将numpy数组或PIL.Image读的图片转换成(C,H, W)的Tensor格式且/255归一化到[0,1.0]之间
        img_tensor = tran(image).float()
        # label_t = np.expand_dims(sample['label'], 0)
        # print("label_t_shape: ", label_t.shape)
        return {'image': img_tensor, 'label': sample['label'], 'label_list': sample['label_list']}


# # 变换ToTensor
class ToTensor(object):

    def __call__(self, sample):
        image = sample['image']
        # print("image_shape: ", image.shape)
        tran = transforms.ToTensor()  # 将numpy数组或PIL.Image读的图片转换成(C,H, W)的Tensor格式且/255归一化到[0,1.0]之间
        image = np.expand_dims(image, 2)
        img_tensor = tran(image).float()
        # label_t = np.expand_dims(sample['label'], 0)
        # print("img_tensor_shape: ", img_tensor.shape)
        # print("label_shape: ", sample['label'].shape)
        return {'image': img_tensor, 'label': sample['label']}


class ShiftPro(object):
    def __call__(self, sample):
        origin_img = sample['image']
        mnt_x = np.array(sample['label'][0])
        # print(mnt_x)
        mnt_y = np.array(sample['label'][1])
        mnt_theta = np.array(sample['label'][2])
        (h, w) = origin_img.shape[:2]
        origin_img = broad_repair(origin_img, w, h)
        w_limit_low = -mnt_x.min()
        w_limit_high = w - mnt_x.max()
        h_limit_low = -mnt_y.min()
        h_limit_hight = h - mnt_y.max()
        if w_limit_high <= w_limit_low:
            w_limit_high = w_limit_low + 1
        if h_limit_hight <= h_limit_low:
            h_limit_hight = h_limit_low + 1
        shift_img = origin_img.copy()
        wshift = np.random.randint(w_limit_low, w_limit_high)
        hshift = np.random.randint(h_limit_low, h_limit_hight)
        # wshift = 0  # 自定义平移尺寸
        # hshift = 0  # 自定义平移尺寸
        for i in range(0, h):
            for j in range(0, w):
                if 0 <= (i - hshift) < h and 0 <= (j - wshift) < w:
                    shift_img[i][j] = origin_img[i - hshift][j - wshift]
                else:
                    shift_img[i][j] = origin_img[1][1]
        mnt_x_ = []
        mnt_y_ = []
        mnt_theta_ = []
        # rgb_img_shift = cv.cvtColor(shift_img, cv.COLOR_GRAY2RGB)
        for k in range(0, len(mnt_x)):
            x_ = mnt_x[k] + wshift
            y_ = mnt_y[k] + hshift
            if 0 <= x_ < h and 0 <= y_ < w:
                # x2_ = np.round(math.sin(mnt_theta[k] * math.pi / 180.0) * 15)
                # y2_ = np.round(math.cos(mnt_theta[k] * math.pi / 180.0) * 15)
                # cv.arrowedLine(rgb_img_shift, (int(x_), int(y_)), (int(x_ + x2_), int(y_ + y2_)), (0, 0, 255), 2, 0, 0, 0.4)
                mnt_x_.append(x_)
                mnt_y_.append(y_)
                mnt_theta_.append(mnt_theta[k])
        # cv.imshow("rgb_img_shift", rgb_img_shift)
        # cv.waitKey(0)
        new_label = []
        new_label.append(mnt_x_)
        new_label.append(mnt_y_)
        new_label.append(mnt_theta_)
        return {'image': shift_img, 'label': new_label}


class FlipVertical(object):
    def __call__(self, sample):
        origin_img = sample['image']
        mnt_x = sample['label'][0]
        # print(mnt_x)
        mnt_y = sample['label'][1]
        mnt_theta = sample['label'][2]
        (h, w) = origin_img.shape[:2]
        flip_vertical_img = origin_img.copy()
        for i in range(0, h):
            for j in range(0, w):
                flip_vertical_img[i][j] = origin_img[h - 1 - i][j]
        mnt_x_ = []
        mnt_y_ = []
        mnt_theta_ = []
        # rgb_img_flip_vertical = cv.cvtColor(flip_vertical_img, cv.COLOR_GRAY2RGB)
        for k in range(0, len(mnt_x)):
            x_ = mnt_x[k]
            y_ = h - 1 - mnt_y[k]
            # x2_ = np.round(math.sin((180 - mnt_theta[k]) * math.pi / 180.0) * 15)
            # y2_ = np.round(math.cos((180 - mnt_theta[k]) * math.pi / 180.0) * 15)
            # cv.arrowedLine(rgb_img_flip_vertical, (int(x_), int(y_)), (int(x_ + x2_), int(y_ + y2_)), (0, 0, 255), 2, 0,
            #                0, 0.4)
            mnt_x_.append(x_)
            mnt_y_.append(y_)
            mnt_theta_.append((180 - mnt_theta[k]) % 360)
        # cv.imshow("rgb_img_flip_vertical", rgb_img_flip_vertical)
        # cv.waitKey(0)
        new_label = []
        new_label.append(mnt_x_)
        new_label.append(mnt_y_)
        new_label.append(mnt_theta_)
        return {'image': flip_vertical_img, 'label': new_label}


class FlipLevel(object):
    def __call__(self, sample):
        origin_img = sample['image']
        mnt_x = sample['label'][0]
        # print(mnt_x)
        mnt_y = sample['label'][1]
        mnt_theta = sample['label'][2]
        (h, w) = origin_img.shape[:2]
        flip_level_img = origin_img.copy()
        for i in range(0, h):
            for j in range(0, w):
                flip_level_img[i][j] = origin_img[i][w - 1 - j]
        mnt_x_ = []
        mnt_y_ = []
        mnt_theta_ = []
        # rgb_img_flip_level = cv.cvtColor(flip_level_img, cv.COLOR_GRAY2RGB)
        for k in range(0, len(mnt_x)):
            x_ = w - 1 - mnt_x[k]
            y_ = mnt_y[k]
            # x2_ = np.round(math.sin((- mnt_theta[k]) * math.pi / 180.0) * 15)
            # y2_ = np.round(math.cos((- mnt_theta[k]) * math.pi / 180.0) * 15)
            # cv.arrowedLine(rgb_img_flip_level, (int(x_), int(y_)), (int(x_ + x2_), int(y_ + y2_)), (0, 0, 255), 2, 0, 0,
            #                0.4)
            mnt_x_.append(x_)
            mnt_y_.append(y_)
            mnt_theta_.append((- mnt_theta[k]) % 360)
        # cv.imshow("rgb_img_flip_level", rgb_img_flip_level)
        # cv.waitKey(0)
        new_label = []
        new_label.append(mnt_x_)
        new_label.append(mnt_y_)
        new_label.append(mnt_theta_)
        return {'image': flip_level_img, 'label': new_label}


class RotatePro(object):
    def __init__(self, angle=0, israndom=True):
        if israndom:
            self.angle = np.random.randint(-10, 10)
        else:
            self.angle = angle

    def __call__(self, sample):
        origin_img = sample['image']
        mnt_x = sample['label'][0]
        # print(mnt_x)
        mnt_y = sample['label'][1]
        mnt_theta = sample['label'][2]
        (h, w) = origin_img.shape[:2]
        origin_img = broad_repair(origin_img, w, h)
        center = (w / 2.0, h / 2.0)
        M = cv.getRotationMatrix2D(center, self.angle, 1.0)
        broad = int(origin_img[1][1])
        rotated_img = cv.warpAffine(origin_img, M, (w, h), borderValue=broad)  # 根据旋转矩阵进行仿射变换
        mnt_x_ = []
        mnt_y_ = []
        mnt_theta_ = []
        # rgb_img_orien = cv.cvtColor(rotated_img, cv.COLOR_GRAY2RGB)
        alpha = math.cos(self.angle * math.pi / 180.0)
        beita = math.sin(self.angle * math.pi / 180.0)
        b1 = (1.0 - alpha) * w / 2.0 - beita * h / 2.0
        b2 = beita * w / 2.0 + (1.0 - alpha) * h / 2.0
        for k in range(0, len(mnt_x)):
            x_ = alpha * mnt_x[k] + beita * mnt_y[k] + b1
            y_ = - beita * mnt_x[k] + alpha * mnt_y[k] + b2
            # x2_ = np.round(math.sin((mnt_theta[k] + self.angle) * math.pi / 180.0) * 15)
            # y2_ = np.round(math.cos((mnt_theta[k] + self.angle) * math.pi / 180.0) * 15)
            # cv.arrowedLine(rgb_img_orien, (int(x_), int(y_)), (int(x_ + x2_), int(y_ + y2_)), (0, 0, 255), 2, 0, 0,
            #                0.4)
            if 0 <= x_ < h and 0 <= y_ < w:
                mnt_x_.append(x_)
                mnt_y_.append(y_)
                mnt_theta_.append((mnt_theta[k] + self.angle) % 360)
        # cv.imshow("rgb_img_orien", rgb_img_orien)
        # cv.waitKey(0)
        new_label = []
        new_label.append(mnt_x_)
        new_label.append(mnt_y_)
        new_label.append(mnt_theta_)
        return {'image': rotated_img, 'label': new_label}


class RandomApply(object):
    def __init__(self, transform, probability):
        self.transform = transform
        self.probability = probability

    def __call__(self, sample):
        if float(torch.rand(1).item()) > self.probability:
            return sample
        return self.transform(sample)


class Normalize(object):

    def __call__(self, sample):
        image = sample['image']
        tran = transforms.Normalize(0.5, 0.5)  # 将numpy数组或PIL.Image读的图片转换成(C,H, W)的Tensor格式且/255归一化到[0,1.0]之间
        img_tensor = tran(image).float()
        # print("img_tensor_shape: ", img_tensor.shape)
        return {'image': img_tensor, 'label': sample['label']}

if __name__ == "__main__":
    '''
    检查ground truth和transform每一步的ground truth
    '''
    mnt_data_dir = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # 上上上级目录
    TRAIN_SET = 'train_data_random.csv'
    TEST_SET = 'test_data_random.csv'
    transform = transforms.Compose([RandomApply(ShiftPro(), 0.5), RandomApply(FlipLevel(), 0.5),
                                RandomApply(FlipVertical(), 0.5), RandomApply(RotatePro(), 0.5),
                                Resize((384, 384), 10, '3D'), ToTensor()])
    train_dataset = MyDatasets(root_dir=mnt_data_dir,
                               names_file=TRAIN_SET,
                               transform=transform)
    for (index, data) in enumerate(train_dataset):
        image = data['image']
        label = data['label']
        print("image shape:", image.shape)
        print("label shape:", np.array(label).shape)
