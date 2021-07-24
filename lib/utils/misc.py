""" Additional utility functions. """
import os
import time
import math
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import random
from scipy.ndimage.filters import gaussian_filter, maximum_filter
from scipy.ndimage.morphology import generate_binary_structure


def ensure_path(path):
    """The function to make log path.
    Args:
      path: the generated saving path.
    """
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)
        print("---  There is this folder for logs! %s ---", path)


class Timer():
    """The class for timer."""
    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)


class Averager():
    """The class to calculate the average."""
    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


def find_peaks(param, img):
    """
    Given a (grayscale) image, find local maxima whose value is above a given
    threshold (param['thre1'])
    :param img: Input image (2d array) where we want to find peaks
    :return: 2d np.array containing the [x,y] coordinates of each peak found
    in the image
    """

    peaks_binary = (maximum_filter(img, footprint=generate_binary_structure(
        2, 1)) == img) * (img > param)  # 局部最大滤波器检测到峰值
    # Note reverse ([::-1]): we return [[x y], [x y]...] instead of [[y x], [y
    # x]...]
    return np.array(np.nonzero(peaks_binary)[::-1]).T  # 非零元素的位置


def compute_resized_coords(coords, resizeFactor):
    """
    Given the index/coordinates of a cell in some input array (e.g. image),
    provides the new coordinates if that array was resized by making it
    resizeFactor times bigger.
    E.g.: image of size 3x3 is resized to 6x6 (resizeFactor=2), we'd like to
    know the new coordinates of cell [1,2] -> Function would return [2.5,4.5]
    :param coords: Coordinates (indices) of a cell in some input array
    :param resizeFactor: Resize coefficient = shape_dest/shape_source. E.g.:
    resizeFactor=2 means the destination array is twice as big as the
    original one
    :return: Coordinates in an array of size
    shape_dest=resizeFactor*shape_source, expressing the array indices of the
    closest point to 'coords' if an image of size shape_source was resized to
    shape_dest
    """

    # 1) Add 0.5 to coords to get coordinates of center of the pixel (e.g.
    # index [0,0] represents the pixel at location [0.5,0.5])
    # 2) Transform those coordinates to shape_dest, by multiplying by resizeFactor
    # 3) That number represents the location of the pixel center in the new array,
    # so subtract 0.5 to get coordinates of the array index/indices (revert
    # step 1)
    return (np.array(coords, dtype=float) + 0.5) * resizeFactor - 0.5


def NMS(heatmaps, peak_thread, upsampFactor=1., bool_refine_center=True, bool_gaussian_filt=False, config=None):
    """
    NonMaximaSuppression: find peaks (local maxima) in a set of grayscale images
    :param heatmaps: set of grayscale images on which to find local maxima (3d np.array,
    with dimensions image_height x image_width x num_heatmaps)
    :param upsampFactor: Size ratio between CPM heatmap output and the input image size.
    Eg: upsampFactor=16 if original image was 480x640 and heatmaps are 30x40xN
    :param bool_refine_center: Flag indicating whether:
     - False: Simply return the low-res peak found upscaled by upsampFactor (subject to grid-snap)
     - True: (Recommended, very accurate) Upsample a small patch around each low-res peak and
     fine-tune the location of the peak at the resolution of the original input image
    :param bool_gaussian_filt: Flag indicating whether to apply a 1d-GaussianFilter (smoothing)
    to each upsampled patch before fine-tuning the location of each peak.
    :return: a NUM_JOINTS x 4 np.array where each row represents a joint type (0=nose, 1=neck...)
    and the columns indicate the {x,y} position, the score (probability) and a unique id (counter)
    """
    # MODIFIED BY CARLOS: Instead of upsampling the heatmaps to heatmap_avg and
    # then performing NMS to find peaks, this step can be sped up by ~25-50x by:
    # (9-10ms [with GaussFilt] or 5-6ms [without GaussFilt] vs 250-280ms on RoG
    # 1. Perform NMS at (low-res) CPM's output resolution
    # 1.1. Find peaks using scipy.ndimage.filters.maximum_filter
    # 2. Once a peak is found, take a patch of 5x5 centered around the peak, upsample it, and
    # fine-tune the position of the actual maximum.
    #  '-> That's equivalent to having found the peak on heatmap_avg, but much faster because we only
    #      upsample and scan the 5x5 patch instead of the full (e.g.) 480x640

    mnt_NMS = []
    cnt_total_joints = 0
    angle_index = np.argmax(heatmaps, axis=0)
    heatmap_loc = np.max(heatmaps, axis=0)
    # For every peak found, win_size specifies how many pixels in each
    # direction from the peak we take to obtain the patch that will be
    # upsampled. Eg: win_size=1 -> patch is 3x3; win_size=2 -> 5x5
    # (for BICUBIC interpolation to be accurate, win_size needs to be >=2!)
    win_size = 2
    peak_coords = find_peaks(peak_thread, heatmap_loc)  # 返回局部最大值，以坐标的形式
    peaks = np.zeros((len(peak_coords), 4))
    for i, peak in enumerate(peak_coords):
        if bool_refine_center:
            x_min, y_min = np.maximum(0, peak - win_size)
            x_max, y_max = np.minimum(
                np.array(heatmap_loc.T.shape) - 1, peak + win_size)

            # Take a small patch around each peak and only upsample that
            # tiny region
            patch = heatmap_loc[y_min:y_max + 1, x_min:x_max + 1]
            map_upsamp = cv.resize(
                patch, None, fx=upsampFactor, fy=upsampFactor, interpolation=cv.INTER_CUBIC)

            # Gaussian filtering takes an average of 0.8ms/peak (and there might be
            # more than one peak per joint!) -> For now, skip it (it's
            # accurate enough)
            map_upsamp = gaussian_filter(
                map_upsamp, sigma=3) if bool_gaussian_filt else map_upsamp

            # Obtain the coordinates of the maximum value in the patch
            location_of_max = np.unravel_index(
                map_upsamp.argmax(), map_upsamp.shape)
            # Remember that peaks indicates [x,y] -> need to reverse it for
            # [y,x]
            location_of_patch_center = compute_resized_coords(
                peak[::-1] - [y_min, x_min], upsampFactor)
            # Calculate the offset wrt to the patch center where the actual
            # maximum is
            refined_center = (location_of_max - location_of_patch_center)
            peak_score = map_upsamp[location_of_max]
        else:
            refined_center = [0, 0]
            # Flip peak coordinates since they are [x,y] instead of [y,x]
            peak_score = heatmap_loc[tuple(peak[::-1])]
        peaks[i, :] = tuple(
            x for x in compute_resized_coords(peak_coords[i], upsampFactor) + refined_center[::-1]) + (
                       0.0, peak_score)
        peaks[i, 2] = angle_index[int(peak[0]), int(peak[1])]
        cnt_total_joints += 1
    mnt_NMS.append(peaks)
    return mnt_NMS


def mnt_map_to_mnt(mnt_map, origin_size):
    mnt = []
    (h, w) = origin_size
    mnt_list_origin_size = []
    # print("mnt_map: ", mnt_map)
    for mnt_type, mnt_peaks in enumerate(mnt_map):
        for peak in mnt_peaks:
            mnt_temp = []
            mnt_x = int(peak[0])
            mnt_y = int(peak[1])
            mnt_theta = (int(peak[2]) + 0.5) * 20
            mnt_property = peak[3]
            mnt_temp.append(mnt_x)
            mnt_temp.append(mnt_y)
            mnt_temp.append(mnt_theta)
            mnt_temp.append(mnt_property)
            mnt.append(mnt_temp)
    return mnt, mnt_list_origin_size


def sort(mnt_list):
    y_temp = []
    for k in range(len(mnt_list)):
        y_temp.append(-mnt_list[k][3])
    index_proper = np.argsort(y_temp)
    mnt_list_sort = mnt_list[index_proper]
    return mnt_list_sort


def mnt_metric(mnt_list, label_list, x_range, y_range, theta_range, based_property=False, shuffle=True):
    if based_property:
        mnt_list = sort(mnt_list)
    if shuffle:
        random.shuffle(mnt_list)
        random.shuffle(label_list)
    mnt_list = list(mnt_list)
    label_list = label_list
    true_label_num = 0
    true_label_list = []
    if not mnt_list or not label_list:
        print("this is an empty list")
        return true_label_num, true_label_list
    for i in range(len(mnt_list)):
        for j in range(len(label_list)):
            if not label_list:
                print("stop metric")
                return true_label_num, true_label_list
            if abs(mnt_list[i][0] - label_list[j][0]) < x_range and \
               abs(mnt_list[i][1] - label_list[j][1]) < y_range and \
               abs(mnt_list[i][2] - label_list[j][2]) < theta_range:
                true_label_list.append(mnt_list[i])
                true_label_num += 1
                label_list.pop(j)
                break
    return true_label_num, true_label_list


def show_NMF(image, mnt_list, mnt_label_list, heatmaps, heatmap_true, true_label_list):
    label_x = []
    label_y = []
    label_theta = []
    predict_x = []
    predict_y = []
    predict_theta = []
    true_x = []
    true_y = []
    true_theta = []
    for i in range(len(mnt_label_list)):
        label_x.append(mnt_label_list[i][0])
        label_y.append(mnt_label_list[i][1])
        label_theta.append(mnt_label_list[i][2])
    for i in range(len(mnt_list)):
        predict_x.append(mnt_list[i][0])
        predict_y.append(mnt_list[i][1])
        predict_theta.append(mnt_list[i][2])
    for i in range(len(true_label_list)):
        true_x.append(true_label_list[i][0])
        true_y.append(true_label_list[i][1])
        true_theta.append(true_label_list[i][2])

    rgb_img = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
    for k in range(0, len(label_x)):
        x2 = np.round(math.sin(label_theta[k] * math.pi / 180.0) * 15)
        y2 = np.round(math.cos(label_theta[k] * math.pi / 180.0) * 15)
        cv.arrowedLine(rgb_img, (int(label_x[k]), int(label_y[k])), (int(label_x[k] + x2), int(label_y[k] + y2)), (0, 255, 0),
                       2, 0, 0, 0.4)
    for k in range(0, len(predict_x)):
        x2 = np.round(math.sin(predict_theta[k] * math.pi / 180.0) * 15)
        y2 = np.round(math.cos(predict_theta[k] * math.pi / 180.0) * 15)
        cv.arrowedLine(rgb_img, (int(predict_x[k]), int(predict_y[k])), (int(predict_x[k] + x2), int(predict_y[k] + y2)), (0, 0, 255),
                       2, 0, 0, 0.4)
    for k in range(0, len(true_x)):
        x2 = np.round(math.sin(true_theta[k] * math.pi / 180.0) * 15)
        y2 = np.round(math.cos(true_theta[k] * math.pi / 180.0) * 15)
        cv.arrowedLine(rgb_img, (int(true_x[k]), int(true_y[k])), (int(true_x[k] + x2), int(true_y[k] + y2)), (255, 0, 0),
                       2, 0, 0, 0.4)
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(cv.cvtColor(image, cv.COLOR_GRAY2RGB))
    plt.plot(label_x, label_y, '.g', label='Ground Truth')  # groundtruth
    plt.plot(predict_x, predict_y, '.b', label='predict')  # 预测的点
    plt.plot(true_x, true_y, '.r', label='true of predict')  # 预测点中正确的点
    plt.title("result_image", y=-0.3)
    plt.legend(loc=3, bbox_to_anchor=(1.1, 1.1), borderaxespad=0.)
    plt.subplot(1, 3, 2)
    plt.title("arrow_image", y=-0.3)
    plt.imshow(rgb_img)
    # plt.subplot(1, 3, 2)
    # plt.title("predict_heatmaps", y=-0.3)
    # plt.imshow(heatmaps)
    # plt.subplot(1, 3, 3)
    # plt.title("groundtruth_heatmaps", y=-0.3)
    # plt.imshow(heatmap_true)
    plt.show()


def save_error_vali(image, mnt_list, mnt_label_list, heatmaps, heatmap_true, true_label_list, image_path):
    label_x = []
    label_y = []
    label_theta = []
    predict_x = []
    predict_y = []
    predict_theta = []
    true_x = []
    true_y = []
    true_theta = []
    for i in range(len(mnt_label_list)):
        label_x.append(mnt_label_list[i][0])
        label_y.append(mnt_label_list[i][1])
        label_theta.append(mnt_label_list[i][2])
    for i in range(len(mnt_list)):
        predict_x.append(mnt_list[i][0])
        predict_y.append(mnt_list[i][1])
        predict_theta.append(mnt_list[i][2])
    for i in range(len(true_label_list)):
        true_x.append(true_label_list[i][0])
        true_y.append(true_label_list[i][1])
        true_theta.append(true_label_list[i][2])

    rgb_img = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
    for k in range(0, len(label_x)):
        x2 = np.round(math.sin(label_theta[k] * math.pi / 180.0) * 15)
        y2 = np.round(math.cos(label_theta[k] * math.pi / 180.0) * 15)
        cv.arrowedLine(rgb_img, (int(label_x[k]), int(label_y[k])), (int(label_x[k] + x2), int(label_y[k] + y2)),
                       (0, 255, 0),
                       2, 0, 0, 0.4)
    for k in range(0, len(predict_x)):
        x2 = np.round(math.sin(predict_theta[k] * math.pi / 180.0) * 15)
        y2 = np.round(math.cos(predict_theta[k] * math.pi / 180.0) * 15)
        cv.arrowedLine(rgb_img, (int(predict_x[k]), int(predict_y[k])),
                       (int(predict_x[k] + x2), int(predict_y[k] + y2)), (0, 0, 255),
                       2, 0, 0, 0.4)
    for k in range(0, len(true_x)):
        x2 = np.round(math.sin(true_theta[k] * math.pi / 180.0) * 15)
        y2 = np.round(math.cos(true_theta[k] * math.pi / 180.0) * 15)
        cv.arrowedLine(rgb_img, (int(true_x[k]), int(true_y[k])), (int(true_x[k] + x2), int(true_y[k] + y2)),
                       (255, 0, 0),
                       2, 0, 0, 0.4)

    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(cv.cvtColor(image, cv.COLOR_GRAY2RGB))
    plt.plot(label_x, label_y, '.g', label='Ground Truth')  # groundtruth
    plt.plot(predict_x, predict_y, '.b', label='predict')  # 预测的点
    plt.plot(true_x, true_y, '.r', label='true of predict')  # 预测点中正确的点
    plt.title("result_image", y=-0.3)
    plt.legend(loc=3, bbox_to_anchor=(1.1, 1.1), borderaxespad=0.)
    plt.subplot(1, 3, 2)
    plt.title("arrow_image", y=-0.3)
    plt.imshow(rgb_img)
    # plt.subplot(1, 3, 2)
    # plt.title("predict_heatmaps", y=-0.3)
    # plt.imshow(heatmaps)
    # plt.subplot(1, 3, 3)
    # plt.title("groundtruth_heatmaps", y=-0.3)
    # plt.imshow(heatmap_true)
    plt.savefig(image_path)
    plt.close('all')



# def count_acc(predict, label):
#     """The function to calculate the .
#     Args:
#       logits: input logits.
#       label: ground truth labels.
#     Return:
#       The output accuracy.
#     """
#     """更改成匹配函数！！！！！！！！！"""
#     # pred = F.softmax(predict, dim=1).argmax(dim=1)
#     # if torch.cuda.is_available():
#     #     return (pred == label).type(torch.cuda.FloatTensor).mean().item()
#     # return (pred == label).type(torch.FloatTensor).mean().item()
#     mnt_map = NMS(heatmaps, NMS_thread, upsampFactor=4)
#     mnt_list, mnt_list_origin_size = mnt_map_to_mnt(mnt_map, args.origin_size)
#     true_label_num, true_label_list = mnt_metric(mnt_list, mnt_label_groundtruth, 12, 12, 12, based_property=False,
#                                                  shuffle=True)


def calculate_recog(args, data, label, label_list, predict_data):
    mnt_groundtruth_num = len(label_list)
    heatmaps = predict_data.cpu().data.numpy()[0]
    mnt_map = NMS(heatmaps, args.NMS_thread, upsampFactor=4)
    mnt_list, mnt_list_origin_size = mnt_map_to_mnt(mnt_map, args.origin_size)
    true_label_num, true_label_list = mnt_metric(mnt_list, label_list, 20, 20, 40, based_property=False, shuffle=True)
    if len(mnt_list) > 0 and mnt_groundtruth_num > 0 and true_label_num > 0:
        acc = true_label_num * 1.0 / len(mnt_list)
        recall = true_label_num * 1.0 / mnt_groundtruth_num
        F1 = 2 * acc * recall / (acc + recall)
    else:
        acc = 0
        recall = 0
        F1 = 0
    print(
        "true_label_num: %d,  mnt_label_true: %d, predict_mnt_lable: %d "
        % (true_label_num, mnt_groundtruth_num, len(mnt_list)))
    image = data.cpu().data.numpy()[0][0]
    # print("iamge_shape: ", image.shape)
    heatmap_true = label.cpu().data.numpy()[0]
    # print("heatmap_shape: ", heatmap_true.shape)
    show_NMF(image, mnt_list, label_list, heatmaps, heatmap_true, true_label_list)
    return acc, recall, F1
