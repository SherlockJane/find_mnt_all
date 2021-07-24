# coding=utf-8

import random
import sys

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

"""Implement the generate of every channel of ground truth heatmap.
:param centerA: int with shape (2,), every coordinate of person's keypoint.
:param accumulate_confid_map: one channel of heatmap, which is accumulated, 
       np.log(100) is the max value of heatmap.
:param params_transform: store the value of stride and crop_szie_y, crop_size_x                 
"""


def putGaussianMaps(center, accumulate_confid_map, sigma, grid_y, grid_x, stride, theta_stride):
    # start = stride / 2.0 - 0.5
    y_range = [i for i in range(int(grid_y))]
    x_range = [i for i in range(int(grid_x))]
    xx, yy = np.meshgrid(x_range, y_range)
    # xx = xx * stride + start
    # yy = yy * stride + start
    new_x = int(np.floor(center[0]/stride))
    new_y = int(np.floor(center[1]/stride))
    d2 = (xx - new_x) ** 2 + (yy - new_y) ** 2
    exponent = d2 / 2.0 / sigma / sigma
    mask = exponent <= 4.6052  # 这个常数值是指什么？ 大于这个常数值的都清零了   距离太远的为零
    cofid_map = np.exp(-exponent)
    cofid_map = np.multiply(mask, cofid_map)
    theta_index = int(np.floor(center[2]/theta_stride))
    accumulate_confid_map[theta_index] += cofid_map
    accumulate_confid_map += cofid_map
    N = int(360 / 20)
    if 0 <= new_x < 96 and 0 <= new_y < 96:
        for k in range(1, int(sigma) * 3 + 1):
            accumulate_confid_map[(theta_index + k) % N][new_y][new_x] += np.exp(-(k * k) / 2.0 / sigma / sigma)
            accumulate_confid_map[(theta_index - k) % N][new_y][new_x] += np.exp(-(k * k) / 2.0 / sigma / sigma)
    accumulate_confid_map[accumulate_confid_map > 1.0] = 1.0
    # show_heatmap(accumulate_confid_map, 18)
    return accumulate_confid_map


def putGaussianMaps_cylinder(center, accumulate_confid_map, sigma, grid_y, grid_x, stride, theta_stride):
    # start = stride / 2.0 - 0.5
    y_range = [i for i in range(int(grid_y))]
    x_range = [i for i in range(int(grid_x))]
    xx, yy = np.meshgrid(x_range, y_range)
    # xx = xx * stride + start
    # yy = yy * stride + start
    new_x = int(np.floor(center[0]/stride))
    new_y = int(np.floor(center[1]/stride))
    d2 = (xx - new_x) ** 2 + (yy - new_y) ** 2
    exponent = d2 / 2.0 / sigma / sigma
    mask = exponent <= 4.6052  # 这个常数值是指什么？ 大于这个常数值的都清零了   距离太远的为零
    cofid_map = np.exp(-exponent)
    cofid_map = np.multiply(mask, cofid_map)
    theta_index = int(np.floor(center[2]/theta_stride))
    accumulate_confid_map[theta_index] += cofid_map
    # print("heatmap_data_center: ", accumulate_confid_map[theta_index])
    N = int(360 / 10)
    if 0 <= new_x < 96 and 0 <= new_y < 96:
        for k in range(1, int(sigma) * 3 + 1):
            length_map = np.exp(-(k * k) / 2.0 / sigma / sigma)
            accumulate_confid_map[(theta_index + k) % N] += np.multiply(cofid_map, length_map)
            accumulate_confid_map[(theta_index - k) % N] += np.multiply(cofid_map, length_map)
            # print("heatmap_data_length: ", accumulate_confid_map[(theta_index + k) % N].max())
            # print("heatmap_data_length: ", accumulate_confid_map[(theta_index - k) % N].max())
    accumulate_confid_map[accumulate_confid_map > 1.0] = 1.0

    return accumulate_confid_map


def putGaussianMaps_ellipsoid(center, accumulate_confid_map, sigma, grid_y, grid_x, stride, theta_stride):
    # start = stride / 2.0 - 0.5
    y_range = [i for i in range(int(grid_y))]
    x_range = [i for i in range(int(grid_x))]
    xx, yy = np.meshgrid(x_range, y_range)
    # xx = xx * stride + start
    # yy = yy * stride + start
    new_x = int(np.floor(center[0]/stride))
    new_y = int(np.floor(center[1]/stride))
    d2 = (xx - new_x) ** 2 + (yy - new_y) ** 2
    exponent = d2 / 2.0 / sigma / sigma
    mask = exponent <= 4.6052  # 这个常数值是指什么？ 大于这个常数值的都清零了   距离太远的为零
    cofid_map = np.exp(-exponent)
    cofid_map = np.multiply(mask, cofid_map)
    theta_index = int(np.floor(center[2]/theta_stride))
    accumulate_confid_map[theta_index] += cofid_map
    # print("heatmap_data_center: ", accumulate_confid_map[theta_index])
    N = int(360 / 10)
    if 0 <= new_x < 96 and 0 <= new_y < 96:
        for k in range(1, int(sigma) * 3 + 1):
            length_map = np.exp(-(k * k) / 2.0 / sigma / sigma)
            accumulate_confid_map[(theta_index + k) % N] += np.multiply(cofid_map, length_map)
            accumulate_confid_map[(theta_index - k) % N] += np.multiply(cofid_map, length_map)
            # print("heatmap_data_length: ", accumulate_confid_map[(theta_index + k) % N].max())
            # print("heatmap_data_length: ", accumulate_confid_map[(theta_index - k) % N].max())
    accumulate_confid_map[accumulate_confid_map > 1.0] = 1.0
    accumulate_confid_map[accumulate_confid_map < 0.01] = 0.0
    return accumulate_confid_map


def show_heatmap_3D(heatmaps, N):
    print("heatmap_data: ", heatmaps.min())
    xyz = np.array(np.nonzero(heatmaps)[::-1])
    print(xyz)
    x1 = xyz[0]
    y1 = xyz[1]
    z1 = xyz[2]
    fig = plt.figure()
    ax = Axes3D(fig)
    cmap = cm.coolwarm
    ax.scatter(x1, y1, z1, s=5, antialiased=False, alpha=0.4, cmap=cmap(heatmaps))

    # 添加坐标轴(顺序是Z, Y, X)
    ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
    ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
    ax.set(xlabel='X',
           ylabel='Y',
           zlabel='Z',
           zticks=np.arange(0, N, 1)
           )
    # 投影到坐标上
    ax.scatter(0, y1, z1, zdir='z', c='green', s=5, cmap=cmap)  # 投影在 yz 平面
    ax.scatter(x1, 0, z1, zdir='z', c='gray', s=5, cmap=cmap)  # 投影在 xz 平面
    ax.scatter(x1, y1, 0, zdir='z', c='cyan', s=5, cmap=cmap)  # 投影在 xy 平面

    # 调整视角
    ax.view_init(elev=20,  # 仰角
                 azim=45  # 方位角
                 )
    #
    # # 添加右侧的色卡条
    # fig.colorbar(fig, shrink=0.6, aspect=8)  # shrink表示整体收缩比例，aspect仅对bar的宽度有影响，aspect值越大，bar越窄
    plt.show()


def show_heatmap(img, label, heatmaps):
    # xyz = np.array(np.nonzero(heatmaps)[::-1])305
    # print(xyz)
    # x1 = xyz[0]
    # y1 = xyz[1]
    # z1 = xyz[2] 305
    # 绘制散点图
    # figure, axes = plt.subplots(2, 1, figsize=(18, 12), dpi=100)
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(cv.cvtColor(img, cv.COLOR_GRAY2RGB))
    plt.subplot(1, 3, 2)
    # plt.imshow(np.sum(heatmaps, axis=0)) 305
    plt.imshow(label)
    plt.subplot(1, 3, 3)
    plt.imshow(heatmaps)
    plt.show()

    # fig = plt.figure() 305
    # ax = Axes3D(fig)
    # ax.scatter(x1, y1, z1, s=5)
    #
    # # 添加坐标轴(顺序是Z, Y, X)
    # ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
    # ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    # ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
    # ax.set(xlabel='X',
    #        ylabel='Y',
    #        zlabel='Z',
    #        zticks=np.arange(0, N, 1)
    #        )
    # # 投影到坐标上
    # ax.scatter(0, y1, z1, zdir='z', c='green', s=5)  # 投影在 yz 平面
    # ax.scatter(x1, 0, z1, zdir='z', c='gray', s=5)  # 投影在 xz 平面
    # ax.scatter(x1, y1, 0, zdir='z', c='cyan', s=5)  # 投影在 xy 平面
    #
    # # 调整视角
    # ax.view_init(elev=20,  # 仰角
    #              azim=45  # 方位角
    #              )
    # #
    # # # 添加右侧的色卡条
    # # fig.colorbar(fig, shrink=0.6, aspect=8)  # shrink表示整体收缩比例，aspect仅对bar的宽度有影响，aspect值越大，bar越窄
    # plt.show() 305