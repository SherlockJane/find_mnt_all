import os
import csv
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import cv2 as cv
import math
from torchvision import transforms
import sys

now_dir = os.path.abspath(os.getcwd())
lib_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))  # 上级目录
mnt_test_dir = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # 上上级目录


class MyDatasets(Dataset):
    """The class to load the dataset"""
    def __init__(self, root_dir, names_file, transform=None, train_aug=True):
        self.root_dir = root_dir
        self.names_file = os.path.join(root_dir, names_file)
        self.transform = transform
        self.size = 0
        self.names_list = []

        if not os.path.isfile(self.names_file):
            print(self.names_file + "does not exist!")
        with open(os.path.join(self.root_dir, names_file)) as f:
            file = csv.reader(f)
            headers = next(file)
            for row in file:
                self.names_list.append(row)
                self.size += 1

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        image_path = os.path.abspath(os.path.join(self.root_dir, self.names_list[idx][3]))
        if not os.path.isfile(image_path):
            print(image_path + 'does not exist!')
            return None
        img = Image.open(image_path)
        image = np.array(img)
        mnt_all = self.names_list[idx][2]
        mnt_x = []
        mnt_y = []
        mnt_theta = []
        mnt_all = mnt_all.replace("[", "").replace("]", "").replace("'", "").split(", ")
        # print(mnt_all)
        for k in range(0, len(mnt_all)):
            mnt_temp = mnt_all[k].split(" ")
            mnt_x.append(int(mnt_temp[1]))
            mnt_y.append(int(mnt_temp[2]))
            mnt_theta.append(int(mnt_temp[3]))
        label = []
        label.append(mnt_x)
        label.append(mnt_y)
        label.append(mnt_theta)
        # print(image)
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample


# transform = transforms.Compose([ShiftPro(), FlipLevel(), FlipVertical(), RotatePro(),
#                                 Resize((384, 384)), ToTensor()])  # 三个通道
# train_dataset = MyDatasets(root_dir=mnt_test_dir,
#                            names_file='train_data_random.csv',
#                            transform=transform)
# # for (cnt,i) in enumerate(train_dataset):
# #     image = i['image']
# #     label = i['label']
# #
# #     cv.imshow("origin", image)
# #     print(label)
# #     cv.waitKey(0)
#
# train_loader = DataLoader(dataset=train_dataset,
#                                  batch_size=10,
#                                  shuffle=True)
# model = my_model.get_model()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# loss_func = torch.nn.MSELoss(reduction='mean')
# for i_batch, sample_batch in enumerate(train_loader):
#     images_batch, labels_batch = sample_batch['image'], sample_batch['label']
#     print("labels_batch_shape: ", labels_batch.shape)
#     out = model(images_batch)
#     labels_batch = labels_batch.float()
#     loss = loss_func(out, labels_batch)
#
#     optimizer.zero_grad()  # 清除梯度
#     loss.backward()  # 误差反向传播
#     optimizer.step()  # 优化器开始工作
