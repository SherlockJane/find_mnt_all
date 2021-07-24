import csv
import os
import random
headers = ['image_name', 'mnt_num', 'mnt', 'save_path']


dirpath_1 = ["data/FVC2002/DB1_A_2002/BMP", "data/FVC2002/DB3_A_2002/BMP"]
mntpath_1 = ["data/FVC2002/DB1_A_2002/MNT", "data/FVC2002/DB3_A_2002/MNT"]
dirpath_2 = ["data/FVC2004/DB1_A_2004/BMP", "data/FVC2004/DB3_A_2004/BMP"]
mntpath_2 = ["data/FVC2004/DB1_A_2004/MNT", "data/FVC2004/DB3_A_2004/MNT"]
num_image = 0
rows = []
rows_2 = []
for i in range(0, 2):
    dirpath1 = dirpath_1[i]
    mntpath1 = mntpath_1[i]
    dirpath2 = dirpath_2[i]
    mntpath2 = mntpath_2[i]
    for root, dirs, files in os.walk(dirpath1):
        for file in files:
            image_path = os.path.join(root, file)
            print(image_path + "\n")
            txt_file = file.replace("bmp", "txt")
            txt_path = os.path.join(mntpath1, txt_file)
            num_image += 1
            num_of_mnt = 0
            mnt = []
            with open(txt_path, "r") as f:
                mnt_datas = f.readlines()
                for mnt_data in mnt_datas:
                    num_of_mnt += 1
                    mnt_ = mnt_data.replace("\n", "")
                    # mnt = mnt_data.split(' ')
                    # x = mnt[1]
                    # y = mnt[2]
                    # mnt[3] = mnt[3].replace("\n", "")
                    # theta = mnt[3]
                    mnt.append(mnt_)
                data_row = [file, len(mnt_datas), mnt, image_path]
                rows.append(data_row)
        for root, dirs, files in os.walk(dirpath2):
            for file in files:
                image_path = os.path.join(root, file)
                print(image_path + "\n")
                txt_file = file.replace("bmp", "txt")
                txt_path = os.path.join(mntpath2, txt_file)
                num_image += 1
                num_of_mnt = 0
                mnt = []
                with open(txt_path, "r") as f:
                    mnt_datas = f.readlines()
                    for mnt_data in mnt_datas:
                        num_of_mnt += 1
                        mnt_ = mnt_data.replace("\n", "")
                        # mnt = mnt_data.split(' ')
                        # x = mnt[1]
                        # y = mnt[2]
                        # mnt[3] = mnt[3].replace("\n", "")
                        # theta = mnt[3]
                        mnt.append(mnt_)
                    data_row = [file, len(mnt_datas), mnt, image_path]
                    rows_2.append(data_row)
rows = list(rows)
rows_2 = list(rows_2)
random.shuffle(rows)
random.shuffle(rows_2)
train_data = []
test_data = []
for i in range(0, int(len(rows)/2)):
    train_data.append(rows[i])
for i in range(int(len(rows)/2), len(rows)):
    test_data.append(rows[i])
for i in range(0, int(len(rows)/2)):
    train_data.append(rows_2[i])
for i in range(int(len(rows_2)/2), len(rows_2)):
    test_data.append(rows_2[i])
with open('train_data_random.csv', 'w', newline='') as f:
    f_csv = csv.writer(f)
    f_csv.writerow(headers)
    f_csv.writerows(train_data)
with open('test_data_random.csv', 'w', newline='') as f:
    f_csv = csv.writer(f)
    f_csv.writerow(headers)
    f_csv.writerows(test_data)


