import os
import os.path as osp
import sys
import numpy as np
import cv2 as cv
import tqdm
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torchvision import transforms
from torch.utils.data import DataLoader
from lib.utils.misc import ensure_path, Timer, Averager, calculate_recog
from lib.datasets.datasets import MyDatasets
from lib.datasets.transforms import ShiftPro, FlipLevel, FlipVertical, RotatePro, Resize, ToTensor, RandomApply, ResizeVali, ToTensorVali
from lib.network.my_model import get_MyModel


class Tester(object):
    """The class that contains the code for the test phase."""
    def __init__(self, args):
        # Set the folder to save the records and checkpoints
        log_base_dir = './logs/'
        if not osp.exists(log_base_dir):
            os.mkdir(log_base_dir)
        train_base_dir = osp.join(log_base_dir, 'test')
        if not osp.exists(train_base_dir):
            os.mkdir(train_base_dir)
        save_path1 = '_'.join([args.model_type, args.loss_func_mode, args.activation_func, args.heatmap_type])
        save_path2 = '_batchsize' + str(args.batch_size) + '_lr' + str(args.base_lr) + \
                     '_explr' + str(args.exp_lr) + '_epoch' + str(args.epoch)
        args.save_path = train_base_dir + '/' + save_path1 + '_' + save_path2
        ensure_path(args.save_path)

        # Set args to be shareable in the class
        self.args = args

        # Load pretrain set
        transform_test = transforms.Compose([Resize((args.origin_size[0], args.origin_size[1])), ToTensorVali()])
        self.test_dataset = MyDatasets(root_dir=args.project_dir, names_file=args.test_dataset,
                                       transform=transform_test,
                                       train_aug=False)
        self.test_loader = DataLoader(dataset=self.test_dataset, batch_size=args.test_batch_size, shuffle=False,
                                      pin_memory=args.pin_memory, num_workers=args.loader_workers, drop_last=True)
        # Build pretrain model
        self.model = get_MyModel(mode=args.model_type, activation_func=args.activation_func)
        state_dict = torch.load(args.eval_weights)
        self.model.load_state_dict(state_dict)
        # Set optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.base_lr, momentum=args.momentum,
                                         weight_decay=args.weight_decay, nesterov=args.nesterov)

        # Set loss function
        if args.loss_func_mode == "MSE":
            self.loss_func = torch.nn.MSELoss(reduction='sum').cuda()
        elif args.loss_func_mode == "L1":
            self.loss_func = torch.nn.L1Loss(reduction='sum').cuda()

        # Set learning rate scheduler
        """学习一下learning rate的step！！！！！！！"""
        # self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.args.pre_step_size, \
        #                                                     gamma=self.args.pre_gamma)

        # Set model to GPU
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            self.model = self.model.cuda()

    def tes(self):
        """The function for the test phase."""
        # Set the timer
        timer = Timer()
        # Set global count to zero
        global_count = 0
        self.model.eval()

        # Set averager classes to record validation losses and accuracies
        val_loss_averager = Averager()
        val_acc_averager = Averager()
        val_recall_averager = Averager()
        val_F1_averager = Averager()
        for i, batch in enumerate(self.test_loader, 1):
            images, label, label_list = batch['image'], batch['label'], batch['label_list']
            if torch.cuda.is_available():
                data = images.cuda()
            else:
                data = images
            # label = label.float()
            if torch.cuda.is_available():
                label = label.type(torch.cuda.FloatTensor)
            else:
                label = label.type(torch.FloatTensor)
            predict_data = self.model(data)
            loss = self.loss_func(predict_data, label)
            acc, recall, F1 = calculate_recog(self.args, data, label, label_list, predict_data)
            val_loss_averager.add(loss.item())
            val_acc_averager.add(acc)
            val_recall_averager.add(recall)
            val_F1_averager.add(F1)
            print('Batch {}, Val, Loss={:.4f} Acc={:.4f} Recall={:.4f} F1={:.4f}'
                  .format(i, loss.item(), acc, recall, F1))
        val_loss_averager = val_loss_averager.item()
        val_acc_averager = val_acc_averager.item()
        val_recall_averager = val_recall_averager.item()
        val_F1_averager = val_F1_averager.item()
        # Print loss and accuracy for this epoch
        print('Epoch {}, Val, Loss={:.4f} Acc={:.4f} Recall={:.4f} F1={:.4f}'
              .format(0, val_loss_averager, val_acc_averager, val_recall_averager, val_F1_averager))
