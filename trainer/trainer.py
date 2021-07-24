""" Trainer for train phase. """
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


class Trainer(object):
    """The class that contains the code for the train phase."""
    def __init__(self, args):
        # Set the folder to save the records and checkpoints
        log_base_dir = './logs/'
        if not osp.exists(log_base_dir):
            os.mkdir(log_base_dir)
        train_base_dir = osp.join(log_base_dir, 'train')
        if not osp.exists(train_base_dir):
            os.mkdir(train_base_dir)
        save_path1 = '_'.join([args.model_type, args.loss_func_mode, args.activation_func,
                               args.dimension_choice, args.heatmap_type])
        save_path2 = '_batchsize' + str(args.batch_size) + '_lr' + str(args.base_lr) + \
                     '_explr' + str(args.exp_lr) + '_epoch' + str(args.epoch)
        args.save_path = train_base_dir + '/' + save_path1 + '_' + save_path2
        ensure_path(args.save_path)

        # Set args to be shareable in the class
        self.args = args

        # Load train set
        # print("transform_start")
        transform_train = transforms.Compose([RandomApply(ShiftPro(), 0.5), RandomApply(FlipLevel(), 0.5),
                                              RandomApply(FlipVertical(), 0.5), RandomApply(RotatePro(), 0.5),
                                              Resize((args.origin_size[0], args.origin_size[1]),
                                                     args.theta_stride, args.dimension_choice), ToTensor()])
        transform_test = transforms.Compose([Resize((args.origin_size[0], args.origin_size[1]),
                                                    args.theta_stride, args.dimension_choice), ToTensorVali()])
        self.train_dataset = MyDatasets(root_dir=args.project_dir, names_file=args.train_dataset, transform=transform_train,
                                        train_aug=True)
        # 数据加载器，将一个dataset和一个sampler组合到一起，并且提供一个在dataset上的可迭代对象。
        self.train_loader = DataLoader(dataset=self.train_dataset, batch_size=args.batch_size, shuffle=True,
                                       pin_memory=args.pin_memory, num_workers=args.loader_workers, drop_last=True)
        # Build train model

        if self.args.init_weights == 'None':
            self.model = get_MyModel(mode=args.model_type, activation_func=args.activation_func,
                                     dimension_choice=args.dimension_choice, theta_stride=args.theta_stride)
            # print("init weights is None")
        else:
            # print("init weights is not None")
            self.model = get_MyModel(mode=args.model_type, activation_func=args.activation_func,
                                     dimension_choice=args.dimension_choice, theta_stride=args.theta_stride)
            state_dict = torch.load(args.init_weights)
            self.model.load_state_dict(state_dict)
        # Build validation model
        self.test_dataset = MyDatasets(root_dir=args.project_dir, names_file=args.test_dataset, transform=transform_test,
                                       train_aug=False)
        self.test_loader = DataLoader(dataset=self.test_dataset, batch_size=args.test_batch_size, shuffle=False,
                                      pin_memory=args.pin_memory, num_workers=args.loader_workers, drop_last=True)

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

    def save_model(self, name):
        """The function to save checkpoints.
        Args:
          name: the name for saved checkpoint
        """
        torch.save(self.model.state_dict(), osp.join(self.args.save_path, name + '.pth'))

    def train(self):
        """The function for the train phase."""

        # Set the pretrain log
        trlog = dict()  # 原来是{}
        trlog['args'] = vars(self.args)
        trlog['train_loss'] = []
        trlog['val_loss'] = []
        trlog['train_acc'] = []
        trlog['val_acc'] = []
        trlog['train_recall'] = []
        trlog['val_recall'] = []
        trlog['train_F1'] = []
        trlog['val_F1'] = []
        trlog['max_F1'] = 0.0
        trlog['max_F1_loss'] = 0.0
        trlog['max_F1_epoch'] = 0

        # Set the timer
        timer = Timer()
        # Set global count to zero
        global_count = 0
        # Set tensorboardX
        writer = SummaryWriter(comment=self.args.save_path)

        # Start pretrain
        for epoch in range(1, self.args.epoch + 1):
            # Update learning rate
            """learning rate 的 step版本"""
            # self.lr_scheduler.step()
            # Set the model to train mode
            self.model.train()

            # Set averager classes to record training losses and accuracies
            train_loss_averager = Averager()
            train_acc_averager = Averager()
            train_F1_averager = Averager()

            # Using tqdm to read samples from train loader
            tqdm_gen = tqdm.tqdm(self.train_loader)
            print(len(self.train_loader))
            print("tqdm_start")
            for i, batch in enumerate(tqdm_gen, 1):
                # Update global count number
                print("i:", i)
                global_count = global_count + 1
                images, label = batch['image'], batch['label']
                print("image_shape: ", images.shape)
                print("label_shape: ", label.shape)
                if torch.cuda.is_available():
                    data = images.cuda()
                else:
                    data = images
                # label = label.float()
                if torch.cuda.is_available():
                    label = label.type(torch.cuda.FloatTensor)
                else:
                    label = label.type(torch.FloatTensor)
                # Output for model
                predict_data = self.model(data)
                # Calculate train loss
                loss = self.loss_func(predict_data, label)/self.args.batch_size
                # Calculate train accuracy
                # print("predict_data_shape: ", predict_data.shape)
                # print("label_shape: ", label.shape)
                # acc = count_acc(predict_data, label)
                acc = 0
                # Write the tensorboardX records
                writer.add_scalar('data/loss', float(loss), global_count)
                writer.add_scalar('data/acc', float(acc), global_count)
                # Print loss and accuracy for this step
                tqdm_gen.set_description('Epoch {}, Loss={:.4f} Acc={:.4f} out_max:{:.5f}'
                                         .format(epoch, loss.item(), acc, predict_data.max()))

                # Add loss and accuracy for the averagers
                train_loss_averager.add(loss.item())
                train_acc_averager.add(acc)

                # Loss backwards and optimizer updates
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Update the averagers
            train_loss_averager = train_loss_averager.item()
            train_acc_averager = train_acc_averager.item()

            """Vali过程还没写"""
            # Start validation for this epoch, set model to eval mode
            self.model.eval()

            # Set averager classes to record validation losses and accuracies
            val_loss_averager = Averager()
            val_acc_averager = Averager()
            val_recall_averager = Averager()
            val_F1_averager = Averager()

            # Print previous information
            if epoch % 10 == 0:
                print('Best Epoch {}, Best Val F1={:.4f}'.format(trlog['max_F1_epoch'], trlog['max_F1']))
            # Run validation
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

            # Update validation averagers
            val_loss_averager = val_loss_averager.item()
            val_acc_averager = val_acc_averager.item()
            val_recall_averager = val_recall_averager.item()
            val_F1_averager = val_F1_averager.item()
            # Write the tensorboardX records
            writer.add_scalar('data/val_loss', float(val_loss_averager), epoch)
            writer.add_scalar('data/val_acc', float(val_acc_averager), epoch)
            writer.add_scalar('data/val_recall', float(val_recall_averager), epoch)
            writer.add_scalar('data/val_F1', float(val_F1_averager), epoch)
            # Print loss and accuracy for this epoch
            print('Epoch {}, Val, Loss={:.4f} Acc={:.4f} Recall={:.4f} F1={:.4f}'
                  .format(epoch, val_loss_averager, val_acc_averager, val_recall_averager, val_F1_averager))

            # Update best saved model
            if val_F1_averager > trlog['max_F1']:
                trlog['max_F1'] = val_F1_averager
                trlog['max_F1_epoch'] = epoch
                self.save_model('max_F1')
            # Save model every 10 epochs
            if epoch % 100 == 0:
                self.save_model('epoch' + str(epoch))

            # Update the logs
            trlog['train_loss'].append(train_loss_averager)
            # trlog['train_acc'].append(train_acc_averager)
            trlog['val_loss'].append(val_loss_averager)
            trlog['val_acc'].append(val_acc_averager)
            trlog['val_recall'].append(val_recall_averager)
            trlog['val_F1'].append(val_F1_averager)

            # Save log
            torch.save(trlog, osp.join(self.args.save_path, 'trlog'))

            if epoch % 100 == 0:
                print('Running Time: {}, Estimated Time: {}'.format(timer.measure(),
                                                                    timer.measure(epoch / self.args.max_epoch)))
        writer.close()



