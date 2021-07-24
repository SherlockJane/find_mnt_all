""" Main function for this repo. """
import argparse
import torch
from lib.utils.gpu_tools import set_gpu
from trainer.trainer import Trainer
from trainer.tester import Tester
import os
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Basic parameters
    parser.add_argument('--project_dir', type=str, default=os.path.abspath(os.path.join(os.getcwd(), ".")))
    parser.add_argument('--model_type', type=str, default='Model8', choices=['Model6', 'Model6_2', 'Model8', 'Model8_DW'])  # The network architecture
    parser.add_argument('--loss_func_mode', default='MSE', choices=['MSE', 'L1'])
    parser.add_argument('--activation_func', default='ReLU', choices=['ReLU', 'swish'])
    parser.add_argument('--dimension_choice', type=str, default='3D', choices=['2D', '3D'])
    parser.add_argument('--heatmap_type', type=str, default='cylinder', choices=['cycle', 'ellipsoid', 'cylinder'])
    parser.add_argument('--theta_stride', type=int, default=10)  # 细节点角度分割的stride
    parser.add_argument('--dataset_dir', type=str, default='data/')  # Dataset folder
    parser.add_argument('--train_dataset', type=str, default='train_data_random.csv')  # Dataset
    parser.add_argument('--test_dataset', type=str, default='test_data_random.csv')  # Dataset
    parser.add_argument('--phase', type=str, default='train', choices=['train', 'test'])  # Phase
    parser.add_argument('--seed', type=int, default=0)  # Manual seed for PyTorch, "0" means using random seed
    parser.add_argument('--disable_cuda', default=False)
    parser.add_argument('--gpu', default='0')  # GPU id
    parser.add_argument('--loader_workers', default=0, type=int, help='number of workers for data loading')

    # Parameters for train phase
    parser.add_argument('--epoch', type=int, default=2000)  # Epoch number for meta-train phase
    parser.add_argument('--batch_size', type=int, default=10)  # The number for different tasks used for meta-train
    parser.add_argument('--base_lr', type=float, default=0.01)  # Learning rate
    parser.add_argument('--exp_lr', type=float, default=0.01)  # Learning rate
    parser.add_argument('--origin_size', default=(384, 384), type=int, help='size of the origin image which input to network')
    parser.add_argument('--init_weights', type=str, default=None)  # The pre-trained weights for meta-train phase
    parser.add_argument('--eval_weights', type=str, default=None)  # The meta-trained weights for meta-eval phase
    parser.add_argument('--nesterov', dest='nesterov', default=True, type=bool)  # 梯度下降
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')  # 冲量
    parser.add_argument('--weight-decay', '--wd', default=0.000, type=float, metavar='W', help='weight decay (default: 1e-4)')  # 权重衰减
    parser.add_argument('--with_vali', default=False)
    parser.add_argument('--is_parallel', default=False)

    # Parameters for test phase
    # parser.add_argument('--test_epoch', type=int, default=100) # Epoch number for pre-train phase
    parser.add_argument('--test_batch_size', type=int, default=1)  # Batch size for pre-train phase
    parser.add_argument('--NMS_thread', default=0.1)
    parser.add_argument('--arrow_image_save_path', type=str, default="/logs/arrow_image/")

    # Set and print the parameters
    args = parser.parse_args()

    # add args.device
    args.device = torch.device('cpu')
    args.pin_memory = False
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        args.pin_memory = True
    print(vars(args))

    # Set the GPU id
    set_gpu(args.gpu)

    # Set manual seed for PyTorch
    # 使用同样的随机初始化种子，保证每次初始化都相同
    if args.seed == 0:
        print('Using random seed.')
        torch.backends.cudnn.benchmark = True
    else:
        print('Using manual seed:', args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Start trainer for pre-train, meta-train or meta-eval
    if args.phase == 'train':
        trainer = Trainer(args)
        trainer.train()
    elif args.phase == 'test':
        tester = Tester(args)
        tester.tes()
    else:
        raise ValueError('Please set correct phase.')
