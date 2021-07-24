# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: QiuXinmin
# This source code is for predict minutiae of figerprint, include location and scale
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
""" Generate commands for train phase. """
import os
gpu = 0
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)


def run_exp(base_lr=0.0001, exp_lr=0.00001, epoch=2000, batch_size=10, with_vali=False, is_parallel=False):
    the_command = 'python main.py' \
                  + ' --epoch=' + str(epoch) \
                  + ' --gpu=' + str(gpu) \
                  + ' --base_lr=' + str(base_lr) \
                  + ' --exp_lr=' + str(exp_lr) \
                  + ' --batch_size=' + str(batch_size) \
                  + ' --with_vali=' + str(with_vali) \
                  + ' --is_parallel=' + str(is_parallel) \
                  + ' --init_weights=' + str(None) \
                  + ' --eval_weights=' + str(None) \
                  + ' --phase=train'

    os.system(the_command)


run_exp(base_lr=0.0001, exp_lr=0.00001, epoch=2000, batch_size=10)
