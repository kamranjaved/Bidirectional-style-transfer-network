import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib
import os
import argparse

import matching
#import gan_old as gan  #original is import gan
import gan_share as gan  #original is import gan
from utils import *

parser = argparse.ArgumentParser()
# training
parser.add_argument('--gpu_num', type=int, default=0)
parser.add_argument('--log_dir', type=str, default='record')
parser.add_argument('--load_dir', type=str, default='record')
parser.add_argument('--continue_tr', type=bool, default=False)
parser.add_argument('--load_epoch', type=int, default=2300)
parser.add_argument('--max_epoch', type=int, default=5000)
parser.add_argument('--similarity_loss', type=str, default='L1')
parser.add_argument('--similarity_lambda', type=float, default=10)
parser.add_argument('--similarity_loss_med', type=str, default='L1')
parser.add_argument('--med_lambda', type=float, default=1)
parser.add_argument('--med_step', type=int, default=0)
# inputs
parser.add_argument('--img_size', type=int, default=272)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--tr_dir', type=str, default="../data/synthesis/DB272prip")
parser.add_argument('--ts_dir', type=str, default="../data/synthesis/DB272prip")
parser.add_argument('--tr_list', type=str, default='tr_list.txt')
parser.add_argument('--ts_list', type=str, default='ts_list.txt')
parser.add_argument('--num_identity', type=int, default=48)
# network
parser.add_argument('--g_model', type=str, default='col_gen')
parser.add_argument('--d_model', type=str, default='PatchGan')
#201005
parser.add_argument('--use_enc_dec', type=bool, default=True)
parser.add_argument('--g_enc_model', type=str, default='col_gen_enc') # col_gen_enc, col_gen_short
parser.add_argument('--g_dec_model', type=str, default='col_gen_dec') # col_gen_dec, col_gen_short
parser.add_argument('--med_channels', type=int, default=256)
#matching
parser.add_argument('--enc_model', type=str, default='alex')
parser.add_argument('--enc_norm', type=str, default='batch')
parser.add_argument('--g_matching_lambda', type=float, default=1)
#learning rate
parser.add_argument('--enc_lr', type=float, default=0.0002)
parser.add_argument('--g_lr', type=float, default=0.0002)
parser.add_argument('--d_lr', type=float, default=0.0002)
#record
parser.add_argument('--print_epoch', type=int, default=10)
parser.add_argument('--save_epoch', type=int, default=100)
#test
parser.add_argument('--ts_batch_size', type=int, default=5)
parser.add_argument('--ts_min_epoch', type=int, default=100)
parser.add_argument('--ts_max_epoch', type=int, default=5000)
parser.add_argument('--ts_unit_epoch', type=int, default=100)
parser.add_argument('--use_gallery', type=bool, default=False)
parser.add_argument('--gall_list', type=str, default='list_gallery_1500.txt')
#train mode
#parser.add_argument('--train_mode', type=str, default='train_with_gan') #original
#--------------
parser.add_argument('--train_mode', type=str, default='train_gan')
#d_meds
parser.add_argument('--d_p2s', type=str, default='False')
parser.add_argument('--d_s2p', type=str, default='False')
parser.add_argument('--med_d_lambda', type=float, default=0.1)

config = parser.parse_args()
config.ts_dir = config.tr_dir
if not os.path.exists(config.log_dir):
    os.mkdir(config.log_dir)

# mode
from_zero=True
if config.train_mode == 'train_gan':
    if config.continue_tr:
        config.load_dir = config.log_dir + '/step0/net_ckpt'
        from_zero = False
    config.log_dir = config.log_dir + '/step0'
elif config.train_mode == 'train_with_fixed_gan':
    if config.continue_tr:
        config.load_dir = config.log_dir + '/step1/net_ckpt'
        from_zero = False
    else:
        config.continue_tr = True
        config.load_dir = config.log_dir + '/step0/net_ckpt'
    config.log_dir = config.log_dir + '/step1'
elif config.train_mode == 'train_with_gan':
    config.continue_tr = True
    config.load_dir = config.log_dir + '/step1/net_ckpt'
    if config.tr_list == 'tr_list_F.txt':
        config.log_dir = config.log_dir + '/step2_F'
    elif config.tr_list == 'tr_list_I.txt':
        config.log_dir = config.log_dir + '/step2_I'
    elif config.tr_list == 'tr_list_IF.txt':
        config.log_dir = config.log_dir + '/step2_IF'
    elif config.tr_list == 'tr_list_UoM-A.txt':
        config.log_dir = config.log_dir + '/step2_A'
    elif config.tr_list == 'tr_list_UoM-B.txt':
        config.log_dir = config.log_dir + '/step2_B'
    else:
        config.log_dir = config.log_dir + '/step2'
else:
    print('Wrong train mode')
    assert False


# Build
#net = matching.matchnet(config, from_zero)
#------------
net = gan.GAN(config)

if net.gpu_num is not None:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(net.gpu_num)
print(device_lib.list_local_devices())

#gan_ = net.build_trainer(mode=config.train_mode, gan_config=config)
net.build_trainer(mode=config.train_mode)
# Variables
if not os.path.exists(net.log_dir):
    os.mkdir(net.log_dir)
txtfile = open(net.log_dir+'/variables.txt', 'w')
print("Enc_vars")


for i in range(len(net.Gen_vars)):
    print(net.Gen_vars[i].name)
    print(net.Gen_vars[i].name, file=txtfile)
print("Dis_vars")
for i in range(len(net.Dis_vars)):
    print(net.Dis_vars[i].name)
    print(net.Dis_vars[i].name, file=txtfile)
print("build model finished")

print("Tr_num: "+str(net.tr_photo_num))
print("Ts_num: "+str(net.ts_photo_num))
txtfile.close()

# Train
#net.train_with_gan(gan_, mode=config.train_mode)
net.train_gan()
