# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
#from PIL import Image
import time
import os
from model import ft_net, ft_net_dense, ft_net_NAS, PCB, AttPS_proxy, AttPS_proxy_128, AttPS_proxy_2048, AttPS_proxy_part_pool, AR, AR_SpatialAtt
import yaml
import math
from shutil import copyfile

#from dataset import sampler
import losses

import pdb
import random
import numpy as np
from tqdm import tqdm

version =  torch.__version__
#fp16
try:
    from apex.fp16_utils import *
    from apex import amp, optimizers
except ImportError: # will be 3.x series
    print('This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda support (https://github.com/NVIDIA/apex) and update pytorch to 1.0')
######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--name',default='ft_ResNet50', type=str, help='output model name')
parser.add_argument('--data_dir',default='../Market/pytorch',type=str, help='training dir path')
parser.add_argument('--train_all', action='store_true', help='use all training data' )
parser.add_argument('--color_jitter', action='store_true', help='use color jitter in training' )
parser.add_argument('--batchsize', default=32, type=int, help='batchsize')
parser.add_argument('--stride', default=2, type=int, help='stride')
parser.add_argument('--erasing_p', default=0, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--use_dense', action='store_true', help='use densenet121' )
parser.add_argument('--use_NAS', action='store_true', help='use NAS' )
parser.add_argument('--warm_epoch', default=0, type=int, help='the first K epoch that needs warm up')
parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
parser.add_argument('--droprate', default=0.5, type=float, help='drop rate')
parser.add_argument('--AR', action='store_true', help='use attribute recognition')
parser.add_argument('--SpatialAtt', action='store_true', help='use attribute recognition with spatial attention')
parser.add_argument('--AttPS', action='store_true', help='use AttPS base + ResNet50' ) 
parser.add_argument('--AttPS_2048', action='store_true', help='use AttPS base + ResNet50 + 2048 emb_dim' )
parser.add_argument('--AttPS_128', action='store_true', help='use AttPS base + ResNet50 + 128 emb_dim' )
parser.add_argument('--PCB', action='store_true', help='use PCB+ResNet50' )
parser.add_argument('--fp16', action='store_true', help='use float16 instead of float32, which will save about 50% memory' )
parser.add_argument('--NegMrgH', action='store_true', help='use Hard Neg Margin loss')
parser.add_argument('--NegMrgP', action='store_true', help='use Proportinal Neg Margin loss')
parser.add_argument('--ProxyNCA', action='store_true', help='use ProxyNCA loss')
parser.add_argument('--ProxyNCArc', action='store_true', help='use ArcFace loss')
parser.add_argument('--AttDiffMat', action='store_true', help='use stored Att Diff Mat')
parser.add_argument('--PartPool', action='store_true', help='use part pooling at vis')
parser.add_argument('--h', action='store_true', help='use high resolution dataset')
opt = parser.parse_args()

fp16 = opt.fp16
data_dir = opt.data_dir
name = opt.name
str_ids = opt.gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    gid = int(str_id)
    if gid >=0:
        gpu_ids.append(gid)

# set gpu ids
if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])
    cudnn.benchmark = True

def bin2dec(b, bits):
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(b.device, b.dtype)
    return torch.sum(mask * b, -1)

def convert_ar_label_Anyang(att_keys):
    converted_labels = []
    for att in att_keys:
        tmp1 = att.split('_')
        ar_label = list(map(int,tmp1[:-1]+list(tmp1[-1])))
        converted_labels.append(ar_label)

    return converted_labels


######################################################################
# Load Data
# ---------
#
transform_train_list = [
    #transforms.Resize((384,192), interpolation=3),
    transforms.Resize((256,128), interpolation=3),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
transform_val_list = [
    #transforms.Resize(size=(384,192),interpolation=3), #Image.BICUBIC
    transforms.Resize(size=(256,128),interpolation=3), #Image.BICUBIC
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]


if opt.color_jitter:
    transform_train_list = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0)] + transform_train_list

print(transform_train_list)
data_transforms = {
    'train': transforms.Compose( transform_train_list ),
    'val': transforms.Compose(transform_val_list),
}


train_high = ''
if opt.h:
     train_high = '_high'

image_datasets = {}
image_datasets['train'] = datasets.ImageFolder(os.path.join(data_dir, 'train' + train_high),
                                          data_transforms['train'])
#image_datasets['val'] = datasets.ImageFolder(os.path.join(data_dir, 'val'),
#                                         data_transforms['val'])

#balanced_sampler = sampler.BalancedSampler(image_datasets['train'], batch_size=opt.batchsize, images_per_class=)

train_att_keys = list(image_datasets['train'].class_to_idx.keys())
all_ar_labels = convert_ar_label_Anyang(train_att_keys)
all_ar_labels = torch.LongTensor(all_ar_labels)
#val_att_keys = list(image_datasets['val'].class_to_idx.keys())

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                             shuffle=True, num_workers=8, pin_memory=True) # 8 workers may work faster
              for x in ['train']}
              #for x in ['train', 'val']}
#dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train']}
class_names = image_datasets['train'].classes

use_gpu = torch.cuda.is_available()

since = time.time()
inputs_tmp, classes_tmp = next(iter(dataloaders['train']))
print(time.time()-since)

num_inst= [[51435, 65711], [38921, 70420, 7805], [92496, 24578, 72], [67356, 49790], [13654, 20155, 20261, 9367, 8516, 9754, 4490, 7760, 7393, 15796], [97393, 19753], [9497, 69764, 25216, 16, 9697, 252, 469, 2235]]

######################################################################
# Training the model
# ------------------
#
# Now, let's write a general function to train a model. Here, we will
# illustrate:
#
# -  Scheduling the learning rate
# -  Saving the best model
#
# In the following, parameter ``scheduler`` is an LR scheduler object from
# ``torch.optim.lr_scheduler``.

y_loss = {} # loss history
y_loss['train'] = []
y_loss['val'] = []
y_err = {}
y_err['train'] = []
y_err['val'] = []

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    #best_model_wts = model.state_dict()
    #best_acc = 0.0
    warm_up = 0.1 # We start from the 0.1*lrRate
    warm_iteration = round(dataset_sizes['train']/opt.batchsize)*opt.warm_epoch # first 5 epoch

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        #for phase in ['train', 'val']:
        for phase in ['train']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
                # JC: att_keys
                att_keys = train_att_keys
            else:
                model.train(False)  # Set model to evaluate mode
                #att_keys = val_att_keys

            running_loss = 0.0
            running_corrects = [0 for i in range(7)]

            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data

                ar_labels = all_ar_labels[labels]
                ar_labels = ar_labels.t()
                ###
                now_batch_size,c,h,w = inputs.shape
                if now_batch_size<opt.batchsize: # skip the last batch
                    continue
                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda().detach())
                    ar_labels = Variable(ar_labels.cuda().detach())
                else:
                    inputs,  ar_labels = Variable(inputs), Variable(ar_labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                if phase == 'val':
                    with torch.no_grad():
                        outputs = model(inputs)
                else:
                    outputs = model(inputs)

                sm = nn.Softmax(dim=1)
                num_att = len(ar_labels)
                loss = 0
                preds = []

                for i in range(num_att):
                    loss += criterion(outputs[i], ar_labels[i])
                    loss += AMD_reg[i](getattr(model, "ar_local_"+str(i)).classifier[0]._parameters['weight_v'], 1.0/torch.FloatTensor(num_inst[i]).cuda())
                    score = sm(outputs[i])
                    preds.append(torch.max(score.data,1))

                # backward + optimize only if in training phase
                if epoch<opt.warm_epoch and phase == 'train': 
                    warm_up = min(1.0, warm_up + 0.9 / warm_iteration)
                    loss *= warm_up

                if phase == 'train':
                    if fp16: # we use optimier to backward loss
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()
                    optimizer.step()
                # statistics
                if int(version[0])>0 or int(version[2]) > 3: # for the new version like 0.4.0, 0.5.0 and 1.0.0
                    running_loss += loss.item() * now_batch_size
                else :  # for the old version like 0.3.0 and 0.3.1
                    running_loss += loss.data[0] * now_batch_size

                for i in range(7):
                    running_corrects[i] += float(torch.sum(preds[i][1] == ar_labels[i].data))

            avg_epoch_acc = 0
            for i in range(7):
                epoch_acc = running_corrects[i] / (dataset_sizes[phase])
                avg_epoch_acc += epoch_acc
                print('{} Acc: {:.4f}'.format(phase, epoch_acc))

            avg_epoch_acc = avg_epoch_acc / 7.0
            epoch_loss = running_loss / (dataset_sizes[phase])
            print('{} Loss: {:.4f} Total Acc: {:.4f}'.format(
                phase, epoch_loss, avg_epoch_acc))

            y_loss[phase].append(epoch_loss)
            y_err[phase].append(1.0-avg_epoch_acc)            
            # deep copy the model
            """
            if phase == 'val':
                last_model_wts = model.state_dict()
                #if epoch%10 == 1 or epoch%10 == 2 or epoch%10 == 3:
                #if epoch%10 == 9:
            """
            if epoch > 0:
                save_network(model, epoch)
                #draw_curve(epoch)

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    #print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    #model.load_state_dict(last_model_wts)
    save_network(model, 'last')
    return model


######################################################################
# Draw Curve
#---------------------------
x_epoch = []
fig = plt.figure()
ax0 = fig.add_subplot(121, title="loss")
ax1 = fig.add_subplot(122, title="top1err")
def draw_curve(current_epoch):
    x_epoch.append(current_epoch)
    ax0.plot(x_epoch, y_loss['train'], 'bo-', label='train')
    ax0.plot(x_epoch, y_loss['val'], 'ro-', label='val')
    ax1.plot(x_epoch, y_err['train'], 'bo-', label='train')
    ax1.plot(x_epoch, y_err['val'], 'ro-', label='val')
    if current_epoch == 0:
        ax0.legend()
        ax1.legend()
    fig.savefig( os.path.join('./model',name,'train.jpg'))

######################################################################
# Save model
#---------------------------
def save_network(network, epoch_label):
    save_filename = 'net_%s.pth'% epoch_label
    save_path = os.path.join('./model',name,save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if torch.cuda.is_available():
        network.cuda(gpu_ids[0])


######################################################################
# Finetuning the convnet
# ----------------------
#
# Load a pretrainied model and reset final fully connected layer.
#

#num_cat = [2, 6, 8, 8, 3, 2, 23, 5, 14, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
num_cat = [2, 3, 3, 2, 10, 2, 8]
if opt.AR:
    if opt.SpatialAtt:
        model = AR_SpatialAtt(num_cat) 
    else:
        model = AR(num_cat)

opt.nclasses = len(class_names)
#opt.nclasses = 2

print(model)

if opt.AR:
    ignored_params = []
    params = filter(lambda p: id(p) not in ignored_params, model.parameters())

    optimizer_ft = optim.SGD([
             {'params': params, 'lr': opt.lr},
             #{'params': model.att_block.parameters(), 'lr': 100*opt.lr},
         ], weight_decay=5e-4, momentum=0.9, nesterov=True)

# Decay LR by a factor of 0.1 every 5 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)
######################################################################
# Train and evaluate
# ^^^^^^^^^^^^^^^^^^
#
# It should take around 1-2 hours on GPU. 
#
dir_name = os.path.join('./model',name)
if not os.path.isdir(dir_name):
    os.mkdirs(dir_name)
#record every run
copyfile('./train_att_backbone_from_AR_Anyang.py', dir_name+'/train_att_backbone_from_AR_Anyang.py')
copyfile('./model.py', dir_name+'/model.py')

# save opts
with open('%s/opts.yaml'%dir_name,'w') as fp:
    yaml.dump(vars(opt), fp, default_flow_style=False)

# model to gpu
model = model.cuda()
criterion = nn.CrossEntropyLoss().cuda()
#criterion = losses.FocalLoss().cuda()
AMD_reg = [losses.AMD_Regularizer(num_cat[i]) for i in range(len(num_cat))]

model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=10)

