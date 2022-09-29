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
from PIL import Image
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
parser.add_argument('--which_epoch',default='last', type=str, help='0,1,2,3...or last')
parser.add_argument('--h', action='store_true', help='use high resolution dataset')
parser.add_argument('--vis', action='store_true', help='visualization')
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


######################################################################
# Make diff mat between attribute sets on PA100K dataset(JC)
# Todo (JC) use mat multiplication istead dup of for-loop.
#---------------------------
def make_att_diff_mat(att_list):

    att_len_dict = {1:3, 4:3, 13:2, 15:4, 19:2, 22:3}                 
    start_idx = [0,1,4,7,8,9,10,11,12,13,15,19,21,22,25]

    att_diff_mat = []
    for att_f in tqdm(att_list):
        att_f = torch.IntTensor(list(map(int,list(att_f))))
        att_diff_list=[]
        for att_s in att_list:
            att_s = torch.IntTensor(list(map(int,list(att_s))))
            diff = att_f != att_s
            diff_val = 0
            for idx in start_idx:
                if idx in att_len_dict:
                    att_len = att_len_dict[idx]
                    if sum(diff[idx:idx+att_len]) > 0:
                        diff_val += 1
                else:
                    diff_val += diff[idx]
                if diff_val < 0 or diff_val > 15:
                    print("diff_att_mat error!")
                    pdb.set_trace()
            att_diff_list.append(diff_val)
        att_diff_mat.append(att_diff_list)
    
    att_diff_mat = np.stack(att_diff_mat)
    att_diff_mat = torch.FloatTensor(att_diff_mat)

    return att_diff_mat

def convert_ar_label_Anyang(att_keys):
    converted_labels = []
    for att in att_keys:
        tmp1 = att.split('_')
        ar_label = list(map(int,tmp1[:-1]+list(tmp1[-1])))
        converted_labels.append(ar_label)

    return converted_labels

######################################################################              
# Load model                                                                                            
#---------------------------                                                        
def load_network(network):                                                          
    save_path = os.path.join('./model',name,'net_%s.pth'%opt.which_epoch)           
    #network.load_state_dict(torch.load(save_path))                                 
    network.load_state_dict(torch.load(save_path),strict=False)                     
    return network                                                                  
                          

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
transform_test_list = [
    #transforms.Resize(size=(384,192),interpolation=3), #Image.BICUBIC
    transforms.Resize(size=(256,128),interpolation=3), #Image.BICUBIC
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]


if opt.color_jitter:
    transform_train_list = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0)] + transform_train_list

IMAGENET_MEAN, IMAGENET_STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])

data_transforms = {
    'train': transforms.Compose( transform_train_list ),
    'test': transforms.Compose(transform_test_list),
}


data_high = ''
if opt.h:
     data_high = '_high'

image_datasets = {}
image_datasets['train'] = datasets.ImageFolder(os.path.join(data_dir, 'train' + data_high),
                                          data_transforms['train'])
image_datasets['test'] = datasets.ImageFolder(os.path.join(data_dir, 'test' + data_high),
                                         data_transforms['test'])

#balanced_sampler = sampler.BalancedSampler(image_datasets['train'], batch_size=opt.batchsize, images_per_class=)

test_att_keys = list(image_datasets['test'].class_to_idx.keys())
all_ar_labels = convert_ar_label_Anyang(test_att_keys)
all_ar_labels = torch.LongTensor(all_ar_labels)

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                             shuffle=False, num_workers=8, pin_memory=True) # 8 workers may work faster
              for x in ['train','test']}
              #for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
class_names = image_datasets['test'].classes

use_gpu = torch.cuda.is_available()

since = time.time()
print(time.time()-since)

y_loss = {} # loss history
y_loss['train'] = []
y_loss['test'] = []
y_err = {}
y_err['train'] = []
y_err['test'] = []

color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

"""
gender_list = ['female', 'male']                    
age_list = ["age under 10", "age 10-20", "age 20-30", "age 30-40", "age 40-50", "age over 50"]                       
tall_list = ["under 110cm", "120-130cm", "130-140cm", "140-150cm", "150-160cm", "160-170cm", "170-180cm", "over 180cm"]         
hair_type_list = ['hair type long', 'hair type normal', 'hair type permed', 'hair type shaved', 'hair type short', 'hair type sporting', 'hair type straight', 'hair type tied_hair']
hair_color_list = ['hair color black', 'hair color brown', 'hair color yellow']
top_type_list = ['long_sleeve', 'short_sleeve'] 
top_color_list = ['top color beige', 'top color black', 'top color blue', 'top color blue_green', 'top color brown', 'top color burgundy', 'top color dark_gray', 'top color gray',
        'top color green', 'top color khaki', 'top color light_brown', 'top color navy', 'top color navy_blue', 'top color navy_green', 'top color orange', 'top color pink',
        'top color purple', 'top color red', 'top color red_brown', 'top color sky_blue', 'top color white', 'top color yellow', 'top color yellow_green']
bottom_type_list = ['dress', 'long_pants', 'long_skirt', 'short_pants', 'short_skirt']
bottom_color_list = ['bottom color beige', 'bottom color black', 'bottom color blue', 'bottom color brown', 'bottom color burgundy', 'bottom color dark_gray', 'bottom color gray', 'bottom color navy',
        'bottom color navy_blue', 'bottom color navy_green', 'bottom color pink', 'bottom color purple', 'bottom color sky_blue', 'bottom color white']       
item_list = ['backpack', 'bag', 'bicycle', 'carrier', 'cellphone', 'electric_wheel', 'etc',   
        'glasses', 'hat', 'mask', 'motorcycle', 'plastic_cup', 'portable_fan', 'stroller',
        'sunglasses', 'umbrella', 'water_bottle']

att_list = [gender_list, age_list, tall_list, hair_type_list, hair_color_list, top_type_list, top_color_list, bottom_type_list, bottom_color_list]
"""
gender_list = ['female', 'male']                    
hair_type_list = ['hair type long', 'hair type normal', 'hair type others']
hair_color_list = ['hair color black', 'hair color brown', 'hair color yellow']
top_type_list = ['long sleeve', 'short sleeve'] 
top_color_list = ['top color brown', 'top color black', 'top color blue', 'top color red', 'top color gray', 'top color green', 'top color yellow', 'top color pink', 'top color purple', 'top color white'] 
bottom_type_list = ['long lower body clothing', 'short lower body clothing'] 
bottom_color_list = ['bottom color brown', 'bottom color black', 'bottom color blue', 'bottom color red', 'bottom color gray', 'bottom color pink', 'bottom color purple', 'bottom color white']
att_list = [gender_list, hair_type_list, hair_color_list, top_type_list, top_color_list, bottom_type_list, bottom_color_list]

def test_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    warm_up = 0.1 # We start from the 0.1*lrRate
    warm_iteration = round(dataset_sizes['train']/opt.batchsize)*opt.warm_epoch # first 5 epoch

    for epoch in range(num_epochs):
        print('-' * 10)
        
        for phase in ['test']:
            model.train(False)  # Set model to evaluate mode
            att_keys = test_att_keys

            running_loss = 0.0
            running_corrects = [0 for i in range(7)]

            img_count = 0

            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data
                ar_labels = all_ar_labels[labels]
                ar_labels = ar_labels.t()
                now_batch_size,c,h,w = inputs.shape
                if use_gpu:
                    inputs = Variable(inputs.cuda().detach())
                    ar_labels = Variable(ar_labels.cuda().detach())
                else:
                    inputs,  ar_labels = Variable(inputs), Variable(ar_labels)
                optimizer.zero_grad()

                # forward
                if phase == 'test':
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
                    score = sm(outputs[i])
                    preds.append(torch.max(score.data,1))

                # visualization
                vis_dir ='./vis_image'
                if not os.path.isdir(vis_dir):
                    os.mkdir(vis_dir)
                if opt.vis:
                    for i in range(now_batch_size):
                        img_name = "test_" + str(img_count)+".jpg"
                        img = inputs[i]
                        img = img.detach().cpu().numpy()
                        img = np.transpose(img,(1,2,0))
                        img = np.clip(255.0 * (img * IMAGENET_STD + IMAGENET_MEAN), 0, 255)
                        img = Image.fromarray(np.uint8(img))

                        plt.subplot(1, 2, 1)
                        plt.imshow(img)
                        plt.axis('off')

                        plt.subplot(1, 2, 2)
                        color_idx = 0
                        for y in range(7):
                            if y<9:
                                att_score = np.round(np.array(preds[y][0][i].cpu()),2)
                                att_name = att_list[y][int(preds[y][1][i])]
                                att_info = att_name + ": "+str(att_score)
                                plt.text(0,0.9 - color_idx*0.06, att_info, color=color_list[color_idx], fontsize=15)
                                color_idx +=1
                            else:
                                if preds[y][1][i] == 0:
                                    continue
                                else:
                                    att_score = np.round(np.array(preds[y][0][i].cpu()),2)
                                    att_name = item_list[y-9]
                                    att_info = att_name + ": "+str(att_score)
                                    plt.text(0,0.9 - color_idx*0.1, att_info, color=color_list[color_idx], fontsize=15)
                                    color_idx +=1

                            plt.axis('off')

                        plt.subplots_adjust(wspace=-0.1)
                        plt.savefig(vis_dir+'/'+img_name)
                        plt.close()
                        img_count += 1

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
        time_elapsed = time.time() - since
        print('Evaluation complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print()

    time_elapsed = time.time() - since
    return model


#num_cat = [2, 6, 8, 8, 3, 2, 23, 5, 14, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
num_cat = [2, 3, 3, 2, 10, 2, 8]
if opt.AR:
    if opt.SpatialAtt:
        model = AR_SpatialAtt(num_cat) 
    else:
        model = AR(num_cat)

opt.nclasses = len(class_names)
#opt.nclasses = 2

model = load_network(model)

if opt.AR:
    ignored_params = []
    params = filter(lambda p: id(p) not in ignored_params, model.parameters())

    optimizer_ft = optim.SGD([
             {'params': params, 'lr': opt.lr},
             #{'params': model.att_block.parameters(), 'lr': 100*opt.lr},
         ], weight_decay=5e-4, momentum=0.9, nesterov=True)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)
######################################################################
# Train and evaluate
# ^^^^^^^^^^^^^^^^^^
#
# It should take around 1-2 hours on GPU. 
#
dir_name = os.path.join('./model',name)
if not os.path.isdir(dir_name):
    os.mkdir(dir_name)
#record every run
copyfile('./model.py', dir_name+'/model.py')

# save opts
with open('%s/opts.yaml'%dir_name,'w') as fp:
    yaml.dump(vars(opt), fp, default_flow_style=False)

model = model.eval()

# model to gpu
model = model.cuda()
criterion = nn.CrossEntropyLoss().cuda()

with torch.no_grad():
    model = test_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=1)

