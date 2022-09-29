import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
import pretrainedmodels

import pdb

def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    return output

######################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in') # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        #init.constant_(m.bias.data, 0.0)

######################################################################
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None
 
    def forward(self, x): 
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 3
        self.spatial = BasicConv(2048, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x): 
        x_out = self.spatial(x)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale, scale
 
class SpatialAtt(nn.Module):
    def __init__(self):
        super(SpatialAtt, self).__init__()
        self.SpatialGate = SpatialGate()

    def forward(self, x): 
        x_out, scale = self.SpatialGate(x)
 
        return x_out, scale

# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, num_bottleneck=512, linear=True, return_f = False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            #add_block += [nn.LeakyReLU(0.1)]
            add_block += [nn.ReLU()]
        if droprate>0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        ### for AR pretraining as bs
        #classifier += [nn.BatchNorm1d(class_num)]
        ###
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier
    def forward(self, x):
        x = self.add_block(x)
        if self.return_f:
            f = x
            x = self.classifier(x)
            return x,f
        else:
            x = self.classifier(x)
            return x

class ClassBlock_AR(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=False, num_bottleneck=512, linear=True, return_f = False):
        super(ClassBlock_AR, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, 4*num_bottleneck)]
        else:
            num_bottleneck = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(4*num_bottleneck)]
        if relu:
            #add_block += [nn.LeakyReLU(0.1)]
            add_block += [nn.ReLU()]
        if droprate>0:
            add_block += [nn.Dropout(p=droprate)]

        add_block += [nn.Linear(4*num_bottleneck,num_bottleneck)]
        add_block += [nn.ReLU()]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        #classifier += [nn.Linear(num_bottleneck, class_num)]
        #classifier += [nn.utils.weight_norm(nn.Linear(num_bottleneck, class_num,bias=False),name='weight')]
        classifier += [nn.utils.weight_norm(nn.Linear(num_bottleneck, class_num,bias=False))]
        ### for AR pretraining as bs
        #classifier += [nn.BatchNorm1d(class_num)]
        ###
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier
    def forward(self, x):
        x = self.add_block(x)
        if self.return_f:
            f = x
            x = self.classifier(x)
            return x,f
        else:
            x = self.classifier(x)
            return x

class ClassBlock_Att(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=True, bnorm=True, num_bottleneck=512, linear=True, return_f = False):
        super(ClassBlock_Att, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
            #add_block += [nn.Tanh()]
        if droprate>0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        att_add_block = []
        att_add_block += [nn.Linear(26, 256)]
        att_add_block += [nn.BatchNorm1d(256)]
        att_add_block += [nn.LeakyReLU(0.1)]
        #att_add_block += [nn.Tanh()]
        #att_add_block += [nn.Dropout(p=droprate)]
        att_add_block += [nn.Linear(256, 512)]
        att_add_block += [nn.BatchNorm1d(512)]
        att_add_block += [nn.LeakyReLU(0.1)]
        #att_add_block += [nn.Tanh()]
        #att_add_block += [nn.Dropout(p=droprate)]
        att_add_block += [nn.Linear(512, 512)]
        att_add_block += [nn.BatchNorm1d(512)]
        att_add_block += [nn.LeakyReLU(0.1)]
        #att_add_block += [nn.Tanh()]
        #att_add_block += [nn.Dropout(p=droprate)]

        att_add_block = nn.Sequential(*att_add_block)
        att_add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck*2, num_bottleneck*2)]
        classifier += [nn.BatchNorm1d(num_bottleneck*2)]
        classifier += [nn.LeakyReLU(0.1)]
        #classifier += [nn.Tanh()]
        classifier += [nn.Dropout(p=droprate)]
        classifier += [nn.Linear(num_bottleneck*2, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.att_add_block = att_add_block
        self.classifier = classifier

    def forward(self, x, att_x):
        #pdb.set_trace()
        x = self.add_block(x)
        att_x = self.att_add_block(att_x)
        joint_x = torch.cat([x,att_x],1)
        if self.return_f:
            f = joint_x
            joint_x = self.classifier(joint_x)
            return joint_x,f
        else:
            joint_x = self.classifier(joint_x)
            return joint_x

class AttBlock(nn.Module):
    def __init__(self, input_dim):
        super(AttBlock, self).__init__()

        att_add_block = []
        att_add_block += [nn.Linear(input_dim, 512)]
        att_add_block += [nn.BatchNorm1d(512)]
        att_add_block += [nn.LeakyReLU(0.1)]
        #att_add_block += [nn.Tanh()]
        #att_add_block += [nn.Dropout(p=droprate)]
        att_add_block += [nn.Linear(512, 1024)]
        att_add_block += [nn.BatchNorm1d(1024)]
        att_add_block += [nn.LeakyReLU(0.1)]
        #att_add_block += [nn.Tanh()]
        #att_add_block += [nn.Dropout(p=droprate)]
        att_add_block += [nn.Linear(1024, 2048)]
        att_add_block += [nn.BatchNorm1d(2048)]
        att_add_block += [nn.LeakyReLU(0.1)]
        #att_add_block += [nn.Tanh()]
        #att_add_block += [nn.Dropout(p=droprate)]

        att_add_block = nn.Sequential(*att_add_block)
        att_add_block.apply(weights_init_kaiming)

        self.att_add_block = att_add_block

    def forward(self, att_x):
        #pdb.set_trace()
        att_x = self.att_add_block(att_x)

        return att_x

class AttBlock_proxy128(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AttBlock_proxy128, self).__init__()

        att_add_block = []
        att_add_block += [nn.Linear(input_dim, int(output_dim/4))]
        att_add_block += [nn.ReLU()]
        #att_add_block += [nn.LeakyReLU(0.1)]

        att_add_block += [nn.Linear(int(output_dim/4), output_dim)]
        att_add_block += [nn.ReLU()]
        #att_add_block += [nn.LeakyReLU(0.1)]

        att_add_block += [nn.Linear(output_dim, output_dim)]

        att_add_block = nn.Sequential(*att_add_block)
        att_add_block.apply(weights_init_kaiming)

        self.att_add_block = att_add_block

    def forward(self, att_x):
        att_x = self.att_add_block(att_x)

        return att_x

class AttBlock_proxy(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AttBlock_proxy, self).__init__()

        att_add_block = []
        att_add_block += [nn.Linear(input_dim, int(output_dim*4))]
        #att_add_block += [nn.BatchNorm1d(int(output_dim/2))]
        #att_add_block += [nn.LeakyReLU(0.1)]
        att_add_block += [nn.ReLU()]
        #att_add_block += [nn.Tanh()]
        #att_add_block += [nn.Dropout(p=droprate)]
        att_add_block += [nn.Linear(int(output_dim*4), output_dim)]
        #att_add_block += [nn.BatchNorm1d(output_dim)]
        #att_add_block += [nn.LeakyReLU(0.1)]
        att_add_block += [nn.ReLU()]
        #att_add_block += [nn.Tanh()]
        #att_add_block += [nn.Dropout(p=droprate)]
        att_add_block += [nn.Linear(output_dim, output_dim)]
        #att_add_block += [nn.BatchNorm1d(output_dim)]
        #att_add_block += [nn.LeakyReLU(0.1)]
        #att_add_block += [nn.Tanh()]
        #att_add_block += [nn.Dropout(p=droprate)]

        att_add_block = nn.Sequential(*att_add_block)
        att_add_block.apply(weights_init_kaiming)

        self.att_add_block = att_add_block

    def forward(self, att_x):
        #pdb.set_trace()
        att_x = self.att_add_block(att_x)

        return att_x

class AttBlock_proxy_2048(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AttBlock_proxy_2048, self).__init__()

        att_add_block = []
        att_add_block += [nn.Linear(input_dim, int(output_dim/2))]
        att_add_block += [nn.ReLU()]
        #att_add_block += [nn.BatchNorm1d(int(output_dim/2))]
        #att_add_block += [nn.Linear(input_dim, int(output_dim/16))]
        #att_add_block += [nn.LeakyReLU(0.1)]
        #att_add_block += [nn.Tanh()]
        #att_add_block += [nn.Dropout(p=droprate)]
        att_add_block += [nn.Linear(int(output_dim/2), output_dim)]
        att_add_block += [nn.ReLU()]
        #att_add_block += [nn.Linear(int(output_dim/16), int(output_dim/4))]
        #att_add_block += [nn.BatchNorm1d(output_dim)]
        #att_add_block += [nn.LeakyReLU(0.1)]
        #att_add_block += [nn.Tanh()]
        #att_add_block += [nn.Dropout(p=droprate)]
        att_add_block += [nn.Linear(output_dim, output_dim)]
        #att_add_block += [nn.Linear(int(output_dim/4), output_dim)]
        #att_add_block += [nn.BatchNorm1d(output_dim)]
        #att_add_block += [nn.LeakyReLU(0.1)]
        #att_add_block += [nn.Tanh()]
        #att_add_block += [nn.Dropout(p=droprate)]

        att_add_block = nn.Sequential(*att_add_block)
        att_add_block.apply(weights_init_kaiming)

        self.att_add_block = att_add_block

    def forward(self, att_x):
        #pdb.set_trace()
        att_x = self.att_add_block(att_x)

        return att_x

class AttBlock_proxy_norm(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AttBlock_proxy_norm, self).__init__()

        att_add_block = []
        att_add_block += [nn.Linear(input_dim, int(output_dim/2))]
        #att_add_block += [nn.BatchNorm1d(int(output_dim/2))]
        att_add_block += [nn.LeakyReLU(0.1)]
        #att_add_block += [nn.Tanh()]
        #att_add_block += [nn.Dropout(p=droprate)]
        att_add_block += [nn.Linear(int(output_dim/2), output_dim)]
        #att_add_block += [nn.BatchNorm1d(output_dim)]
        att_add_block += [nn.LeakyReLU(0.1)]
        #att_add_block += [nn.Tanh()]
        #att_add_block += [nn.Dropout(p=droprate)]
        #att_add_block += [nn.Linear(output_dim, output_dim)]
        #att_add_block += [nn.BatchNorm1d(output_dim)]
        #att_add_block += [nn.LeakyReLU(0.1)]
        #att_add_block += [nn.Tanh()]
        #att_add_block += [nn.Dropout(p=droprate)]

        att_add_block = nn.Sequential(*att_add_block)
        att_add_block.apply(weights_init_kaiming)

        self.att_add_block = att_add_block

    def forward(self, att_x):
        #pdb.set_trace()
        att_x = self.att_add_block(att_x)

        return att_x



# Define the ResNet50-based Model
class ft_net(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2):
        super(ft_net, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1,1)
            model_ft.layer4[0].conv2.stride = (1,1)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.model = model_ft
        self.classifier = ClassBlock(2048, class_num, droprate)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x

# Define the DenseNet121-based Model
class ft_net_dense(nn.Module):

    def __init__(self, class_num, droprate=0.5):
        super().__init__()
        model_ft = models.densenet121(pretrained=True)
        model_ft.features.avgpool = nn.AdaptiveAvgPool2d((1,1))
        model_ft.fc = nn.Sequential()
        self.model = model_ft
        # For DenseNet, the feature dim is 1024 
        self.classifier = ClassBlock(1024, class_num, droprate)

    def forward(self, x):
        x = self.model.features(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x

# Define the NAS-based Model
class ft_net_NAS(nn.Module):

    def __init__(self, class_num, droprate=0.5):
        super().__init__()  
        model_name = 'nasnetalarge' 
        # pip install pretrainedmodels
        model_ft = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
        model_ft.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        model_ft.dropout = nn.Sequential()
        model_ft.last_linear = nn.Sequential()
        self.model = model_ft
        # For DenseNet, the feature dim is 4032
        self.classifier = ClassBlock(4032, class_num, droprate)

    def forward(self, x):
        x = self.model.features(x)
        x = self.model.avg_pool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x
    
# Define the ResNet50-based Model (Middle-Concat)
# In the spirit of "The Devil is in the Middle: Exploiting Mid-level Representations for Cross-Domain Instance Matching." Yu, Qian, et al. arXiv:1711.08106 (2017).
class ft_net_middle(nn.Module):

    def __init__(self, class_num, droprate=0.5):
        super(ft_net_middle, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.model = model_ft
        self.classifier = ClassBlock(2048+1024, class_num, droprate)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        # x0  n*1024*1*1
        x0 = self.model.avgpool(x)
        x = self.model.layer4(x)
        # x1  n*2048*1*1
        x1 = self.model.avgpool(x)
        x = torch.cat((x0,x1),1)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x

# Part Model proposed in Yifan Sun etal. (2018)
class PCB(nn.Module):
    def __init__(self, class_num ):
        super(PCB, self).__init__()

        self.part = 6 # We cut the pool5 to 6 parts
        model_ft = models.resnet50(pretrained=True)
        self.model = model_ft
        self.avgpool = nn.AdaptiveAvgPool2d((self.part,1))
        self.dropout = nn.Dropout(p=0.5)
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1,1)
        self.model.layer4[0].conv2.stride = (1,1)
        # define 6 classifiers
        for i in range(self.part):
            name = 'classifier'+str(i)
            setattr(self, name, ClassBlock(2048, class_num, droprate=0.5, relu=False, bnorm=True, num_bottleneck=256))

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        part = {}
        predict = {}
        # get six part feature batchsize*2048*6
        for i in range(self.part):
            part[i] = torch.squeeze(x[:,:,i])
            name = 'classifier'+str(i)
            c = getattr(self,name)
            predict[i] = c(part[i])

        # sum prediction
        #y = predict[0]
        #for i in range(self.part-1):
        #    y += predict[i+1]
        y = []
        for i in range(self.part):
            y.append(predict[i])
        return y

class PCB_test(nn.Module):
    def __init__(self,model):
        super(PCB_test,self).__init__()
        self.part = 6
        self.model = model.model
        self.avgpool = nn.AdaptiveAvgPool2d((self.part,1))
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1,1)
        self.model.layer4[0].conv2.stride = (1,1)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.avgpool(x)
        y = x.view(x.size(0),x.size(1),x.size(2))
        return y

# JC-Att Baseline
class AttPS(nn.Module):
    def __init__(self, class_num ):
        super(AttPS, self).__init__()

        model_ft = models.resnet50(pretrained=True)
        self.model = model_ft
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(p=0.5)
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1,1)
        self.model.layer4[0].conv2.stride = (1,1)

        self.classifier = ClassBlock_Att(2048, class_num, droprate=0.5, bnorm=True, num_bottleneck=512)

    def forward(self, x, att_x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), x.size(1))

        y = self.classifier(x, att_x)

        return y

class AttPS_test(nn.Module):
    def __init__(self, class_num ):
        super(AttPS_test, self).__init__()

        model_ft = models.resnet50(pretrained=True)
        self.model = model_ft
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(p=0.5)
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1,1)
        self.model.layer4[0].conv2.stride = (1,1)

        self.classifier = ClassBlock_Att(2048, class_num, droprate=0.5, bnorm=True, num_bottleneck=512, return_f=True)

    def forward(self, x, att_x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), x.size(1))

        y, joint_feature = self.classifier(x, att_x)

        return y, joint_feature

# JC-Att Baseline_dupVis
class AttPS_dupVis(nn.Module):
    def __init__(self, class_num, dup_num, extract_f=False):
        super(AttPS_dupVis, self).__init__()

        model_ft = models.resnet50(pretrained=True)
        self.model = model_ft
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(p=0.5)
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1,1)
        self.model.layer4[0].conv2.stride = (1,1)

        self.dup_num = dup_num
        self.extract_f = extract_f

        self.att_block = AttBlock(26)

        self.classifier = ClassBlock(4096, class_num, droprate=0.5, bnorm=True, num_bottleneck=1024)

    def forward(self, x, att_x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.avgpool(x)
        #x = self.dropout(x)
        x = x.view(x.size(0), x.size(1))
        att_x = self.att_block(att_x)

        x = x.repeat(self.dup_num,1)

        concat_x = torch.cat((x,att_x),1)

        if self.extract_f == True:
            return x, att_x

        else:
            y = self.classifier(concat_x)

            return y

# JC-Att Baseline
class AttPS_proxy(nn.Module):
    def __init__(self, output_dim):
        super(AttPS_proxy, self).__init__()

        model_ft = models.resnet50(pretrained=True)
        self.model = model_ft
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(p=0.5)
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1,1)
        self.model.layer4[0].conv2.stride = (1,1)

        self.att_block = AttBlock_proxy(26, output_dim)

        vis_block = []
        vis_block += [nn.Linear(2048, output_dim*2)]
        ###JC updated 0909
        vis_block += [nn.Linear(output_dim*2, output_dim)]
        vis_block += [nn.Linear(output_dim, output_dim)]
        ###JC updated 0909

        vis_block = nn.Sequential(*vis_block)
        vis_block.apply(weights_init_kaiming)

        self.vis_block = vis_block

    def forward(self, x, att_x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), x.size(1))

        x = self.vis_block(x)

        att_x = self.att_block(att_x)

        return x, att_x


class AttPS_proxy_128(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AttPS_proxy_128, self).__init__()

        model_ft = models.resnet50(pretrained=True)
        self.model = model_ft
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(p=0.5)
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1,1)
        self.model.layer4[0].conv2.stride = (1,1)

        self.att_block = AttBlock_proxy(input_dim, output_dim)

        vis_block = []
        vis_block += [nn.Linear(2048, output_dim*4)]
        vis_block += [nn.ReLU()]
        vis_block += [nn.Linear(output_dim*4, output_dim)]
        vis_block += [nn.ReLU()]
        vis_block += [nn.Linear(output_dim, output_dim)]

        vis_block = nn.Sequential(*vis_block)
        vis_block.apply(weights_init_kaiming)

        self.vis_block = vis_block

    def forward(self, x, att_x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), x.size(1))

        x = self.vis_block(x)

        att_x = self.att_block(att_x)

        return x, att_x

class AttPS_proxy_2048(nn.Module):
    def __init__(self, output_dim):
        super(AttPS_proxy_2048, self).__init__()

        model_ft = models.resnet50(pretrained=True)
        self.model = model_ft
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(p=0.5)
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1,1)
        self.model.layer4[0].conv2.stride = (1,1)

        self.att_block = AttBlock_proxy_2048(26, output_dim)

        vis_block = []
        vis_block += [nn.Linear(2048, output_dim)]

        vis_block = nn.Sequential(*vis_block)
        vis_block.apply(weights_init_kaiming)

        self.vis_block = vis_block

    def forward(self, x, att_x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), x.size(1))

        x = self.vis_block(x)

        att_x = self.att_block(att_x)

        return x, att_x

class AttPS_W2V_proxy_2048(nn.Module):
    def __init__(self, output_dim):
        super(AttPS_W2V_proxy_2048, self).__init__()

        model_ft = models.resnet50(pretrained=True)
        self.model = model_ft
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(p=0.5)
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1,1)
        self.model.layer4[0].conv2.stride = (1,1)

        self.att_block = AttBlock_proxy_2048(4500, output_dim)

        vis_block = []
        vis_block += [nn.Linear(2048, output_dim)]

        vis_block = nn.Sequential(*vis_block)
        vis_block.apply(weights_init_kaiming)

        self.vis_block = vis_block

    def forward(self, x, att_x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), x.size(1))

        x = self.vis_block(x)

        att_x = self.att_block(att_x)

        return x, att_x

class AttPS_proxy_norm(nn.Module):
    def __init__(self, output_dim):
        super(AttPS_proxy_norm, self).__init__()

        model_ft = models.resnet50(pretrained=True)
        self.model = model_ft
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(p=0.5)
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1,1)
        self.model.layer4[0].conv2.stride = (1,1)

        self.att_block = AttBlock_proxy_norm(26, output_dim)

        vis_block = []
        vis_block += [nn.Linear(2048, output_dim)]

        vis_block = nn.Sequential(*vis_block)
        vis_block.apply(weights_init_kaiming)

        att_last = nn.Linear(512, output_dim)
        att_last.apply(weights_init_kaiming)

        self.vis_block = vis_block
        self.att_last = att_last

    def forward(self, x, att_x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), x.size(1))

        s = l2_norm(x)

        x = self.vis_block(x)

        att_x = self.att_block(att_x)

        att_x = l2_norm(att_x)

        att_x = self.att_last(att_x)

        return x, att_x

# JC-Vis part pooling
class AttPS_proxy_part_pool(nn.Module):
    def __init__(self, output_dim):
        super(AttPS_proxy_part_pool, self).__init__()

        model_ft = models.resnet50(pretrained=True)
        self.model = model_ft
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(p=0.5)
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1,1)
        self.model.layer4[0].conv2.stride = (1,1)

        self.att_block = AttBlock_proxy128(26, output_dim)

        vis_block = []
        #vis_block += [nn.Linear(2048*4, 2048)]
        vis_block += [nn.Linear(2048*4, 4096)]
        vis_block += [nn.ReLU()]
        #vis_block += [nn.Linear(2048, output_dim)]
        vis_block += [nn.Linear(4096, output_dim)]
        vis_block += [nn.ReLU()]
        vis_block += [nn.Linear(output_dim, output_dim)]

        vis_block = nn.Sequential(*vis_block)
        vis_block.apply(weights_init_kaiming)

        self.vis_block = vis_block

    def forward(self, x, att_x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x1 = self.avgpool(x)
        x1 = x1.view(x1.size(0), x1.size(1))
        x2 = x[:,:,:8,:]
        x2 = self.avgpool(x2)
        x2 = x2.view(x2.size(0), x2.size(1))
        x3 = x[:,:,8:16,:]
        x3 = self.avgpool(x3)
        x3 = x3.view(x3.size(0), x3.size(1))
        x4 = x[:,:,16:24,:]
        x4 = self.avgpool(x4)
        x4 = x4.view(x4.size(0), x4.size(1))
        """
        x1 = x[:,:,:6,:]
        x1 = self.avgpool(x1)
        x1 = x1.view(x1.size(0), x1.size(1))
        x2 = x[:,:,6:12,:]
        x2 = self.avgpool(x2)
        x2 = x2.view(x2.size(0), x2.size(1))
        x3 = x[:,:,12:18,:]
        x3 = self.avgpool(x3)
        x3 = x3.view(x3.size(0), x3.size(1))
        x4 = x[:,:,18:24,:]
        x4 = self.avgpool(x4)
        x4 = x4.view(x4.size(0), x4.size(1))
        """

        x = torch.cat((x1,x2,x3,x4), 1)

        x = self.vis_block(x)

        att_x = self.att_block(att_x)

        return x, att_x

class AttPS_part_proxy(nn.Module):
    def __init__(self, output_dim):
        super(AttPS_part_proxy, self).__init__()

        model_ft = models.resnet50(pretrained=True)
        self.model = model_ft
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(p=0.5)
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1,1)
        self.model.layer4[0].conv2.stride = (1,1)

        self.att_block_w = AttBlock_proxy(26, output_dim)
        self.att_block_t = AttBlock_proxy(11, int(output_dim/2))
        self.att_block_b = AttBlock_proxy(8, int(output_dim/2))

        vis_block_w = [nn.Linear(2048, output_dim)]
        vis_block_w = nn.Sequential(*vis_block_w)
        vis_block_w.apply(weights_init_kaiming)

        vis_block_t = [nn.Linear(2048, int(output_dim/2))]
        vis_block_t = nn.Sequential(*vis_block_t)
        vis_block_t.apply(weights_init_kaiming)

        vis_block_b = [nn.Linear(2048, int(output_dim/2))]
        vis_block_b = nn.Sequential(*vis_block_b)
        vis_block_b.apply(weights_init_kaiming)

        self.vis_block_w = vis_block_w
        self.vis_block_t = vis_block_t
        self.vis_block_b = vis_block_b

    def forward(self, x, att_x, att_x_t, att_x_b):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x_w = self.avgpool(x)
        x_w = x_w.view(x_w.size(0), x_w.size(1))
        x_t = self.avgpool(x[:,:,:16,:])
        x_t = x_t.view(x_t.size(0), x_t.size(1))
        x_b = self.avgpool(x[:,:,8:,:])
        x_b = x_b.view(x_b.size(0), x_b.size(1))

        x_w = self.vis_block_w(x_w)
        x_t = self.vis_block_t(x_t)
        x_b = self.vis_block_b(x_b)

        att_x_w = self.att_block_w(att_x)
        att_x_t = self.att_block_t(att_x_t)
        att_x_b = self.att_block_b(att_x_b)

        return x_w, x_t, x_b, att_x_w, att_x_t, att_x_b

class AttPS_proxy_with_softmax(nn.Module):
    def __init__(self, output_dim):
        super(AttPS_proxy_with_softmax, self).__init__()

        model_ft = models.resnet50(pretrained=True)
        self.model = model_ft
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(p=0.5)
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1,1)
        self.model.layer4[0].conv2.stride = (1,1)

        self.att_block = AttBlock_proxy(26, output_dim)

        vis_block = []
        vis_block += [nn.Linear(2048, output_dim*2)]
        vis_block += [nn.Linear(output_dim*2, output_dim)]
        vis_block += [nn.Linear(output_dim, output_dim)]
 
        vis_block = nn.Sequential(*vis_block)
        vis_block.apply(weights_init_kaiming)

        self.vis_block = vis_block

        #self.classifier = ClassBlock(1024, 2020, droprate=0.5, bnorm=True, num_bottleneck=4096)
        self.vis_classifier = ClassBlock(512, 2020, droprate=0.0, bnorm=False, num_bottleneck=1024)
        self.att_classifier = ClassBlock(512, 2020, droprate=0.0, bnorm=False, num_bottleneck=1024)

    def forward(self, x, att_x, labels):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), x.size(1))

        x = self.vis_block(x)

        att_x = self.att_block(att_x)

        #joint_x = torch.cat((x, att_x[labels]),1)
        att_batch = att_x[labels]

        outputs_v = self.vis_classifier(x)
        outputs_a = self.att_classifier(att_batch)

        return x, att_x, outputs_v, outputs_a

class AttPS_proxy_ARpretrained(nn.Module):
    def __init__(self, model, input_dim, output_dim):
        super(AttPS_proxy_ARpretrained, self).__init__()

        #model_ft = models.resnet50(pretrained=True)
        self.model = model.model
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(p=0.5)
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1,1)
        self.model.layer4[0].conv2.stride = (1,1)

        self.att_block = AttBlock_proxy(input_dim, output_dim)

        vis_block = []
        vis_block += [nn.Linear(2048, output_dim*4)]
        vis_block += [nn.ReLU()]
        vis_block += [nn.Linear(output_dim*4, output_dim)]
        vis_block += [nn.ReLU()]
        vis_block += [nn.Linear(output_dim, output_dim)]

        vis_block = nn.Sequential(*vis_block)
        vis_block.apply(weights_init_kaiming)

        self.vis_block = vis_block

    def forward(self, x, att_x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), x.size(1))

        x = self.vis_block(x)

        att_x = self.att_block(att_x)

        return x, att_x

class AttPS_proxy_ARpretrained_with_SpatialAtt(nn.Module):
    def __init__(self, model, output_dim):
        super(AttPS_proxy_ARpretrained_with_SpatialAtt, self).__init__()

        #model_ft = models.resnet50(pretrained=True)
        self.model = model.model
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(p=0.5)
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1,1)
        self.model.layer4[0].conv2.stride = (1,1)

        self.att_block = AttBlock_proxy(26, output_dim)

        self.SpatialAtt = model.SpatialAtt

        vis_block = []
        vis_block += [nn.Linear(2048, output_dim)]

        vis_block = nn.Sequential(*vis_block)
        vis_block.apply(weights_init_kaiming)

        self.vis_block = vis_block

    def forward(self, x, att_x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x, attention = self.SpatialAtt(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), x.size(1))

        x = self.vis_block(x)

        att_x = self.att_block(att_x)

        return x, att_x

class AR(nn.Module):                                                                                                      
    def __init__(self, num_cat):
        super(AR, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        self.model = model_ft
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(p=0.5)
        self.num_cat = len(num_cat)
        #self.fclayer = nn.Linear(2048,1024).apply(weights_init_classifier)
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1,1)
        self.model.layer4[0].conv2.stride = (1,1)
        for i in range(self.num_cat):
            setattr(self, "ar_local_"+str(i), ClassBlock_AR(2048, num_cat[i], 0.0, relu=True, num_bottleneck=128))
 
    def forward(self, x): 
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.avgpool(x)
        x=torch.squeeze(x[:,:,:])

        y = []
        for i in range(self.num_cat):
            ar_local = getattr(self, "ar_local_"+str(i))
            predict = ar_local(x)
            y.append(predict)
            
        return y

class AR_SpatialAtt(nn.Module):                                                                                          
    def __init__(self, num_cat):
        super(AR_SpatialAtt, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        self.model = model_ft
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(p=0.5)
        self.num_cat = len(num_cat)
        #self.fclayer = nn.Linear(2048,1024).apply(weights_init_classifier)
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1,1)
        self.model.layer4[0].conv2.stride = (1,1)
        self.SpatialAtt = SpatialAtt()
        for i in range(self.num_cat):
            setattr(self, "ar_local_"+str(i), ClassBlock(2048, num_cat[i], 0.5, relu=True, num_bottleneck=1024))
 
    def forward(self, x): 
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x, attention = self.SpatialAtt(x)
        x = self.avgpool(x)
        x=torch.squeeze(x[:,:,:])

        y = []
        for i in range(self.num_cat):
            ar_local = getattr(self, "ar_local_"+str(i))
            predict = ar_local(x)
            y.append(predict)
            
        return y

'''
# debug model structure
# Run this code with:
python model.py
'''
if __name__ == '__main__':
# Here I left a simple forward function.
# Test the model, before you train it. 
    net = ft_net(751, stride=1)
    net.classifier = nn.Sequential()
    print(net)
    input = Variable(torch.FloatTensor(8, 3, 256, 128))
    output = net(input)
    print('net output size:')
    print(output.shape)
