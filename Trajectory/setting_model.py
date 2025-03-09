'''
creat various model

Version: pytorch 1.13.1
Author: LN
'''

import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable                 
import matplotlib.pyplot as plt
import torch.nn.functional as F
import time       
import os    

# get device_cpu or gpu
device = torch.device("cuda" if torch.cuda.is_available() else cpu)
print("CUDA Device Count: {}".format(torch.cuda.device_count()))
print("CUDA Device Name: {}".format(torch.cuda.get_device_name(0)))

# Parameter initialization
p1, p2, p3 = 250, 255, 259

#
# creat model_fc
class FC_inverse_Net(nn.Module):
    def __init__(self):
        super(FC_inverse_Net, self).__init__()
        self.layer1 = nn.Linear(600, 800)
        self.layer2 = nn.Linear(800, 1600)
        self.layer3 = nn.Linear(1600, 3200)
        self.layer4 = nn.Linear(3200, 800)
        self.layer5 = nn.Linear(800, 200)
        self.layer6 = nn.Linear(200, 64)             
        self.layer7 = nn.Linear(64, 4)
        self.activation_1 = nn.LeakyReLU()
        self.activation_2 = nn.ReLU()

    def forward(self, inputs):
        x1 = self.activation_1(self.layer1(inputs))      # (100,300)--(100,800)
        x2 = self.activation_1(self.layer2(x1))
        x3 = self.activation_1(self.layer3(x2))
        x4 = self.activation_1(self.layer4(x3))
        x5 = self.activation_1(self.layer5(x4))
        x6 = self.activation_1(self.layer6(x5))
        x7 = self.layer7(x6)
        return x7
    
# creat model_fc
class FC_Net(nn.Module):
    def __init__(self):
        super(FC_Net, self).__init__()
        self.layer1 = nn.Linear(4, 3200)                             # features=9       
        self.layer2 = nn.Linear(3200, 1600)
        self.layer3 = nn.Linear(1600, 800)
        self.layer4 = nn.Linear(800, 600)
        self.layer5 = nn.Linear(600, 300)                   
        self.activation_1 = nn.LeakyReLU()
        self.activation_2 = nn.ReLU()

    def forward(self, inputs):
        x1 = self.activation_1(self.layer1(inputs))
        x2 = self.activation_1(self.layer2(x1))
        x3 = self.activation_1(self.layer3(x2))
        x4 = self.activation_1(self.layer4(x3))
        x5 = self.layer5(x4)
        return x5
    
    # creat model_fc
class FC_Net_2D(nn.Module):
    def __init__(self):
        super(FC_Net_2D, self).__init__()
        self.layer1 = nn.Linear(4, 1600)                             # features=9       
        self.layer2 = nn.Linear(1600, 3200)
        self.layer3 = nn.Linear(3200, 1600)
        self.layer4 = nn.Linear(1600, 800)
        self.layer5 = nn.Linear(800, 600)                   
        self.activation_1 = nn.LeakyReLU()
        self.activation_2 = nn.ReLU()

    def forward(self, inputs):
        x1 = self.activation_1(self.layer1(inputs))
        x2 = self.activation_1(self.layer2(x1))
        x3 = self.activation_1(self.layer3(x2))
        x4 = self.activation_1(self.layer4(x3))
        x5 = self.layer5(x4)
        return x5
    
# add to Dropout or nn.BatchNorm1d--------------model.train() and model.eval() is important
# creat CNN_S
class DB1_Net(nn.Module):
    def __init__(self):
        super(DB1_Net, self).__init__()
        self.layer0 = nn.Sequential(
            nn.Linear(4, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 60)
        )
        

    def forward(self, inputs):
        x = self.layer0(inputs)                      # DB1直接将4个结构参数映射到60
        return x


class DB2_Net(nn.Module):
    def __init__(self):
        super(DB2_Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(60, 1600),
            nn.ReLU(),
            #nn.Dropout(0.5),
            nn.Linear(1600, 3200),
            nn.ReLU(),
            #nn.Dropout(0.5),
            nn.Linear(3200, 1600),
            nn.ReLU(),
            #nn.Dropout(0.5),
            nn.Linear(1600, 800),
            nn.ReLU(),
            #nn.Dropout(0.5),
            nn.Linear(800, 600),                          
        )

    def forward(self, x):
        x = self.layer1(x)                            # DB2直接将60个抽样点映射到600个散点
        return x   
    
# Add to CNN
class CNN_Net(nn.Module):
    def __init__(self):
        super(CNN_Net, self).__init__()
        
        self.layer2 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=1, out_channels=16, kernel_size=3, padding=1),      # (10,16,11)
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU()
            )
        
        self.layer3 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=1, kernel_size=3, padding=1)
            )
        
        self.layer1 = nn.Sequential(
            nn.Linear(4, 3200),
            nn.ReLU(),
            nn.Linear(3200, 1600),
            nn.ReLU(),
            nn.Linear(1600, 800),
            nn.ReLU(),
            nn.Linear(800, 400),
            nn.ReLU(),
            nn.Linear(400, 60)
            )

    def forward(self, x):
        # 先使用全连接层映射出高维数据
        x = self.layer1(x)
        x = x.view(-1,1,60)
        # 再使用转置卷积和卷积获得细节
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(-1, 60)
        return x

    
class ResBlock(nn.Module):
    # 定义残差块
    def __init__(self, in_channels, out_channels):                    # BN层删除, 如果包含BN层会batch_size=1
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            #nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            #nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.skip = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0),
        )
        self.BN_Relu = nn.Sequential(
            #nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)

        )

    def forward(self, x):
        x = self.conv(x) + self.skip(x)
        x = self.BN_Relu(x)
        return x
    
    
class ResNet(nn.Module):
    def __init__(self, bilinear=False):
        super(ResNet, self).__init__()
        self.M_layer2 = ResBlock(1,64)                                           # (N, 128, 300)
        #self.pooling1 = nn.ConvTranspose1d(64, 64, kernel_size=3, padding = 1)       # (N, 128, 100)
        self.M_layer3 = ResBlock(64,1)                                          # (N, 256, 100)
        #self.pooling2 = nn.ConvTranspose1d(1, 1, kernel_size=3, padding = 1)        # (N, 256, 1)
        self.fc_1 = torch.nn.Sequential(
        nn.Linear(600, 800),
        nn.ReLU(),
        nn.Linear(800, 1600),
        nn.ReLU(),
        nn.Linear(1600, 3200),
        nn.ReLU(),
        nn.Linear(3200, 800),
        nn.ReLU(),
        nn.Linear(800, 200),
        nn.ReLU(),
        nn.Linear(200, 64),
        nn.ReLU(),
        nn.Linear(64, 4),
        )
    def forward(self, x):
        x = x.view(-1, 1, 600)
        x = self.M_layer2(x)
        x = self.M_layer3(x)
        x = x.view(-1, 600)
        out = self.fc_1(x)
        return out