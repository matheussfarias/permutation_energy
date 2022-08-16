'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
import numpy as np
import time
from functions import *

def preprocessing(x, w, padding):
    inp_unf = torch.nn.functional.unfold(x, (w.shape[2], w.shape[3]), padding=padding)
    A = inp_unf.transpose(1, 2)
    B = w.view(w.size(0), -1).t()
    return A,B

def postprocessing(x, exp_x, bias):
    x = x.transpose(1,2)
    x = x.view(exp_x.shape)

    # bias
    bias = torch.broadcast_to(bias,(x.shape[0],x.shape[1])).reshape(x.shape[0],x.shape[1],1,1)
    x += bias
    return x


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        exp_x = self.conv1(x)

        A,B = preprocessing(x, self.conv1.weight, self.conv1.padding)
        result = torch.matmul(A,B)
        partial_result = result
        result_total=[]
        performances = []
        K = A.shape[2]
        total_time=0
        for i in range(partial_result.shape[0]):
            print('Current: '+ str(i))
            t1=time.time()
            result, valor, energy_value = cim(A[i].detach(), B.detach(), 1, 12, 12, 6, permutation = 'sorted', prints=False)
            f=open('./results/energy.txt','a')
            np.savetxt(f, [int(energy_value)], fmt='%1.3f', newline=", ")
            f.write("\n")
            f.close()
            t2=time.time()
            print(t2-t1)
            performances.append(valor)
            result_total.append(result)
            total_time += t2-t1
        print(total_time)
        total_time=0

        result_total = torch.stack(result_total).reshape(partial_result.shape)

        print(performances)
        print('Batch Done')

        x = postprocessing(result_total, exp_x, self.conv1.bias)

        out = self.bn1(x)
        out = self.relu(out)
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def forward_a(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg[1:]:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def test():
    net = VGG('VGG11')
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

# test()
