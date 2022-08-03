import torch
import torch.nn as nn
import time
from functions import *
import matplotlib.pyplot as plt
import os


def preprocessing(x, w):
        inp_unf = torch.nn.functional.unfold(x, (w.shape[2], w.shape[3]))
        A = inp_unf.transpose(1, 2)
        B = w.view(w.size(0), -1).t()
        return A,B

def postprocessing(x, exp_x, bias):
    x = x.transpose(1, 2)
    x = x.view(exp_x.shape)
    
    # bias
    bias = torch.broadcast_to(bias,(x.shape[0],x.shape[1])).reshape(x.shape[0],x.shape[1],1,1)
    x += bias
    return x

class LeNet_5(nn.Module):
    """This class defines a standard DNN model based on LeNet_5"""

    def __init__(self):
        """Initialization"""

        super(LeNet_5, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=10)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=10)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(20, 20, kernel_size=5)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(720,10)
        return

    def forward(self, x):
        """Forward propagation procedure"""
        exp_x = self.conv1(x)
        A,B = preprocessing(x, self.conv1.weight)
        result = torch.matmul(A,B)
        partial_result = result
        result_total=[]
        performances = []
        K = A.shape[2]
        total_time=0
        for i in range(partial_result.shape[0]):
            print('Current: '+ str(i))
            t1=time.time()
            result, valor, energy_value = cim(A[i].detach(), B.detach(), 1, 12, 12, 6, permutation = 'sorted', prints=False, gaussian_approximation=False, len_section=1)
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

        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = x.view(-1, 720)
        x = self.fc1(x)
        return x
    
    def forward_a(self, x):
        """Forward propagation procedure"""
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = x.view(-1, 720)
        x = self.fc1(x)
        return x