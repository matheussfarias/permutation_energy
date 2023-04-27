'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
import numpy as np
import time
from functions import *


def preprocessing(x, w, padding):
    inp_unf = torch.nn.functional.unfold(x, (w.shape[2], w.shape[3]), padding = padding)
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

def fc_to_cim(x, layer, v_ref = 1, d = 12, wq = 8, adc = 12, permutation = 'random', grid = False, perc=[68, 27], num_sec=100, b_set = None, opt=0, add_noise=0, noise_gain=0, prints=False, apply_noise_red=False):
    opt_grind = 0.1
    print(perc)
    dim = layer(x)
    partial_result = dim
    result_total=[]
    performances = []
    #K = A.shape[2]
    total_time=0
    shape = [[int(opt_grind*partial_result.shape[0])]]
    shape.append(list(partial_result.shape[1:]))
    shape = [item for sublist in shape for item in sublist]
    if grid:
        for i in range(int(opt_grind*partial_result.shape[0])):
            print('Current: '+ str(i))
            t1=time.time()
            result, valor, energy_value,active,s,noise, n_adcs = cim(x[i][np.newaxis, :].detach(), layer.weight.detach().T, v_ref, d, wq, adc, permutation = permutation, prints=prints,perc=perc, num_sec=num_sec, b_set = b_set, opt=opt, add_noise=add_noise, noise_gain=noise_gain, apply_noise_red=apply_noise_red)
            print(active)
            f=open('./grid/energy.txt','a')
            np.savetxt(f, [int(energy_value)], fmt='%1.3f', newline=", ")
            f.write("\n")
            f.close()
            f=open('./grid/active.txt','a')
            np.savetxt(f, [active], fmt='%1.3f', newline=", ")
            f.write("\n")
            f.close()
            f=open('./results/noise.txt','a')
            np.savetxt(f, [noise], fmt='%1.3f', newline=", ")
            f.write("\n")
            f.close()
            f=open('./results/s.txt','a')
            np.savetxt(f, [s.cpu()], fmt='%1.3f', newline=", ")
            f.write("\n")
            f.close()
            f=open('./results/n_adcs.txt','a')
            np.savetxt(f, [n_adcs.cpu()], fmt='%1.3f', newline=", ")
            f.write("\n")
            f.close()
            t2=time.time()
            print(t2-t1)
            performances.append(valor)
            result_total.append(result)
            total_time += t2-t1
        result_total = torch.stack(result_total).reshape(shape)
        result_total = torch.cat((result_total, partial_result[int(opt_grind*partial_result.shape[0]):,:]))
        result_total = result_total.reshape(partial_result.shape)
    else:
        for i in range(partial_result.shape[0]):
            print('Current: '+ str(i))
            t1=time.time()
            result, valor, energy_value,active,s,noise, n_adcs = cim(x[i][np.newaxis, :].detach(), layer.weight.detach().T, v_ref, d, wq, adc, permutation = permutation, prints=prints,perc=perc, num_sec=num_sec, b_set = b_set, opt=opt, add_noise=add_noise, noise_gain=noise_gain, apply_noise_red=apply_noise_red)
            f=open('./results/energy.txt','a')
            np.savetxt(f, [int(energy_value)], fmt='%1.3f', newline=", ")
            f.write("\n")
            f.close()
            f=open('./results/active.txt','a')
            np.savetxt(f, [active], fmt='%1.3f', newline=", ")
            f.write("\n")
            f.close()
            f=open('./results/noise.txt','a')
            np.savetxt(f, [noise.cpu()], fmt='%1.3f', newline=", ")
            f.write("\n")
            f.close()
            f=open('./results/s.txt','a')
            np.savetxt(f, s.cpu(), fmt='%1.3f', newline=", ")
            f.write("\n")
            f.close()
            f=open('./results/n_adcs.txt','a')
            np.savetxt(f, [n_adcs.cpu()], fmt='%1.3f', newline=", ")
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

    x = result_total + layer.bias
    return x


def conv_to_cim(x, layer, v_ref = 1, d = 12, wq = 8, adc = 12, permutation = 'sorted', grid = False, perc=[68, 27], num_sec=1, b_set = [12,12,12,12,12,12,12,12], opt=0, add_noise=0, noise_gain=0, prints=False, apply_noise_red=False):
    opt_grind = 0.1
    dim = layer(x)
    A,B = preprocessing(x, layer.weight, layer.padding)
    ta = time.perf_counter()
    result = torch.matmul(A,B)
    partial_result = result
    result_total=[]
    performances = []
    K = A.shape[2]
    total_time=0
    shape = [[int(opt_grind*partial_result.shape[0])]]
    shape.append(list(partial_result.shape[1:]))
    shape = [item for sublist in shape for item in sublist]
    if grid:
        for i in range(int(opt_grind*partial_result.shape[0])):
            print('Current: '+ str(i))
            t1=time.time()
            result, valor, energy_value, active, s, noise, n_adcs = cim(A[i].detach(), B.detach(), v_ref, d, wq, adc, permutation = permutation, prints=prints, perc=perc, num_sec=num_sec, b_set=b_set, opt=opt, add_noise=add_noise, noise_gain=noise_gain, apply_noise_red=apply_noise_red)
            f=open('./grid/energy.txt','a')
            np.savetxt(f, [int(energy_value)], fmt='%1.3f', newline=", ")
            f.write("\n")
            f.close()
            f=open('./grid/active.txt','a')
            np.savetxt(f, [active], fmt='%1.3f', newline=", ")
            f.write("\n")
            f.close()
            f=open('./grid/noise.txt','a')
            np.savetxt(f, [noise], fmt='%1.3f', newline=", ")
            f.write("\n")
            f.close()
            f=open('./grid/s.txt','a')
            np.savetxt(f, [s], fmt='%1.3f', newline=", ")
            f.write("\n")
            f.close()
            f=open('./grid/n_adcs.txt','a')
            np.savetxt(f, [n_adcs], fmt='%1.3f', newline=", ")
            f.write("\n")
            f.close()
            t2=time.time()
            print(t2-t1)
            performances.append(valor)
            result_total.append(result)
            total_time += t2-t1
        print(total_time)
        total_time=0
        result_total = torch.stack(result_total).reshape(shape)
        result_total = torch.cat((result_total, partial_result[int(opt_grind*partial_result.shape[0]):,:]))
        result_total = result_total.reshape(partial_result.shape)
    else:
        for i in range(partial_result.shape[0]):
            print('Current: '+ str(i))
            t1 = time.perf_counter()
            result, valor, energy_value, active, s, noise, n_adcs = cim(A[i].detach(), B.detach(), v_ref, d, wq, adc, permutation = permutation, prints=prints,perc=perc, num_sec=num_sec, b_set=b_set, opt=opt, add_noise=add_noise, noise_gain=noise_gain, apply_noise_red=apply_noise_red)
            f=open('./results/energy.txt','a')
            np.savetxt(f, [int(energy_value)], fmt='%1.3f', newline=", ")
            f.write("\n")
            f.close()
            f=open('./results/active.txt','a')
            np.savetxt(f, [active], fmt='%1.3f', newline=", ")
            f.write("\n")
            f.close()
            f=open('./results/s.txt','a')
            np.savetxt(f, s.cpu(), fmt='%1.3f', newline=", ")
            f.write("\n")
            f.close()
            f=open('./results/noise.txt','a')
            np.savetxt(f, [noise.cpu()], fmt='%1.3f', newline=", ")
            f.write("\n")
            f.close()
            f=open('./results/n_adcs.txt','a')
            np.savetxt(f, [n_adcs.cpu()], fmt='%1.3f', newline=", ")
            f.write("\n")
            f.close()
            t2=time.perf_counter()
            print(t2-t1)
            performances.append(valor)
            result_total.append(result)
            total_time += t2-t1
        print(total_time)
        total_time=0
        result_total = torch.stack(result_total).reshape(partial_result.shape)
    print(performances)
    print('Batch Done')

    x = postprocessing(result_total, dim, layer.bias)
    return x


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.count = 0
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)

        self.mpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avpool2 = nn.AvgPool2d(kernel_size=1, stride=1)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)
        self.avpool3 = nn.AvgPool2d(kernel_size=1, stride=1)

        self.mpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avpool4 = nn.AvgPool2d(kernel_size=1, stride=1)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU(inplace=True)
        self.avpool5 = nn.AvgPool2d(kernel_size=1, stride=1)

        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU(inplace=True)
        self.avpool6 = nn.AvgPool2d(kernel_size=1, stride=1)

        self.mpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avpool7 = nn.AvgPool2d(kernel_size=1, stride=1)

        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.relu5 = nn.ReLU(inplace=True)
        self.avpool8 = nn.AvgPool2d(kernel_size=1, stride=1)

        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(512)
        self.relu6 = nn.ReLU(inplace=True)
        self.avpool9 = nn.AvgPool2d(kernel_size=1, stride=1)

        self.mpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avpool10 = nn.AvgPool2d(kernel_size=1, stride=1)

        self.conv7 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(512)
        self.relu7 = nn.ReLU(inplace=True)
        self.avpool11 = nn.AvgPool2d(kernel_size=1, stride=1)

        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(512)
        self.relu8 = nn.ReLU(inplace=True)
        self.avpool12 = nn.AvgPool2d(kernel_size=1, stride=1)

        self.mpool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avpool13 = nn.AvgPool2d(kernel_size=1, stride=1)

        self.fc1 = nn.Linear(512, 10)

    def forward(self, x):
        #x = self.conv1(x)
        x = conv_to_cim(x, self.conv1, opt=1, permutation = 'sorted', num_sec=9, b_set = torch.FloatTensor([8, 8, 8, 8, 8, 7, 7, 6]).to(device), add_noise = 1, noise_gain = 0.01, apply_noise_red=True)
        #27
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.mpool1(x)
        x = self.avpool2(x)

        #x = self.conv2(x)
        x = conv_to_cim(x, self.conv2, opt=1, permutation = 'sorted', num_sec=9, b_set = torch.FloatTensor([8, 8, 8, 8, 8, 7, 7, 6]).to(device),  add_noise = 1, noise_gain = 0.01, apply_noise_red=True)
        #576
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.avpool3(x)

        x = self.mpool2(x)
        x = self.avpool4(x)

        #x = self.conv3(x)
        x = conv_to_cim(x, self.conv3, opt=1, permutation = 'sorted', num_sec=9, b_set = torch.FloatTensor([8, 8, 8, 8, 8, 7, 7, 6]).to(device),  add_noise = 1, noise_gain = 0.01, apply_noise_red=True)
        #1152
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.avpool5(x)

        x = self.mpool3(x)
        x = self.avpool6(x)

        #x = self.conv4(x)
        x = conv_to_cim(x, self.conv4, opt=1, permutation = 'sorted', num_sec=9, b_set = torch.FloatTensor([8, 8, 8, 8, 8, 7, 7, 6]).to(device),  add_noise = 1, noise_gain = 0.01, apply_noise_red=True)        
        #2304
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.avpool7(x)

        #x = self.conv5(x)
        x = conv_to_cim(x, self.conv5, opt=1, permutation = 'sorted', num_sec=9, b_set = torch.FloatTensor([8, 8, 8, 8, 8, 7, 7, 6]).to(device),  add_noise = 1, noise_gain = 0.01, apply_noise_red=True)
        #2304
        x = self.bn5(x)
        x = self.relu5(x)
        x = self.avpool8(x)

        x = self.mpool4(x)
        x = self.avpool9(x)

        #x = self.conv6(x)
        x = conv_to_cim(x, self.conv6, opt=1, permutation = 'sorted', num_sec=9, b_set = torch.FloatTensor([8, 8, 8, 8, 8, 7, 7, 6]).to(device),  add_noise = 1, noise_gain = 0.01, apply_noise_red=True)
        #4608
        x = self.bn6(x)
        x = self.relu6(x)
        x = self.avpool10(x)

        #x = self.conv7(x)
        x = conv_to_cim(x, self.conv7, opt=1, permutation = 'sorted', num_sec=9, b_set = torch.FloatTensor([8, 8, 8, 8, 8, 7, 7, 6]).to(device),  add_noise = 1, noise_gain = 0.01, apply_noise_red=True)
        #4608
        x = self.bn7(x)
        x = self.relu7(x)
        x = self.avpool11(x)

        x = self.mpool5(x)
        x = self.avpool12(x)

        x = x.reshape(x.size(0), -1)
        #x = self.fc1(x)
        #512
        x = fc_to_cim(x, self.fc1, opt=1, permutation = 'sorted', num_sec=8, b_set = torch.FloatTensor([8, 8, 8, 8, 8, 7, 7, 6]).to(device),  add_noise = 1, noise_gain = 0.01, apply_noise_red=True)
        return x

def test():
    net = VGG()
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

# test()
