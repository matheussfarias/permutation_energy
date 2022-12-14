import torch
import torch.nn as nn
import time
from functions import *
import matplotlib.pyplot as plt
import os
np.set_printoptions(suppress=True)
grid_perc= [100]
grid_sec= 1
grid_b_set= None

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

def fc_to_cim(x, layer, v_ref = 1, d = 12, wq = 8, adc = 12, permutation = 'random', grid = False, perc=[68, 27], num_sec=100, b_set = None, opt=0, add_noise=0, noise_gain=0, prints=False):
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
            result, valor, energy_value,active,s,noise, n_adcs = cim(x[i][np.newaxis, :].detach(), layer.weight.detach().T, v_ref, d, wq, adc, permutation = permutation, prints=prints,perc=perc, num_sec=num_sec, b_set = b_set, opt=opt, add_noise=add_noise, noise_gain=noise_gain)
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
            result, valor, energy_value,active,s,noise, n_adcs = cim(x[i][np.newaxis, :].detach(), layer.weight.detach().T, v_ref, d, wq, adc, permutation = permutation, prints=prints,perc=perc, num_sec=num_sec, b_set = b_set, opt=opt, add_noise=add_noise, noise_gain=noise_gain)
            f=open('./results/energy.txt','a')
            np.savetxt(f, [int(energy_value)], fmt='%1.3f', newline=", ")
            f.write("\n")
            f.close()
            f=open('./results/active.txt','a')
            np.savetxt(f, [active], fmt='%1.3f', newline=", ")
            f.write("\n")
            f.close()
            f=open('./results/noise.txt','a')
            np.savetxt(f, [noise], fmt='%1.3f', newline=", ")
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


def conv_to_cim(x, layer, v_ref = 1, d = 12, wq = 8, adc = 12, permutation = 'sorted', grid = False, perc=[68, 27], num_sec=1, b_set = [12,12,12,12,12,12,12,12], opt=0, add_noise=0, noise_gain=0, prints=False):
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
            result, valor, energy_value, active, s, noise, n_adcs = cim(A[i].detach(), B.detach(), v_ref, d, wq, adc, permutation = permutation, prints=prints, perc=perc, num_sec=num_sec, b_set=b_set, opt=opt, add_noise=add_noise, noise_gain=noise_gain)
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
            result, valor, energy_value, active, s, noise, n_adcs = cim(A[i].detach(), B.detach(), v_ref, d, wq, adc, permutation = permutation, prints=prints,perc=perc, num_sec=num_sec, b_set=b_set, opt=opt, add_noise=add_noise, noise_gain=noise_gain)
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
            np.savetxt(f, [noise], fmt='%1.3f', newline=", ")
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
        #x = self.conv1(x)
        #x = conv_to_cim(x, self.conv1, num_sec=100, b_set = (8*np.ones((100,8))).tolist())
        x = conv_to_cim(x, self.conv1, opt=0, permutation = 'random', num_sec=100, b_set = torch.FloatTensor([6, 6, 6, 6, 6, 6, 6, 6]).to(device), add_noise = 0, noise_gain = 0)
        x = self.relu1(x)

        #x = self.conv2(x)
        #x = conv_to_cim(x, self.conv2, num_sec=1000, b_set = (8*np.ones((1000,8))).tolist())
        x = conv_to_cim(x, self.conv2, opt=0, permutation = 'random', num_sec=1000, b_set = torch.FloatTensor([6, 6, 6, 6, 6, 6, 6, 6]).to(device), add_noise = 0, noise_gain = 0)
        x = self.relu2(x)
        
        #x = self.conv3(x)
        #x = conv_to_cim(x, self.conv3, num_sec=500, b_set = (8*np.ones((500,8))).tolist())
        x = conv_to_cim(x, self.conv3, opt=0, permutation = 'random', num_sec=500, b_set = torch.FloatTensor([6, 6, 6, 6, 6, 6, 6, 6]).to(device), add_noise = 0, noise_gain = 0)
        x = self.relu3(x)

        x = x.reshape(-1, 720)
        #x = self.fc1(x)
        #x = fc_to_cim(x, self.fc1, num_sec=720, b_set = (8*np.ones((720,8))).tolist())
        x = fc_to_cim(x, self.fc1, opt=0, permutation = 'random', num_sec=720, b_set = torch.FloatTensor([6, 6, 6, 6, 6, 6, 6, 6]).to(device), add_noise = 0, noise_gain = 0)
        return x

    def forward_a(self, x):
        """Forward propagation procedure"""
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = x.reshape(-1, 720)
        x = self.fc1(x)
        return x

class LeNet_5_Grid(nn.Module):
    """This class defines a standard DNN model based on LeNet_5"""

    def __init__(self):
        """Initialization"""

        super(LeNet_5_Grid, self).__init__()
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
        #x = self.conv1(x)
        ta = time.perf_counter()
        x = conv_to_cim(x, self.conv1, grid=True, perc = grid_perc, num_sec=grid_sec, b_set = grid_b_set)
        tb=time.perf_counter()
        print(tb-ta)
        exit()
        x = self.relu1(x)

        #x = self.conv2(x)
        x = conv_to_cim(x, self.conv2, grid=True, perc = grid_perc, num_sec=grid_sec, b_set = grid_b_set)
        x = self.relu2(x)
        
        #x = self.conv3(x)
        x = conv_to_cim(x, self.conv3, grid=True, perc = grid_perc, num_sec=grid_sec, b_set = grid_b_set)
        x = self.relu3(x)

        x = x.reshape(-1, 720)
        #x = self.fc1(x)
        x = fc_to_cim(x, self.fc1, grid=True, perc = grid_perc, num_sec=grid_sec, b_set = grid_b_set)
        return x
    
    def forward_a(self, x):
        """Forward propagation procedure"""
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = x.reshape(-1, 720)
        x = self.fc1(x)
        return x