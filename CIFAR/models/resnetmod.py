'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from functions import *

cuda_act = True

if cuda_act == True:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device('cpu')

def preprocessing(x, w, padding, stride):
    inp_unf = torch.nn.functional.unfold(x, (w.shape[2], w.shape[3]), padding = padding, stride=stride)
    A = inp_unf.transpose(1, 2)
    B = w.view(w.size(0), -1).t()
    return A,B

def postprocessing(x, exp_x):
    x = x.transpose(1, 2)
    x = x.view(exp_x.shape)
    return x

def fc_to_cim(x, layer, v_ref = 1, d = 12, wq = 8, adc = 12, permutation = 'random', grid = False, perc=[68, 27], num_sec=100, b_set = None, opt=0, add_noise=0, noise_gain=0, prints = False):
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
            np.savetxt(f, [s], fmt='%1.3f', newline=", ")
            f.write("\n")
            f.close()
            f=open('./results/n_adcs.txt','a')
            np.savetxt(f, [n_adcs], fmt='%1.3f', newline=", ")
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

    x = result_total
    return x


def conv_to_cim(x, layer, v_ref = 1, d = 12, wq = 8, adc = 12, permutation = 'sorted', grid = False, perc=[68, 27], num_sec=1, b_set = [12,12,12,12,12,12,12,12], opt=0, add_noise=0, noise_gain=0, prints=False):
    opt_grind = 0.1
    print(perc)
    dim = layer(x)

    A,B = preprocessing(x, layer.weight, layer.padding, layer.stride)
    #sns_data = B.flatten().detach().numpy()
    #sns.displot(sns_data, color = 'crimson', kde=True)
    #plt.xlabel("Weights")
    #sns.despine(left=False, bottom = False, top = False, right = False)
    #plt.tight_layout()
    #plt.savefig("./plots/conv3.pdf")
    #plt.show()
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
    print(result_total.shape)

    x = postprocessing(result_total, dim)
    return x



class ResNetMod(nn.Module):
    def __init__(self):
        super(ResNetMod, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)

        #normal
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)

        #shortcut
        self.seq1 = nn.Sequential()
        self.relu3 = nn.ReLU(inplace=True)

        #normal
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU(inplace=True)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3,stride=1, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(128)
        self.relu5 = nn.ReLU(inplace=True)

        #shortcut
        self.conv6 = nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False)
        self.bn6 = nn.BatchNorm2d(128)
        self.relu6 = nn.ReLU(inplace=True)

        #normal
        self.conv7 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(256)
        self.relu7 = nn.ReLU(inplace=True)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3,stride=1, padding=1, bias=False)
        self.bn8 = nn.BatchNorm2d(256)
        self.relu8 = nn.ReLU(inplace=True)

        #shortcut
        self.conv9 = nn.Conv2d(128, 256, kernel_size=1, stride=2, bias=False)
        self.bn9 = nn.BatchNorm2d(256)
        self.relu9 = nn.ReLU(inplace=True)

        #normal
        self.conv10 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn10 = nn.BatchNorm2d(512)
        self.relu10 = nn.ReLU(inplace=True)
        self.conv11 = nn.Conv2d(512, 512, kernel_size=3,stride=1, padding=1, bias=False)
        self.bn11 = nn.BatchNorm2d(512)
        self.relu11 = nn.ReLU(inplace=True)

        #shortcut
        self.conv12 = nn.Conv2d(256, 512, kernel_size=1, stride=2, bias=False)
        self.bn12 = nn.BatchNorm2d(512)
        self.relu12 = nn.ReLU(inplace=True)

        self.linear = nn.Linear(512, 10)

    def forward(self, x):
        #out = self.conv1(x)
        #out = conv_to_cim(x, self.conv1, permutation = 'random', num_sec=27, b_set = (np.multiply(np.array([8, 8, 8, 7, 5, 4, 2, 1]),np.ones((27,8)))).tolist(), opt=0)
        out = conv_to_cim(x, self.conv1, permutation = 'random', num_sec=9, b_set = torch.FloatTensor([8, 8, 8, 7, 5, 4, 2, 1]).to(device), opt=0)
        out = self.bn1(out)
        out = self.relu1(out)
        out1 = out.detach()

        #out = self.conv2(out)
        #out = conv_to_cim(out, self.conv2, permutation = 'random', num_sec=576, b_set = (np.multiply(np.array([8, 8, 8, 7, 5, 4, 2, 1]),np.ones((576,8)))).tolist(), opt=0)
        out = conv_to_cim(out, self.conv2, permutation = 'random', num_sec=9, b_set = torch.FloatTensor([8, 8, 8, 7, 5, 4, 2, 1]).to(device), opt=0)
        out = self.bn2(out)
        out = self.relu2(out)
        
        #out = self.conv3(out)
        #out = conv_to_cim(out, self.conv3, permutation = 'random', num_sec=576, b_set = (np.multiply(np.array([8, 8, 8, 7, 5, 4, 2, 1]),np.ones((576,8)))).tolist(), opt=0)
        out = conv_to_cim(out, self.conv3, permutation = 'random', num_sec=9, b_set = torch.FloatTensor([8, 8, 8, 7, 5, 4, 2, 1]).to(device), opt=0)
        out = self.bn3(out)
        out_shortcut = self.seq1(out1)
        out = out + out_shortcut
        out = self.relu3(out)
        
        out2 = out.detach()

        #out = self.conv4(out)
        #out = conv_to_cim(out, self.conv4, permutation = 'random', num_sec=576, b_set = (np.multiply(np.array([8, 8, 8, 7, 5, 4, 2, 1]),np.ones((576,8)))).tolist(), opt=0)
        out = conv_to_cim(out, self.conv4, permutation = 'random', num_sec=9, b_set = torch.FloatTensor([8, 8, 8, 7, 5, 4, 2, 1]).to(device), opt=0)
        out = self.bn4(out)
        out = self.relu4(out)

        #out = self.conv5(out)
        #out = conv_to_cim(out, self.conv5, permutation = 'random', num_sec=1152, b_set = (np.multiply(np.array([8, 8, 8, 7, 5, 4, 2, 1]),np.ones((1152,8)))).tolist(), opt=0)
        out = conv_to_cim(out, self.conv5, permutation = 'random', num_sec=9, b_set = torch.FloatTensor([8, 8, 8, 7, 5, 4, 2, 1]).to(device), opt=0)
        out = self.bn5(out)

        #out_shortcut = self.conv6(out2)
        #out_shortcut = conv_to_cim(out2, self.conv6, permutation = 'random', num_sec=64, b_set = (np.multiply(np.array([8, 8, 8, 7, 5, 4, 2, 1]),np.ones((64,8)))).tolist(), opt=0)
        out_shortcut = conv_to_cim(out2, self.conv6, permutation = 'random', num_sec=8, b_set = torch.FloatTensor([8, 8, 8, 7, 5, 4, 2, 1]).to(device), opt=0)
        out_shortcut = self.bn6(out_shortcut)
        out = out + out_shortcut
        out = self.relu6(out)

        out3 = out.detach()

        #out = self.conv7(out)
        #out = conv_to_cim(out, self.conv7, permutation = 'random', num_sec=1152, b_set = (np.multiply(np.array([8, 8, 8, 7, 5, 4, 2, 1]),np.ones((1152,8)))).tolist(), opt=0)
        out = conv_to_cim(out, self.conv7, permutation = 'random', num_sec=9, b_set = torch.FloatTensor([8, 8, 8, 7, 5, 4, 2, 1]).to(device), opt=0)
        out = self.bn7(out)
        out = self.relu7(out)
        
        #out = self.conv8(out)
        #out = conv_to_cim(out, self.conv8, permutation = 'random', num_sec=2304, b_set = (np.multiply(np.array([8, 8, 8, 7, 5, 4, 2, 1]),np.ones((2304,8)))).tolist(), opt=0)
        out = conv_to_cim(out, self.conv8, permutation = 'random', num_sec=9, b_set = torch.FloatTensor([8, 8, 8, 7, 5, 4, 2, 1]).to(device), opt=0)
        out = self.bn8(out)
        #out_shortcut = self.conv9(out3)
        #out_shortcut = conv_to_cim(out3, self.conv9, permutation = 'random', num_sec=256, b_set = (np.multiply(np.array([8, 8, 8, 7, 5, 4, 2, 1]),np.ones((256,8)))).tolist(), opt=0)
        out_shortcut = conv_to_cim(out3, self.conv9, permutation = 'random', num_sec=8, b_set = torch.FloatTensor([8, 8, 8, 7, 5, 4, 2, 1]).to(device), opt=0)
        out_shortcut = self.bn9(out_shortcut)
        out = out + out_shortcut
        out = self.relu9(out)

        out4 = out.detach()

        #out = self.conv10(out)
        #out = conv_to_cim(out, self.conv10, permutation = 'random', num_sec=2304, b_set = (np.multiply(np.array([8, 8, 8, 7, 5, 4, 2, 1]),np.ones((2304,8)))).tolist(), opt=0)
        out = conv_to_cim(out, self.conv10, permutation = 'random', num_sec=9, b_set = torch.FloatTensor([8, 8, 8, 7, 5, 4, 2, 1]).to(device), opt=0)
        out = self.bn10(out)
        out = self.relu10(out)
        
        #out = self.conv11(out)
        #out = conv_to_cim(out, self.conv11, permutation = 'random', num_sec=4608, b_set = (np.multiply(np.array([8, 8, 8, 7, 5, 4, 2, 1]),np.ones((4608,8)))).tolist(), opt=0)
        out = conv_to_cim(out, self.conv11, permutation = 'random', num_sec=9, b_set = torch.FloatTensor([8, 8, 8, 7, 5, 4, 2, 1]).to(device), opt=0)
        out = self.bn11(out)
        #out_shortcut = self.conv12(out4)
        #out_shortcut = conv_to_cim(out4, self.conv12, permutation = 'random', num_sec=256, b_set = (np.multiply(np.array([8, 8, 8, 7, 5, 4, 2, 1]),np.ones((256,8)))).tolist(), opt=0)
        out_shortcut = conv_to_cim(out4, self.conv12, permutation = 'random', num_sec=8, b_set = torch.FloatTensor([8, 8, 8, 7, 5, 4, 2, 1]).to(device), opt=0)
        out_shortcut = self.bn12(out_shortcut)
        out = out + out_shortcut
        out = self.relu12(out)
        
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        #out = self.linear(out)
        #out = fc_to_cim(out, self.linear, permutation = 'random', num_sec=512, b_set = (np.multiply(np.array([8, 8, 8, 7, 5, 4, 2, 1]),np.ones((512,8)))).tolist(), opt=0)
        out = fc_to_cim(out, self.linear, permutation = 'random', num_sec=8, b_set = torch.FloatTensor([8, 8, 8, 7, 5, 4, 2, 1]).to(device), opt=0)
        
        return out    
