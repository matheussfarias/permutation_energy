from imp import C_EXTENSION
import io
from tkinter import E
import numpy as np
import torch
import time
from prettytable import PrettyTable
import matplotlib.pyplot as plt

global device

cuda_act = True

if cuda_act == True:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device('cpu')

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    modules=0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
        modules+=1
    print(table)
    print(f"Total Trainable Params: {total_params}")
    print(f"Total Number of Trainable Modules: {modules}")
    return total_params

def compute_tiles_nn(A, B, B_signs, t, add_noise, noise_gain):
    tiles = [0]
    M = A.shape[0]
    K = A.shape[1]
    N = B.shape[1]
    q = B.shape[2]

    
    
    ans = torch.zeros(M,N,q).to(device)
    if t != "random":
        B_temp = []
        A_temp = []
        for i in range(K):
            for j in range(N):
                temp_row=0
                for k in range(q):
                    temp_row += (B[i][j][k])*2**(-k-1)
                B_temp.append((temp_row,B[i][j],(i,j),torch.sum(B[i][j])))
                A_temp.append((A[:,i],(i,j)))
        

        if t == "sorted":
            B_temp.sort(key=sorting_order)
        if t == "area":
            B_temp.sort(key=sorting_area)
        indeces = []
        for i in B_temp:
            indeces.append(i[2])

        B_new = []
        B_signs_new = []
        indeces.sort(key=sorting_area)
        A_new=[]
        indeces_b=[]
        for i in range(int(len(indeces)/(N))):
            for k in range(N):
                B_new.append((B.cpu()[indeces[i+k*K][0], indeces[i+k*K][1]]).to(device))
                B_signs_new.append((B_signs.cpu()[indeces[i+k*K][0], indeces[i+k*K][1]]).to(device))
        
        for i in range(len(indeces)):
            A_new.append(torch.FloatTensor(A.cpu()[:, indeces[i][0]]).to(device))
        A_new = torch.stack(A_new).to(device)
        A_final = []
        for i in range(int(len(A_new)/K)):
            A_final.append(A_new[i*K:(i+1)*K].t())
    
        A_final = torch.stack(A_final)
        
        B_new = torch.stack(B_new).reshape(B.shape)
        B_signs_new = torch.stack(B_signs_new).reshape(B_signs.shape)

        B_pim = []
        B_signs_pim = []
        for i in range(N):
            for j in range(K):
                B_pim.append(B_new[j][i])

        B_pim = torch.stack(B_pim).reshape((N,K,q))

        B_signs_pim = (-1)**B_signs_new.t()

        B_pim_signed = []
        for k in range (N):
            for i in range(K):
                B_pim_signed.append(B_pim[k][i]*B_signs_pim[k][i])
        
        B_pim_signed = torch.stack(B_pim_signed).reshape(N,K,q)

        results_pims = []
        results_pims_final = []

        for j in range (N):
            for i in range(K):
                results_pims.append(torch.outer(A_final[j][:,i], B_pim_signed[j][i]))
            results_pims_final.append(torch.stack(results_pims).reshape(K, M, q))
            results_pims=[]

        tiles = results_pims_final
        powers = torch.FloatTensor([2**(-i) for i in range(0,q)]).to(device)
        powers_q = torch.FloatTensor([2**(q-i-1) for i in range(0,q)]).to(device)
        mean_temp = torch.mean(torch.mul(B_pim_signed,powers), axis=2)
        #ans = np.sum(results_pims_final,axis=1)
    else:
        A_final = A
        B_new = B
        B_signs_new = B_signs

        B_pim = []
        B_signs_pim = []
        for i in range(N):
            for j in range(K):
                B_pim.append(B_new[j][i])


        B_pim = torch.stack(B_pim).reshape((N,K,q)).to(device)

        B_signs_pim = (-1)**B_signs_new.t()

        B_pim_signed = []
        for k in range (N):
            for i in range(K):
                B_pim_signed.append(B_pim[k][i]*B_signs_pim[k][i])
        
        B_pim_signed = torch.stack(B_pim_signed).reshape(N,K,q).to(device)

        results_pims = []
        results_pims_final = []
        for j in range (N):
            for i in range(K):
                results_pims.append(torch.outer(A_final[:,i], B_pim_signed[j][i]))
            results_pims_final.append(torch.stack(results_pims).reshape(K, M, q))
            results_pims=[]

        tiles = results_pims_final
        #ans = np.sum(results_pims_final,axis=1)
        powers = torch.FloatTensor([2**(-i) for i in range(0,q)]).to(device)
        powers_q = torch.FloatTensor([2**(q-i-1) for i in range(0,q)]).to(device)
        mean_temp = torch.mean(torch.mul(B_pim_signed,powers), axis=2)

    s = torch.zeros(q+1).to(device)
    s_n = torch.zeros((N,q+1)).to(device)

    for k in range(N):
        for i in range(K):
            for j in range(q):
                if B_pim_signed[k][i][j]!=0:
                    s_n[k][q-j]+=1
                    s[q-j]+=1

                    break
                if j==q-1:
                    s_n[k][0]+=1
                    s[0]+=1
    
    if t!='random':
        sections=[]
        for j in range(N):
            sec=[]
            for i in range (q+1):
                if i ==q:
                    sec.append(B_pim_signed[j][int(torch.sum(s_n[j][:i])):])
                    break
                sec.append(B_pim_signed[j][int(torch.sum(s_n[j][:i])):int(torch.sum(s_n[j][:(i+1)]))])
            sections.append(sec)
        noise_value=0
        for j in range(N):
            for i in range(q+1):
                if sections[j][i].size()[0]!=0:
                    temp = torch.zeros(sections[j][i].shape).to(device)
                    for a in range(temp.size()[0]):
                        for b in range(temp.size()[1]):
                            temp[a][b]=b-a+temp.size()[0]

                    posi_1 = (torch.mul(abs(sections[j][i]),temp)-(q-i)).to(device)
                    posi_2 = (posi_1>=0).type(torch.int)
                    noise = abs(torch.mul(posi_1, posi_2))
                    noise_value += torch.sum(noise/(torch.sum(abs(sections[j][i]))+1e-12))
                    if add_noise:
                        sign = torch.rand(noise.shape).to(device)
                        B_pim_signed[j][int(torch.sum(s_n[j][:i])):int(torch.sum(s_n[j][:(i+1)]))] += (-1)**((sign > 0.5).type(torch.int8))*noise_gain*noise
    else:
        num_sections = 2
        sections=[]
        B_pim_signed_edit = torch.reshape(B_pim_signed, (N, num_sections, int(K/num_sections), q))
        noise_value=0
        for j in range(N):
            for i in range(num_sections):
                temp = torch.zeros(B_pim_signed_edit[j][i].shape)
                for a in range(temp.size()[0]):
                    for b in range(temp.size()[1]):
                        temp[a][b]=b-a+temp.size()[0]
                posi_1 = torch.mul(abs(B_pim_signed_edit[j][i]),temp)
                posi_2 = (posi_1>=0).type(torch.int)
                noise = abs(torch.mul(posi_1, posi_2))
                noise_value += torch.sum(noise/(torch.sum(abs(B_pim_signed_edit[j][i])))+1e-12)
                if add_noise:
                    sign = torch.rand(noise.shape)
                    B_pim_signed_edit[j][i] = B_pim_signed_edit[j][i] + (-1)**((sign > 0.5).type(torch.int))*noise_gain*noise
                
        B_pim_signed = torch.reshape(B_pim_signed_edit, (N, K, q))
    return tiles, B_pim_signed, A_final, mean_temp, s, s_n, noise_value

def adc(A, B_digital, C_correct, v_ref, b, permutation, perc, num_sec, b_set, s_n, opt):
    N = B_digital.shape[0]
    K = B_digital.shape[1]
    q = B_digital.shape[2]


    if permutation=='random':
        M = A.shape[0]

        max_outputs_1 = max_normal(A, B_digital, M, K, N, q, permutation, v_ref, perc, num_sec, s_n, opt)
        max_outputs_2 = max_pos(A, B_digital, M, K, N, q, permutation, v_ref, perc, num_sec, s_n, opt)
        max_outputs_3 = max_neg(A, B_digital, M, K, N, q, permutation, v_ref, perc, num_sec, s_n, opt)

        max_outputs_1 = torch.abs(max_outputs_1)
        max_outputs_2 = torch.abs(max_outputs_2)
        max_outputs_3 = torch.abs(max_outputs_3)

        max_1 = torch.max(max_outputs_1, max_outputs_2)
        max_2 = torch.max(max_1, max_outputs_3)
    else:
        M=A.shape[1]

        max_outputs_1 = max_normal(A, B_digital, M, K, N, q, permutation, v_ref, perc, num_sec, s_n, opt)
        max_outputs_2 = max_pos(A, B_digital, M, K, N, q, permutation, v_ref, perc, num_sec,  s_n, opt)
        max_outputs_3 = max_neg(A, B_digital, M, K, N, q, permutation, v_ref, perc, num_sec,  s_n, opt)

        max_outputs_1 = torch.abs(max_outputs_1)
        max_outputs_2 = torch.abs(max_outputs_2)
        max_outputs_3 = torch.abs(max_outputs_3)

        max_1 = torch.max(max_outputs_1, max_outputs_2)
        max_2 = torch.max(max_1, max_outputs_3)
    energy=0

    q_step=[]
    if b_set ==None:
        for i in range(num_sec):
            q_step.append(torch.abs(max_2[i])/(torch.pow(2,b_set[i])-1 + 1e-12) + 1e-12)
    else: 
        for j in range(N): 
            for i in range(num_sec): 
                if b_set[i] == torch.zeros(len(b_set[i])).tolist():
                    q_step.append(torch.zeros(max_2[j][i].shape))
                else: 
                    q_step.append(torch.abs(max_2[j][i])/(torch.pow(2,b_set[i])-1 + 1e-12) + 1e-12)

    #q_step_0 = torch.abs(max_2)/(2**(b)-1) + 1e-12
    #q_step_1 = torch.abs(max_2)/(2**(b-1)-1) + 1e-12
    #q_step_2 = torch.abs(max_2)/(2**(b-2)-1) + 1e-12
    q_step = torch.stack(q_step).reshape(max_2.shape).to(device)

    digital_without_sign = []
    for i in range(C_correct.shape[0]):
        for j in range(num_sec):
            digital_without_sign.append(torch.abs(torch.round(C_correct[i][j]/(q_step[i][j]+1e-12))*q_step[i][j]))
        #digital_without_sign.append(torch.abs(torch.round(C_correct[i][0]/q_step[0][i][0])*q_step[0][i][0]))
        #digital_without_sign.append(torch.abs(torch.round(C_correct[i][1]/q_step[1][i][1])*q_step[1][i][1]))
        #digital_without_sign.append(torch.abs(torch.round(C_correct[i][2]/q_step[2][i][2])*q_step[2][i][2]))
    digital_without_sign = torch.stack(digital_without_sign).reshape(C_correct.shape).to(device)
    energy = 0
    s = np.zeros(q+1)
    z = np.zeros(q+1)
    n_adcs=0

    if b_set==None:
        for j in range(C_correct.shape[0]):
            sec = (torch.sum(C_correct[j],axis=1) != 0).type(torch.int)
            for i in range(num_sec):
                energy+=torch.sum(sec[i])*2**(b-i)
                #sum = torch.sum(sec[i])
                #s[sum]+=1
                #print(len(sec[i]))
                #exit()
                #for k in range(len(sec[i])):
                #    if sec[i][k]==1:
                #        z[q-k]+=1
                #        exit()
                #        break
                #    else:
                #        z[0]+=1
                #print(sum)
                #print(s)
                #print(sec)

    else:
        for j in range(C_correct.shape[0]):
            sec = (torch.sum(C_correct[j],axis=1) != 0).type(torch.int)
            for i in range(num_sec):
                energy+=torch.sum(torch.mul(sec[i],torch.pow(2, b_set[i])))
                n_adcs+=torch.sum((sec[i]>0).type(torch.int))
                #sum = torch.sum(sec[i])
                #s[sum]+=1
                #for k in range(len(sec[i])):
                #    if sec[i][k]==1:
                #        z[q-k]+=1
                #        break
                #    if k == q-1:
                #        z[0]+=1
        #print(s)
        #print(z)
        #exit()

    digital_outputs = torch.zeros(digital_without_sign.shape).to(device)
    output_signs = (C_correct<0).type(torch.int)
    digital_outputs = torch.mul(digital_without_sign, (-1)**output_signs).to(torch.float).to(device)

    return digital_outputs, n_adcs, energy

def quantize(x,q):
    low = torch.min(x)
    x_shifted = (x-low)
    high = torch.max(x_shifted)
    x_shifted_scaled = x_shifted*(2**q-1)/high
    x_quantized = (torch.floor(x_shifted_scaled.detach().clone()+.5)).type(torch.int16)
    return x_quantized, (low, high)

def dequantize(x, extra_args, q):
    low, high = extra_args 
    signs = (x<0).type(torch.int)
    x_shifted = abs(x).type(torch.float32)*high/(2**q-1) 
    x = x_shifted + low
    x = np.multiply(x,(-1)**signs)
    return x 

def convert_to_bin(x,q):
    x = x.item()
    result=[]
    for i in range(q):
        result.append(x%2)
        x=x//2
    result.reverse()
    return result

def ef_convert_to_neg_bin(x,N):
    result = []
    i=0
    g1 = (x>=1).type(torch.float)
    result.append(g1)
    x = x-g1

    while(i<N-1):
        db = 2*x
        g1 = (db>=1).type(torch.float)
        result.append(g1)
        x = db - g1
        i+=1
    result = torch.transpose(torch.stack(result), 1, 2)
    result = torch.transpose(result, 0, 2)
    return result

def filling_array(x, q, powers_2 = False):
    B_digital=[]
    ta = time.perf_counter()
    if powers_2:
        quantized, args = quantize(x,q)
        #dequantized = dequantize(quantized,args,q)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                B_digital.append(convert_to_bin(quantized[i][j],q))
        return torch.tensor(B_digital).reshape(x.shape[0],x.shape[1],q).to(device), args
    else:
        #for i in range(x.shape[0]):
        #    for j in range(x.shape[1]):
        #        B_digital.append(convert_to_neg_bin(x[i][j],q))
        #return torch.FloatTensor(B_digital).reshape(x.shape[0],x.shape[1],q).to(device), 0
        return ef_convert_to_neg_bin(x,q), 0


def max_normal(A,B_digital, M, K, N, q, permutation, v_ref,perc, num_sec,  s_n, opt):
    results_pims=[]
    results_pims_final=[]
    A = torch.ones(A.shape).to(device)*v_ref
    if permutation == 'random':
        for j in range(N):
            for i in range(K):
                results_pims.append(torch.outer(A[:,i], B_digital[j][i]))
            results_pims_final.append(torch.stack(results_pims).reshape(K, M, q).to(device))
            results_pims=[]
    
    else:
        for j in range (N):
            for i in range(K):
                results_pims.append(torch.outer(A[j][:,i], B_digital[j][i]))
            results_pims_final.append(torch.stack(results_pims).reshape(K, M, q).to(device))
            results_pims=[]
        
    tiles_normal = results_pims_final

    C_wait = tiles_normal
    C_tiles = []
    C_after_sum=[]

    #perc = [68.2, 27.2]
    #print([0,int(np.floor((perc[0]/100)*K))])
    #print([int(np.ceil((perc[0]/100)*K)),int(np.floor(((perc[0]+perc[1])/100)*K))])
    #print([int(np.ceil(((perc[0]+perc[1])/100)*K)),K])
    C_wait = torch.stack(C_wait).to(device)
    if opt:
        for j in range(N):
            for i in range (num_sec):
                if i ==num_sec-1:
                    C_tiles.append(torch.sum(C_wait[j][np.sum(s_n[j][:i], dtype=int):],axis=0))
                    break
                C_tiles.append(torch.sum(C_wait[j][np.sum(s_n[j][:i], dtype=int):np.sum(s_n[j][:(i+1)], dtype=int)],axis=0))

            C_tiles = torch.stack(C_tiles).to(device)
            C_after_sum.append(C_tiles)
            C_tiles=[]
        C_wait = torch.stack(C_after_sum).reshape(N,num_sec,M,q).to(device)
    else:    
        C_wait = C_wait.reshape(N, num_sec, int(K/num_sec), M, q)
        C_wait = torch.sum(C_wait, axis=2)
        #
        #for j in range(N):
        #    for i in range (num_sec):
        #        if i ==num_sec-1:
        #            C_tiles.append(torch.sum(C_wait[j][np.sum(perc[:i], dtype=int):],axis=0))
        #            break
        #        C_tiles.append(torch.sum(C_wait[j][np.sum(perc[:i], dtype=int):np.sum(perc[:(i+1)], dtype=int)],axis=0))
        #
        #    C_tiles = torch.stack(C_tiles).to(device)
        #    C_after_sum.append(C_tiles)
        #    C_tiles=[]
        #C_wait = torch.stack(C_after_sum).reshape(N,num_sec,M,q).to(device)

    return C_wait

def max_pos(A,B_digital, M, K, N, q, permutation, v_ref, perc, num_sec, s_n, opt):
    results_pims=[]
    results_pims_final=[]
    B_digital = torch.where(B_digital > 0, 1, 0)
    A = torch.ones(A.shape).to(device)*v_ref
    if permutation == 'random':
        for j in range(N):
            for i in range(K):
                results_pims.append(torch.outer(A[:,i], B_digital[j][i]))
            results_pims_final.append(torch.stack(results_pims).reshape(K, M, q).to(device))
            results_pims=[]
    
    else:
        for j in range (N):
            for i in range(K):
                results_pims.append(torch.outer(A[j][:,i], B_digital[j][i]))
            results_pims_final.append(torch.stack(results_pims).reshape(K, M, q).to(device))
            results_pims=[]
    
    tiles_normal = results_pims_final

    C_wait = tiles_normal
    C_tiles = []
    C_after_sum=[]
    #perc = [68.2, 27.2]
    #print([0,int(np.floor((perc[0]/100)*K))])
    #print([int(np.ceil((perc[0]/100)*K)),int(np.floor(((perc[0]+perc[1])/100)*K))])
    #print([int(np.ceil(((perc[0]+perc[1])/100)*K)),K])
    C_wait = torch.stack(C_wait).to(device)
    if opt:
        for j in range(N):
            for i in range (num_sec):
                if i ==num_sec-1:
                    C_tiles.append(torch.sum(C_wait[j][np.sum(s_n[j][:i], dtype=int):],axis=0))
                    break
                C_tiles.append(torch.sum(C_wait[j][np.sum(s_n[j][:i], dtype=int):np.sum(s_n[j][:(i+1)], dtype=int)],axis=0))

            C_tiles = torch.stack(C_tiles).to(device)
            C_after_sum.append(C_tiles)
            C_tiles=[]
        C_wait = torch.stack(C_after_sum).reshape(N,num_sec,M,q).to(device)
    else:
        C_wait = C_wait.reshape(N, num_sec, int(K/num_sec), M, q)
        C_wait = torch.sum(C_wait, axis=2)
        #for j in range(N):
        #    for i in range (num_sec):
        #        if i ==num_sec-1:
        #            C_tiles.append(torch.sum(C_wait[j][np.sum(perc[:i], dtype=int):],axis=0))
        #            break
        #        C_tiles.append(torch.sum(C_wait[j][np.sum(perc[:i], dtype=int):np.sum(perc[:(i+1)], dtype=int)],axis=0))
        #
        #    C_tiles = torch.stack(C_tiles).to(device)
        #    C_after_sum.append(C_tiles)
        #    C_tiles=[]
        #C_wait = torch.stack(C_after_sum).reshape(N,num_sec,M,q).to(device)
    '''
    C_wait = tiles_normal
    C_tiles = []
    C_after_sum=[]
    for j in range(N):
        C_tiles.append(torch.sum(C_wait[j][0:1+int(np.floor((perc[0]/100)*K))],axis=0))
        C_tiles.append(torch.sum(C_wait[j][int(np.ceil((perc[0]/100)*K)):1+int(np.floor(((perc[0]+perc[1])/100)*K))],axis=0))
        C_tiles.append(torch.sum(C_wait[j][int(np.ceil(((perc[0]+perc[1])/100)*K)):],axis=0))

        C_tiles = torch.stack(C_tiles).to(device)
        C_after_sum.append(C_tiles)
        C_tiles=[]

    C_wait1 = torch.stack(C_after_sum).reshape(N,3,M,q).to(device)
    '''
    return C_wait

def max_neg(A,B_digital, M, K, N, q, permutation, v_ref,perc, num_sec, s_n, opt):
    results_pims=[]
    results_pims_final=[]
    B_digital = torch.where(B_digital > 0, -1, B_digital.to(int))
    A = torch.ones(A.shape).to(device)*v_ref
    if permutation == 'random':
        for j in range(N):
            for i in range(K):
                results_pims.append(torch.outer(A[:,i], B_digital[j][i]))
            results_pims_final.append(torch.stack(results_pims).reshape(K, M, q).to(device))
            results_pims=[]
    
    else:
        for j in range (N):
            for i in range(K):
                results_pims.append(torch.outer(A[j][:,i], B_digital[j][i]))
            results_pims_final.append(torch.stack(results_pims).reshape(K, M, q).to(device))
            results_pims=[]
    
    tiles_normal = results_pims_final

    C_wait = tiles_normal
    C_tiles = []
    C_after_sum=[]

    #perc = [68.2, 27.2]
    #print([0,int(np.floor((perc[0]/100)*K))])
    #print([int(np.ceil((perc[0]/100)*K)),int(np.floor(((perc[0]+perc[1])/100)*K))])
    #print([int(np.ceil(((perc[0]+perc[1])/100)*K)),K])
    C_wait = torch.stack(C_wait).to(device)
    if opt:
        for j in range(N):
            for i in range (num_sec):
                if i ==num_sec-1:
                    C_tiles.append(torch.sum(C_wait[j][np.sum(s_n[j][:i], dtype=int):],axis=0))
                    break
                C_tiles.append(torch.sum(C_wait[j][np.sum(s_n[j][:i], dtype=int):np.sum(s_n[j][:(i+1)], dtype=int)],axis=0))

            C_tiles = torch.stack(C_tiles).to(device)
            C_after_sum.append(C_tiles)
            C_tiles=[]
        C_wait = torch.stack(C_after_sum).reshape(N,num_sec,M,q).to(device)
    else:
        C_wait = C_wait.reshape(N, num_sec, int(K/num_sec), M, q)
        C_wait = torch.sum(C_wait, axis=2)
        #for j in range(N):
        #    for i in range (num_sec):
        #        if i ==num_sec-1:
        #            C_tiles.append(torch.sum(C_wait[j][np.sum(perc[:i], dtype=int):],axis=0))
        #            break
        #        C_tiles.append(torch.sum(C_wait[j][np.sum(perc[:i], dtype=int):np.sum(perc[:(i+1)], dtype=int)],axis=0))
        #
        #    C_tiles = torch.stack(C_tiles).to(device)
        #    C_after_sum.append(C_tiles)
        #    C_tiles=[]
        #C_wait = torch.stack(C_after_sum).reshape(N,num_sec,M,q).to(device)
    
    return C_wait

def convert_to_neg_bin(x,N):
    result=[]
    i=0
    if x>=1:
        x=x-1
        result.append(1)
    else:
        result.append(0)
    while(i<N-1):
        if x*2<1:
            result.append(0)
            x=x*2
        else:
            result.append(1)
            x=x*2 - 1
        i+=1
    return result

def dac(x,d,v_ref):
    maximum = torch.max(x)
    return torch.round((x/(maximum+1e-12))*(2**d-1))*v_ref/(2**d-1), maximum, v_ref

def sorting_order(A):
    return A[0] #sorting by lower to higher

def sorting_area(A):
    return A[-1] #sorting by number of 1s

def stochastic_rounding(x):
    floor = torch.floor(x)
    val = x - floor
    sample = torch.rand(x.shape)
    go_floor = sample<=val
    shape_prev = x.shape
    f_go_floor = torch.flatten(go_floor)
    f_x = torch.flatten(x)
    for i in range(len(f_x)):
        if f_go_floor[i]:
            f_x[i] = torch.floor(f_x[i])
        else:
            f_x[i] = torch.ceil(f_x[i])
    return f_x.reshape(shape_prev)



def cim(A, B, v_ref, d, q, b, permutation, prints, perc, num_sec, b_set, opt, add_noise, noise_gain):
    ta = time.perf_counter()
    if prints:
        print('Starting CIMulator...\n')
    # dac settings
    v_ref=v_ref
    d=d

    # weight matrix settings
    q = q
    powers = torch.FloatTensor([2**(-i) for i in range(0,q)]).to('cuda')

    # adc settings
    b=b
    permutation = permutation
    K = A.shape[1]
    M = A.shape[0]
    N = B.shape[1]
    # input and weights
    # digital A
    if prints:
        print('A: ' + str(A))
    
    # analog B
    B_signs = (B<0).type(torch.int).to(device)
    if prints:
        print('B: ' + str(B))
        print('B_signs: ' + str(B_signs))

    # correct result
    C = torch.matmul(A,B).to(device)
    # conversions
    # converting A to analog
    t1=time.time()
    A_analog, maximum, v_ref = dac(A,d,v_ref)
    A_back = A_analog*(maximum/v_ref)
    err_dac = torch.sum(1/2*(A_back - A)**2)
    t2=time.time()
    if prints:
        print('\nError due to DAC: ' + str(err_dac))
        print('Time due to DAC: ' + str(t2-t1))
    # converting B to digital
    t1=time.time()
    B_digital, args = filling_array(torch.abs(B), q, False)
    ta = time.perf_counter()
    B_back = torch.sum(torch.mul(B_digital, powers), axis=-1)
    #B_digital, args = filling_array(torch.abs(B), q)
    #B_back = dequantize(torch.sum(torch.mul(B_digital, powers_q), axis=-1), args, q)
    err_mm = torch.sum(1/2*(torch.abs(B) - B_back)**2)    

    t2=time.time()
    if prints:
        print('Error due to matrix B conversion: ' + str(err_mm))
        print('Time due to matrix B conversion: ' + str(t2-t1))
    
    # operating
    # actual MM
    t1=time.time()
    C_wait, B_new, A_new, means, s, s_n, noise = compute_tiles_nn(A_analog, B_digital, B_signs, permutation, add_noise, noise_gain)

    C_tiles = []

    C_after_sum=[]

    C_wait_opt = torch.stack(C_wait).to(device)
    C_wait = torch.stack(C_wait).to(device)
    
    C_wait = C_wait.reshape(N, num_sec, int(K/num_sec), M, q)
    C_wait = torch.sum(C_wait, axis=2)
    
    C_tiles = []
    
    C_after_sum=[]
    for j in range(N):
        for i in range (q+1):
            if i ==q:
                C_tiles.append(torch.sum(C_wait_opt[j][int(torch.sum(s_n[j][:i])):],axis=0))
                break
            C_tiles.append(torch.sum(C_wait_opt[j][int(torch.sum(s_n[j][:i])):int(torch.sum(s_n[j][:(i+1)]))],axis=0))
        C_tiles = torch.stack(C_tiles).to(device)
        C_after_sum.append(C_tiles)
        C_tiles=[]
    C_wait_opt = torch.stack(C_after_sum).reshape(N,q+1,M,q).to(device)
    

    
    t2=time.time()
    if prints:
        print('Time due to MM: ' + str(t2-t1))

    # scaling for each column power

    if opt==1:
        #C_wait2 = torch.multiply(C_wait_opt, powers_q)
        C_wait2 = torch.multiply(C_wait_opt, powers)
        num_sec = q+1
    else:
        #C_wait2 = torch.multiply(C_wait, powers_q)
        C_wait2 = torch.multiply(C_wait, powers)
    C_compare = torch.sum(C_wait2, axis=1)

    #C_wait2 = torch.multiply(C_wait, powers_q)
    #C_compare = torch.sum(dequantize(C_wait2,args,q), axis=1)

    # converting output to digital
    t1=time.time()

    if opt==1:
        digital_outputs, n_adcs, energy_value = adc(A_new, B_new, C_wait_opt, v_ref, b, permutation, perc, num_sec, b_set, s_n, opt)
    else:
        digital_outputs, n_adcs, energy_value = adc(A_new, B_new, C_wait, v_ref, b, permutation, perc, num_sec, b_set, s_n, opt)
    #digital_outputs = torch.sum(dequantize(digital_outputs,args,q), axis=1)
    #digital_outputs = torch.multiply(digital_outputs, powers_q)
    digital_outputs = torch.multiply(digital_outputs, powers)
    digital_outputs = torch.sum(digital_outputs, axis=1)
    #print(digital_outputs)
    #print(C_compare)
    #print(digital_outputs - C_compare)
    err_adc = torch.sum(1/2*(digital_outputs - C_compare)**2)

    #print(err_adc)
    t2=time.time()
    if prints:
        print('Error due to ADC: ' + str(err_adc))
        print('Energy due to ADC: ' + str(energy_value))
        print('Time due to ADC: ' + str(t2-t1))

    # horizontal add
    #C_wait3 = dequantize(torch.sum(digital_outputs, axis=-1), args,q)
    C_wait3 = torch.sum(digital_outputs, axis=-1)

    # uncomment if dont want to use adc
    #C_wait3 = dequantize(torch.sum(C_compare,axis=-1), args, q)
    #C_wait3 = torch.sum(C_compare,axis=-1)

    # converting output back to digital
    result = (C_wait3*maximum/v_ref).t()

    # calculating error
    error = torch.sum(1/2*(result - C)**2)
    active=0
    if prints:
        print('\nResults...')
        print('Expected result: ' + str(C))
        print('Obtained result: ' + str(result))
        print('Total error: ' + str(error))
    tb = time.perf_counter()
    print (tb-ta)
    exit()
    return result, (error, err_adc), energy_value, active, s, noise, n_adcs