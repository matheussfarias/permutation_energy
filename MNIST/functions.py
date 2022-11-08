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
    M = A.shape[0]
    K = A.shape[1]
    N = B.shape[1]
    q = B.shape[2]

    if t != "random":
        powers = torch.FloatTensor([2**(-i) for i in range(-1,q-1)]).to(device)
        B_transpose = torch.transpose(B, 0, 1)
        B_signs_transpose = B_signs.T

        _, indeces = torch.sort(input = torch.sum(torch.mul(powers,B_transpose), axis=2), stable=True)
        stacked = B_transpose.reshape((B_transpose.shape[0]*B_transpose.shape[1], B_transpose.shape[2]))
        stacked_signs = B_signs_transpose.flatten()
        
        shift = torch.mul(torch.arange(N), (K)*torch.ones(N)).to(device)
        shift = shift.repeat_interleave(K).reshape(indeces.shape)
        shifted = (shift + indeces).flatten()
        shifted = shifted.type(torch.long)

        B_transpose_sorted = stacked[shifted].reshape(B_transpose.shape)
        B_signs_transpose_sorted = stacked_signs[shifted].reshape(B_signs_transpose.shape)

        B_new = torch.transpose(B_transpose_sorted, 0, 1)
        B_signs_new = B_signs_transpose_sorted.T

        A_transpose = A.T
        A_new=A_transpose[indeces].reshape(N*K,M)
    

        A_new = A_new.reshape(N,K,M)
        A_final = torch.transpose(A_new, 1, 2)

        B_pim = torch.transpose(B_new, 0, 1)
        B_signs_pim = (-1)**B_signs_new.t()
        B_signs_pim = B_signs_pim.reshape(B_signs_pim.shape[0],B_signs_pim.shape[1],1).repeat(1,1,q).flatten()
        B_signs_pim = B_signs_pim.reshape(B_pim.shape)
        B_pim_signed = torch.mul(B_pim, B_signs_pim)

        A_final = torch.transpose(A_final, 1, 2)
        C_wait = torch.einsum('bij,bik->bijk', A_final, B_pim_signed)
        A_final = torch.transpose(A_final, 1, 2)

    else:
        A_final = A
        B_new = B
        B_signs_new = B_signs

        B_signs_pim = []

        B_pim = torch.transpose(B_new, 0, 1)
        
        B_signs_pim = (-1)**B_signs_new.t()
        B_signs_pim = B_signs_pim.reshape(B_signs_pim.shape[0],B_signs_pim.shape[1],1).repeat(1,1,q).flatten()
        B_signs_pim = B_signs_pim.reshape(B_pim.shape)
        B_pim_signed = torch.mul(B_pim, B_signs_pim)

        A_final = torch.transpose(A_final, 0, 1)
        C_wait = torch.einsum('ij,bik->bijk', A_final, B_pim_signed)
        A_final = torch.transpose(A_final, 0, 1)

    zeros = torch.sum((B_pim == torch.zeros(B_pim.shape).to(device)).all(dim=2).type(torch.int), axis=1)
    highest_cols = q - torch.argmax(B_pim, dim=2)
    v = highest_cols.max()+1
    id = highest_cols + (v*torch.arange(highest_cols.shape[0]).to(device))[:,None]
    s_n = torch.bincount(id.ravel(),minlength=v*highest_cols.shape[0]).reshape(-1,v)
    s = torch.sum(s_n, axis=0)
    s_n = s_n.T
    s_n[0] = zeros
    s_n[-1] -= zeros
    s_n = s_n.T

    noise_value = 0
    if noise_value != 0:
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
                    temp = torch.zeros(B_pim_signed_edit[j][i].shape).to(device)
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
    return C_wait, B_pim_signed, A_final, 0, s, s_n, noise_value

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

    b_set = b_set.expand(max_2.shape)
    q_step=max_2/(torch.pow(2,b_set)-1 + 1e-12) + 1e-12
    digital_without_sign = torch.abs(torch.round(C_correct/(q_step + 1e-12))*q_step)

    sec = (torch.sum(C_correct, axis=2)!=0).type(torch.int)
    b_set = b_set.flatten()[:q]
    b_set = b_set.expand(sec.shape)
    energy = torch.sum(torch.mul(sec,torch.pow(2, b_set)))
    n_adcs = torch.sum((sec>0).type(torch.int))

    digital_outputs = torch.zeros(digital_without_sign.shape).to(device)
    output_signs = (C_correct<0).type(torch.int)
    digital_outputs = torch.mul(digital_without_sign, (-1)**output_signs).to(torch.float).to(device)

    return digital_outputs, n_adcs, energy

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

def max_normal(A,B_digital, M, K, N, q, permutation, v_ref,perc, num_sec,  s_n, opt):
    A = torch.ones(A.shape).to(device)*v_ref
    if permutation == 'random':
        A = torch.transpose(A, 0, 1)
        C_wait = torch.einsum('ij,bik->bijk', A, B_digital)
        A = torch.transpose(A, 0, 1)
    
    else:
        A = torch.transpose(A, 1, 2)
        C_wait = torch.einsum('bij,bik->bijk', A, B_digital)
        A = torch.transpose(A, 1, 2)
        
    C_tiles = []
    C_after_sum=[]

    if opt:
        for j in range(N):
            for i in range (num_sec):
                C_tiles.append(torch.sum(C_wait[j][int(torch.sum(s_n[j][:i])):int(torch.sum(s_n[j][:(i+1)]))],axis=0))

            C_tiles = torch.stack(C_tiles).to(device)
            C_after_sum.append(C_tiles)
            C_tiles=[]
        C_wait = torch.stack(C_after_sum).reshape(N,num_sec,M,q).to(device)
    else:    
        C_wait = C_wait.reshape(N, num_sec, int(K/num_sec), M, q)
        C_wait = torch.sum(C_wait, axis=2)

    return C_wait

def max_pos(A,B_digital, M, K, N, q, permutation, v_ref, perc, num_sec, s_n, opt):
    B_digital = torch.where(B_digital > 0, 1, 0)
    A = torch.ones(A.shape).to(device)*v_ref
    if permutation == 'random':
        A = torch.transpose(A, 0, 1)
        C_wait = torch.einsum('ij,bik->bijk', A, B_digital)
        A = torch.transpose(A, 0, 1)
    
    else:
        A = torch.transpose(A, 1, 2)
        C_wait = torch.einsum('bij,bik->bijk', A, B_digital)
        A = torch.transpose(A, 1, 2)
    
    C_tiles = []
    C_after_sum=[]
    
    if opt:
        for j in range(N):
            for i in range (num_sec):
                C_tiles.append(torch.sum(C_wait[j][int(torch.sum(s_n[j][:i])):int(torch.sum(s_n[j][:(i+1)]))],axis=0))

            C_tiles = torch.stack(C_tiles).to(device)
            C_after_sum.append(C_tiles)
            C_tiles=[]
        C_wait = torch.stack(C_after_sum).reshape(N,num_sec,M,q).to(device)
    else:
        C_wait = C_wait.reshape(N, num_sec, int(K/num_sec), M, q)
        C_wait = torch.sum(C_wait, axis=2)
        
    return C_wait

def max_neg(A,B_digital, M, K, N, q, permutation, v_ref,perc, num_sec, s_n, opt):
    B_digital = torch.where(B_digital > 0, -1, B_digital.to(int))
    A = torch.ones(A.shape).to(device)*v_ref
    if permutation == 'random':
        A = torch.transpose(A, 0, 1)
        C_wait = torch.einsum('ij,bik->bijk', A, B_digital)
        A = torch.transpose(A, 0, 1)
    
    else:
        A = torch.transpose(A, 1, 2)
        C_wait = torch.einsum('bij,bik->bijk', A, B_digital)
        A = torch.transpose(A, 1, 2)
    
    C_tiles = []
    C_after_sum=[]

    if opt:
        for j in range(N):
            for i in range (num_sec):
                C_tiles.append(torch.sum(C_wait[j][int(torch.sum(s_n[j][:i])):int(torch.sum(s_n[j][:(i+1)]))],axis=0))

            C_tiles = torch.stack(C_tiles).to(device)
            C_after_sum.append(C_tiles)
            C_tiles=[]
        C_wait = torch.stack(C_after_sum).reshape(N,num_sec,M,q).to(device)
    else:
        C_wait = C_wait.reshape(N, num_sec, int(K/num_sec), M, q)
        C_wait = torch.sum(C_wait, axis=2)

    return C_wait

def dac(x,d,v_ref):
    maximum = torch.max(x)
    return torch.round((x/(maximum+1e-12))*(2**d-1))*v_ref/(2**d-1), maximum, v_ref

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
    if prints:
        print('Starting CIMulator...\n')
    # dac settings
    v_ref=v_ref
    d=d

    # weight matrix settings
    q = q
    powers = torch.FloatTensor([2**(-i) for i in range(0,q)]).to(device)

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
    t1 = time.perf_counter()
    A_analog, maximum, v_ref = dac(A,d,v_ref)
    A_back = A_analog*(maximum/v_ref)
    err_dac = torch.sum(1/2*(A_back - A)**2)
    t2 = time.perf_counter()
    if prints:
        print('\nError due to DAC: ' + str(err_dac))
        print('Time due to DAC: ' + str(t2-t1))
    
    # converting B to digital
    t1 = time.perf_counter()
    B_digital = ef_convert_to_neg_bin(torch.abs(B), q)
    B_back = torch.sum(torch.mul(B_digital, powers), axis=-1)
    err_mm = torch.sum(1/2*(torch.abs(B) - B_back)**2)
    t2 = time.perf_counter()    
    if prints:
        print('Error due to matrix B conversion: ' + str(err_mm))
        print('Time due to matrix B conversion: ' + str(t2-t1))
    # operating
    # actual MM
    t1 = time.perf_counter()
    C_wait, B_new, A_new, means, s, s_n, noise = compute_tiles_nn(A_analog, B_digital, B_signs, permutation, add_noise, noise_gain)
    t2 = time.perf_counter()
    if prints:
        print('Time due to MM: ' + str(t2-t1))

    # scaling for each column power
    t1 = time.perf_counter()
    if opt==1:
        C_tiles = []
        C_after_sum=[]
        C_wait_opt = C_wait
        for j in range(N):
            for i in range (q+1):
                C_tiles.append(torch.sum(C_wait_opt[j][int(torch.sum(s_n[j][:i])):int(torch.sum(s_n[j][:(i+1)]))],axis=0))
            C_tiles = torch.stack(C_tiles).to(device)
            C_after_sum.append(C_tiles)
            C_tiles=[]
        C_wait_opt = torch.stack(C_after_sum).reshape(N,q+1,M,q).to(device)
        C_wait2 = torch.multiply(C_wait_opt, powers)
        num_sec = q+1
    else:
        C_wait = C_wait.reshape(N, num_sec, int(K/num_sec), M, q)
        C_wait = torch.sum(C_wait, axis=2)
        C_wait2 = torch.multiply(C_wait, powers)
    C_compare = torch.sum(C_wait2, axis=1)


    # converting output to digital
    if opt==1:
        digital_outputs, n_adcs, energy_value = adc(A_new, B_new, C_wait_opt, v_ref, b, permutation, perc, num_sec, b_set, s_n, opt)
    else:
        digital_outputs, n_adcs, energy_value = adc(A_new, B_new, C_wait, v_ref, b, permutation, perc, num_sec, b_set, s_n, opt)
    digital_outputs = torch.multiply(digital_outputs, powers)
    digital_outputs = torch.sum(digital_outputs, axis=1)
    err_adc = torch.sum(1/2*(digital_outputs - C_compare)**2)
    t2 = time.perf_counter()
    if prints:
        print('Error due to ADC: ' + str(err_adc))
        print('Energy due to ADC: ' + str(energy_value))
        print('Time due to ADC: ' + str(t2-t1))

    # horizontal add
    C_wait3 = torch.sum(digital_outputs, axis=-1)

    # uncomment if dont want to use adc
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
    return result, (error, err_adc), energy_value, active, s, noise, n_adcs