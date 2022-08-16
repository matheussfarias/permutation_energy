from tkinter import E
import numpy as np
import torch
import time
from prettytable import PrettyTable

global device

cuda_act = False

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

def compute_tiles_nn(A, B, B_signs, t):
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
                B_new.append(torch.FloatTensor(B.cpu()[indeces[i+k*K][0], indeces[i+k*K][1]]).to(device))
                B_signs_new.append(torch.IntTensor(B_signs.cpu()[indeces[i+k*K][0], indeces[i+k*K][1]]).to(device))
        
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
        mean_temp = torch.mean(torch.mul(B_pim_signed,powers), axis=2)
        
    return tiles, B_pim_signed, A_final, mean_temp

def adc(A, B_digital, C_correct, v_ref, b, permutation, perc):
    N = B_digital.shape[0]
    K = B_digital.shape[1]
    q = B_digital.shape[2]


    if permutation=='random':
        M = A.shape[0]

        max_outputs_1 = max_normal(A, B_digital, M, K, N, q, permutation, v_ref, perc)
        max_outputs_2 = max_pos(A, B_digital, M, K, N, q, permutation, v_ref, perc)
        max_outputs_3 = max_neg(A, B_digital, M, K, N, q, permutation, v_ref, perc)

        max_outputs_1 = torch.abs(max_outputs_1)
        max_outputs_2 = torch.abs(max_outputs_2)
        max_outputs_3 = torch.abs(max_outputs_3)

        max_1 = torch.max(max_outputs_1, max_outputs_2)
        max_2 = torch.max(max_1, max_outputs_3)
    else:
        M=A.shape[1]

        max_outputs_1 = max_normal(A, B_digital, M, K, N, q, permutation, v_ref, perc)
        max_outputs_2 = max_pos(A, B_digital, M, K, N, q, permutation, v_ref, perc)
        max_outputs_3 = max_neg(A, B_digital, M, K, N, q, permutation, v_ref, perc)

        max_outputs_1 = torch.abs(max_outputs_1)
        max_outputs_2 = torch.abs(max_outputs_2)
        max_outputs_3 = torch.abs(max_outputs_3)

        max_1 = torch.max(max_outputs_1, max_outputs_2)
        max_2 = torch.max(max_1, max_outputs_3)
    energy=0

    
    q_step_0 = torch.abs(max_2)/(2**(b)-1) + 1e-12
    q_step_1 = torch.abs(max_2)/(2**(b-1)-1) + 1e-12
    q_step_2 = torch.abs(max_2)/(2**(b-2)-1) + 1e-12
    
    digital_without_sign = []
    for i in range(C_correct.shape[0]):
        digital_without_sign.append(torch.abs(torch.round(C_correct[i][0]/q_step_0[i][0])*q_step_0[i][0]))
        digital_without_sign.append(torch.abs(torch.round(C_correct[i][1]/q_step_1[i][1])*q_step_1[i][1]))
        digital_without_sign.append(torch.abs(torch.round(C_correct[i][2]/q_step_2[i][2])*q_step_2[i][2]))

    digital_without_sign = torch.stack(digital_without_sign).reshape(C_correct.shape).to(device)

    energy = C_correct.shape[0]*C_correct.shape[1]*C_correct.shape[2]*2**(b)/3 + C_correct.shape[0]*C_correct.shape[1]*C_correct.shape[2]*2**(b-1)/3 + C_correct.shape[0]*C_correct.shape[1]*C_correct.shape[2]*2**(b-2)/3
    adcs=b
    
    digital_outputs = torch.zeros(digital_without_sign.shape).to(device)
    output_signs = (C_correct<0).type(torch.int)
    for i in range(digital_outputs.shape[0]):
        for j in range(digital_outputs.shape[1]):
            for k in range(digital_outputs.shape[2]):
                digital_outputs[i][j][k] = digital_without_sign[i][j][k]*(-1)**output_signs[i][j][k]

    return digital_outputs, adcs, energy

def filling_array(x, q):
    B_digital=[]
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            B_digital.append(convert_to_neg_bin(x[i][j],q))
    return torch.Tensor(B_digital).reshape(x.shape[0],x.shape[1],q).to(device)

def max_normal(A,B_digital, M, K, N, q, permutation, v_ref,perc):
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
    
    for j in range(N):
        C_tiles.append(torch.sum(C_wait[j][0:1+int(np.floor((perc[0]/100)*K))],axis=0))
        C_tiles.append(torch.sum(C_wait[j][int(np.ceil((perc[0]/100)*K)):1+int(np.floor(((perc[0]+perc[1])/100)*K))],axis=0))
        C_tiles.append(torch.sum(C_wait[j][int(np.ceil(((perc[0]+perc[1])/100)*K)):],axis=0))

        C_tiles = torch.stack(C_tiles).to(device)
        C_after_sum.append(C_tiles)
        C_tiles=[]

    C_wait = torch.stack(C_after_sum).reshape(N,3,M,q).to(device)

    return C_wait

def max_pos(A,B_digital, M, K, N, q, permutation, v_ref, perc):
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
    for j in range(N):
        C_tiles.append(torch.sum(C_wait[j][0:1+int(np.floor((perc[0]/100)*K))],axis=0))
        C_tiles.append(torch.sum(C_wait[j][int(np.ceil((perc[0]/100)*K)):1+int(np.floor(((perc[0]+perc[1])/100)*K))],axis=0))
        C_tiles.append(torch.sum(C_wait[j][int(np.ceil(((perc[0]+perc[1])/100)*K)):],axis=0))

        C_tiles = torch.stack(C_tiles).to(device)
        C_after_sum.append(C_tiles)
        C_tiles=[]

    C_wait = torch.stack(C_after_sum).reshape(N,3,M,q).to(device)
    
    return C_wait

def max_neg(A,B_digital, M, K, N, q, permutation, v_ref,perc):
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
    for j in range(N):
        C_tiles.append(torch.sum(C_wait[j][0:1+int(np.floor((perc[0]/100)*K))],axis=0))
        C_tiles.append(torch.sum(C_wait[j][int(np.ceil((perc[0]/100)*K)):1+int(np.floor(((perc[0]+perc[1])/100)*K))],axis=0))
        C_tiles.append(torch.sum(C_wait[j][int(np.ceil(((perc[0]+perc[1])/100)*K)):],axis=0))

        C_tiles = torch.stack(C_tiles).to(device)
        C_after_sum.append(C_tiles)
        C_tiles=[]

    C_wait = torch.stack(C_after_sum).reshape(N,3,M,q).to(device)
    
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
    return torch.round((x/maximum)*(2**d-1))*v_ref/(2**d-1), maximum, v_ref

def sorting_order(A):
    return A[0] #sorting by lower to higher

def sorting_area(A):
    return A[-1] #sorting by number of 1s


def cim(A, B, v_ref, d, q, b, permutation, prints, perc):
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
    B_signs = (B<0).type(torch.int)

    if prints:
        print('B: ' + str(B))
        print('B_signs: ' + str(B_signs))

    # correct result
    C = torch.matmul(A,B)

    # conversions
    # converting A to analog
    t1=time.time()
    A_analog, maximum, v_ref = dac(A,d,v_ref)
    A_back = A_analog*maximum/v_ref
    err_dac = torch.sum(1/2*(A_back - A)**2)
    t2=time.time()
    if prints:
        print('\nError due to DAC: ' + str(err_dac))
        print('Time due to DAC: ' + str(t2-t1))

    # converting B to digital
    t1=time.time()
    B_digital = filling_array(torch.abs(B), q)
    B_back = torch.sum(torch.mul(B_digital, powers), axis=-1)
    err_mm = torch.sum(1/2*(torch.abs(B) - B_back)**2)    

    t2=time.time()
    if prints:
        print('Error due to matrix B conversion: ' + str(err_mm))
        print('Time due to matrix B conversion: ' + str(t2-t1))

    # operating
    # actual MM
    t1=time.time()

    C_wait, B_new, A_new, means = compute_tiles_nn(A_analog, B_digital, B_signs, permutation)
    '''
    t1=time.time()
    C_wait, B_signs, B, A_new = compute_tiles_nn_past(A_analog, B_digital, B_signs, permutation)
    t2=time.time()
    print('Time before optimizing MM: ', t2-t1)
    '''
    C_tiles = []

    C_after_sum=[]
    #perc = [68.2, 27.2]
    #print([0,int(np.floor((perc[0]/100)*K))])
    #print([int(np.ceil((perc[0]/100)*K)),int(np.floor(((perc[0]+perc[1])/100)*K))])
    #print([int(np.ceil(((perc[0]+perc[1])/100)*K)),K])

    '''
    C_after=[]
    for j in range(N):
        C_tiles.append(torch.sum(C_wait[j][0:1+int(np.floor((perc[0]/100)*K))],axis=0))
        C_tiles.append(torch.sum(C_wait[j][int(np.ceil((perc[0]/100)*K)):1+int(np.floor(((perc[0]+perc[1])/100)*K))],axis=0))
        C_tiles.append(torch.sum(C_wait[j][int(np.ceil(((perc[0]+perc[1])/100)*K)):],axis=0))
    
        C_tiles = torch.stack(C_tiles)
        C_tiles = torch.sum(C_tiles,axis=0)
        C_after.append(C_tiles)
        C_tiles=[]
    print(torch.stack(C_after))
    '''

    for j in range(N):
        C_tiles.append(torch.sum(C_wait[j][0:1+int(np.floor((perc[0]/100)*K))],axis=0))
        C_tiles.append(torch.sum(C_wait[j][int(np.ceil((perc[0]/100)*K)):1+int(np.floor(((perc[0]+perc[1])/100)*K))],axis=0))
        C_tiles.append(torch.sum(C_wait[j][int(np.ceil(((perc[0]+perc[1])/100)*K)):],axis=0))

        C_tiles = torch.stack(C_tiles).to(device)
        C_after_sum.append(C_tiles)
        C_tiles=[]

    C_wait = torch.stack(C_after_sum).reshape(N,3,M,q).to(device)
    #print(torch.sum(C_wait,axis=1))
    t2=time.time()
    if prints:
        print('Time due to MM: ' + str(t2-t1))

    # scaling for each column power
    C_wait2 = torch.multiply(C_wait, powers)
    C_compare = torch.sum(C_wait2, axis=1)

    # converting output to digital
    t1=time.time()
    digital_outputs, adcs, energy_value = adc(A_new, B_new, C_wait2, v_ref, b, permutation, perc)
    
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
    C_wait3 = torch.sum(digital_outputs, axis=-1)

    # uncomment if dont want to use adc
    #C_wait3 = torch.sum(C_wait2, axis=-1)
    
    # converting output back to digital
    result = (C_wait3*maximum/v_ref).t()

    # calculating error
    error = torch.sum(1/2*(result - C)**2)

    if prints:
        print('\nResults...')
        print('Expected result: ' + str(C))
        print('Obtained result: ' + str(result))
        print('Total error: ' + str(error))

    return result, (error, err_adc), energy_value