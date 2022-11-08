C_tiles = []
    C_after_sum=[]
    j = 2
    i = 2
    print(s_n)
    print(s_n.shape)
    print(C_wait_opt2)
    print(C_wait_opt2.shape)

    print(C_wait_opt2[j][int(torch.sum(s_n[j][:i])):int(torch.sum(s_n[j][:(i+1)]))])
    print(s_n)
    transpose = s_n.T
    #print(torch.transpose(C_wait_opt2,0,1)[int(torch.sum(s_n[:i]))])
    for i in range(q+1):
        print(int(torch.sum(s_n[0][:i])),int(torch.sum(s_n[0][:(i+1)])))
        print(int(torch.sum(s_n[1][:i])),int(torch.sum(s_n[1][:(i+1)])))
        print(int(torch.sum(s_n[2][:i])),int(torch.sum(s_n[2][:(i+1)])))
        print('oi')
    
    for i in range(q+1):
        ini = torch.sum(s_n.T[:(i)], axis=0)
        end = torch.sum(s_n.T[:(i+1)], axis=0)
        print(ini)
        print(end)
        id = torch.range(ini,end)
        id = torch.FloatTensor([2**(-i) for i in range(0,q)]).to(device)
        print(id)
        exit()
        print('oi')
    exit()
    print(s_n.T[:(i+1)])
    print(torch.sum(s_n.T[:(i+1)], axis=0))
    print(torch.sum(s_n.T[:(i)], axis=0))
    print(C_wait_opt2.shape)
    #for i in range (q+1):
    #    C_tiles.append(C_wait_opt2[torch.sum(s_n.T[:(i)], axis=0), torch.sum(s_n.T[:(i+1)], axis=0)])
    #C_tiles = torch.stack(C_tiles).to(device)
    #print(C_tiles)
    #print(C_tiles.shape)
    #print(C_wait_opt.shape)
    exit()

    print(C_wait_opt.shape)
    transpose_C = torch.transpose(C_wait_opt, 0, 1)
    print(transpose_C[int(torch.sum(transpose[:i])):int(torch.sum(transpose[:(i+1)]))])
    exit()
    print(C_wait_opt[j][int(torch.sum(s_n[j][:i])):int(torch.sum(s_n[j][:(i+1)]))])

    exit()