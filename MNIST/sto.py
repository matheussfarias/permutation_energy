import torch

x = torch.FloatTensor([2.2,5.4])
def sto(x):
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

print(sto(x))
