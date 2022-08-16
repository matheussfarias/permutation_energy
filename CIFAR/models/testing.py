import torch

inp = torch.randn(1, 3, 10, 12)
w = torch.randn(2, 3, 4, 5)
inp_unf = torch.nn.functional.unfold(inp, (4, 5)).transpose(1, 2)
B = w.view(w.size(0), -1).t()
out_unf = inp_unf.matmul(B)
out_unf = out_unf.transpose(1, 2)
out = out_unf.view(1, 2, 7, 8)
print (out.shape)