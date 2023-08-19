import torch

class GTVBlock(torch.nn.Module):
    def __init__(self, hidden_channels) -> None:
        super().__init__()

        self.z = None
        self.lin = torch.nn.Linear(hidden_channels, hidden_channels)

    def forward(self, x, sizes_subg=None):
        # x = inputs[0]
        # if self.z is None:
        #     self.z = torch.zeros(x.shape[0]-1, (x.shape[-1])).to(x.device)
        # if self.z.shape[0] != x.shape[0]-1:
        #     self.z = torch.zeros(x.shape[0]-1, (x.shape[-1])).to(x.device)
        self.z = self.lin(self.z)
        x, self.z = GTV(x, self.z)
        return x
        # return (x, inputs[1], inputs[2], inputs[3])


def GTV (y, z):
    lamda = 0.3
    alpha = 8   #.5
    T = lamda/2
    # z = z.to(y.device) 
    ##for feature diff
    # diff = z[:, 1:] - z[:, 0:-1]
    # append = torch.cat((z[:,0:1], diff, z[:,-2:-1]), 1)
    # x = y - append
    # xdiff = x[:, 1:]-x[:, 0:-1]
    # zt = z + 1 / alpha * xdiff
    # TT = torch.zeros(z.shape[1]) + T
    # zt = torch.maximum(torch.minimum(zt[:,0:], TT), -TT)
    ##for node diff
    diff = z[1:,:] - z[0:-1,:]
    append = torch.cat((z[0:1,:],-diff,z[-2:-1,:]),0)
    # print(y.shape, append.shape)
    x = y - append
    xdiff = x[1:,:]-x[0:-1,:]
    zt = z + 1/alpha*xdiff
    TT = torch.zeros(z.shape[1],device=zt.device) + T
    zt = torch.maximum(torch.minimum(zt[0:, :], TT), -TT)
    return x, zt