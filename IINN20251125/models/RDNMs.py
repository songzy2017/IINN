import numpy as np
import torch
import torch.nn as nn


class Dendrite(nn.Module):
    def __init__(self, num_synapse, num_dendrite, flexible_synapse):
        super(Dendrite, self).__init__()
        self.num_synapse = num_synapse
        self.num_dendrite = num_dendrite
        if flexible_synapse is not False:
            self.k = nn.Parameter(torch.ones(self.num_dendrite, self.num_synapse) * 10.0, requires_grad=True)
        else:
            self.k = nn.Parameter(torch.ones(self.num_dendrite, self.num_synapse) * 10.0, requires_grad=False)
        parame_buffer = torch.rand(self.num_dendrite, self.num_synapse) - 0.5
        self.w = nn.Parameter(parame_buffer * 2.0, requires_grad=True)
        self.q = nn.Parameter(parame_buffer * 1.0, requires_grad=True)
        self.mask = self.mask_gen()
        self.module_list = nn.ModuleList()
        self.module_list.append(nn.Linear(2 ** self.num_synapse - 1, 1))
        # print('Dendrite is ready.')

    def mask_gen(self):
        buffer = [bin(x)[2:].zfill(self.num_synapse) for x in list(range(1, 2 ** self.num_synapse))]
        mask = nn.Parameter(torch.tensor(np.array([list(x) for x in buffer], dtype=np.float32)), requires_grad=False)
        return mask

    def forward(self, x):
        synape_out = torch.sigmoid(self.k * (self.w * x - self.q))
        out = (synape_out[0, 0] ** self.mask).unsqueeze(0).unsqueeze(0)
        for i in range(1, self.num_dendrite):
            out = torch.cat([out, (synape_out[0, i] ** self.mask).unsqueeze(0).unsqueeze(0)], dim=1)
        for j in range(1, len(x)):
            out_buffer = (synape_out[j, 0] ** self.mask).unsqueeze(0).unsqueeze(0)
            for i in range(1, self.num_dendrite):
                out_buffer = torch.cat([out_buffer, (synape_out[j, i] ** self.mask).unsqueeze(0).unsqueeze(0)], dim=1)
            out = torch.cat([out, out_buffer], dim=0)
        out = torch.prod(out, dim=-1)
        out = self.module_list[-1](out).squeeze(-1).unsqueeze(-2)
        return out


class RDNM(nn.Module):
    def __init__(self, num_synapse, num_dendrite, num_layers, num_soma, flexible_synapse, flexible_soma):
        super(RDNM, self).__init__()
        self.num_layers = num_layers
        if flexible_soma is not False:
            self.ks = nn.Parameter(torch.ones(num_soma) * 10.0, requires_grad=True)
        else:
            self.ks = nn.Parameter(torch.ones(num_soma) * 10.0, requires_grad=False)
        self.module_list = nn.ModuleList()
        for i in range(self.num_layers):
            self.module_list.append(Dendrite(num_synapse, num_dendrite, flexible_synapse))
            self.module_list.append(nn.Linear(num_dendrite, num_soma))
        self.module_list.append(nn.Linear(self.num_layers, num_soma))
        print('***** DNM is ready. *****')

    def forward(self, x):
        out_buffer_all = self.module_list[0](x)
        out_all = self.module_list[1](out_buffer_all)
        for i in range(1, self.num_layers):
            out_buffer = self.module_list[2 * i](x)
            out = self.module_list[2 * i + 1](out_buffer)
            out_buffer_all = torch.cat([out_buffer_all, out_buffer], dim=-2)
            out_all = torch.cat([out_all, out], dim=-2)
        out_all = self.module_list[-1](out_all.squeeze(-1))
        out_all = torch.sigmoid(self.ks * out_all.squeeze(-1))
        return out_all
