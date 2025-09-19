import torch
import torch.nn as nn
from collections import defaultdict
import torch.nn.functional as F
from torch.autograd import Variable
import time
import numpy as np



INIT_RANGE = 0.5


class Binarize(torch.autograd.Function):
    """Deterministic binarization."""
    @staticmethod
    def forward(ctx, X):
        y = torch.where(X > 0, torch.ones_like(X), torch.zeros_like(X)-1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class BinarizeLayer(nn.Module):
    """Implement the feature discretization and binarization."""

    def __init__(self, n, input_dim, left=None, right=None):
        super(BinarizeLayer, self).__init__()
        self.n = n
        self.input_dim = input_dim
        self.disc_num = input_dim[0]
        self.output_dim = self.disc_num + self.n * self.input_dim[1] * 2
        self.layer_type = 'binarization'
        self.dim2id = {i: i for i in range(self.output_dim)}

        self.register_buffer('left', left)
        self.register_buffer('right', right)

        if self.input_dim[1] > 0:
            if self.left is not None and self.right is not None:
                cl = self.left + torch.rand(self.n, self.input_dim[1]) * (self.right - self.left)
                cr = self.left + torch.rand(self.n, self.input_dim[1]) * (self.right - self.left)

            else:
                cl = 3. * (2. * torch.rand(self.n, self.input_dim[1]) - 1.)
                cr = 3. * (2. * torch.rand(self.n, self.input_dim[1]) - 1.)
                print("cr:", cr.shape)
            self.register_buffer('cl', cl)
            self.register_buffer('cr', cr)
    def forward(self, x):
        if self.input_dim[1] > 0:
            x_disc, x = x[:, 0: self.input_dim[0]], x[:, self.input_dim[0]:]
            x = x.unsqueeze(-1)

            return torch.cat((Binarize.apply(x_disc), Binarize.apply(x - self.cl.t()).reshape(x.shape[0], -1),
                                 Binarize.apply(self.cr.t() - x).reshape(x.shape[0], -1)), dim=1)

        return Binarize.apply(x)


    def clip(self):
        if self.input_dim[1] > 0 and self.left is not None and self.right is not None:
            self.cl.data = torch.where(self.cl.data > self.right, self.right, self.cl.data)
            self.cl.data = torch.where(self.cl.data < self.left, self.left, self.cl.data)

            self.cr.data = torch.where(self.cr.data > self.right, self.right, self.cr.data)
            self.cr.data = torch.where(self.cr.data < self.left, self.left, self.cr.data)

    def get_bound_name(self, feature_name, mean=None, std=None):
        bound_name = []
        for i in range(self.input_dim[0]):
            bound_name.append(feature_name[i])
        if self.input_dim[1] > 0:
            for c, op in [(self.cl, '>'), (self.cr, '<')]:
                c = c.detach().cpu().numpy()
                for i, ci in enumerate(c.T):
                    fi_name = feature_name[self.input_dim[0] + i]
                    for j in ci:
                        if mean is not None and std is not None:
                            j = j * std[fi_name] + mean[fi_name]
                        bound_name.append('{} {} {:.3f}'.format(fi_name, op, j))
        return bound_name




class LRLayer(nn.Module):

    def __init__(self, n, input_dim):
        super(LRLayer, self).__init__()
        self.n = n
        self.input_dim = input_dim
        self.output_dim = self.n
        self.layer_type = 'linear'
        self.fc1 = nn.Linear(self.input_dim, self.output_dim)

    def forward(self, x):
        output = self.fc1(x)

        return output


    def clip(self):
        for param in self.fc1.parameters():
            param.data.clamp_(-1.0, 1.0)




class Selection_Layer(nn.Module):

    def __init__(self, n, input_dim):
        super(Selection_Layer, self).__init__()
        self.n = n
        self.input_dim = input_dim
        self.output_dim = self.n
        self.layer_type = 'selection_negation'
        self.W_op = nn.Parameter(INIT_RANGE * torch.randn(self.n, 1))
        self.W_conn = nn.Parameter(INIT_RANGE * torch.rand(self.input_dim, self.n))
        self.W_negation = nn.Parameter(torch.rand(self.input_dim, self.n) - 0.5)
        self.node_activation_cnt = None
        self.forward_tot = None

    def forward(self, x, prev_w_op=None):

        negation = Binarize.apply(self.W_negation)

        W_conn = Binarize.apply(self.W_conn)

        W_op_ori = (Binarize.apply(self.W_op)+1)/2
        W_op = torch.cat([1-W_op_ori, W_op_ori], dim=-1)

        x_ = x.unsqueeze(-1) * negation.unsqueeze(0)
        input_ = (((x_+1)/2) * ((W_conn+1)/2).unsqueeze(0)) * 2 - 1 # -1 , 1

        del x_

        Disj_output = torch.amax(input_, 1).squeeze().unsqueeze(-1)
        Conj_output =  torch.amin(input_, 1).squeeze().unsqueeze(-1)

        output = (torch.cat((Disj_output, Conj_output), dim=-1) * W_op.unsqueeze(0)).sum(dim=2).squeeze()
        # output = torch.where(output == -2, torch.zeros_like(output), output)
        return output, W_op

    def clip(self):
        self.W_op.data.clamp_(-1.0, 1.0)
        self.W_conn.data.clamp_(-1.0, 1.0)
        # pass



class Selection_Layer_mask(nn.Module):
    def __init__(self, n, input_dim):
        super(Selection_Layer_mask, self).__init__()
        self.n = n
        self.input_dim = input_dim
        self.output_dim = self.n
        self.layer_type = 'selection_mask'
        self.W_op = nn.Parameter(INIT_RANGE * torch.randn(self.n, 1))
        self.W_conn = nn.Parameter(torch.ones(self.input_dim, self.n))
        self.node_activation_cnt = None
        self.forward_tot = None

    def forward(self, x, prev_w_op=None):

        W_conn =(Binarize.apply(self.W_conn) + 1) / 2
        W_op_ori = ((Binarize.apply(self.W_op)+1)/2)
        W_op = torch.cat([1-W_op_ori, W_op_ori], dim=-1)
        if prev_w_op is not None:
            Conn_mask = torch.add(W_op[:,0].unsqueeze(1), prev_w_op[:,0]) - 2 * W_op[:,0].unsqueeze(1) * prev_w_op[:,0]
        self.W_Conn_rule_print = W_conn * Conn_mask.t()
        W_conn = W_conn * Conn_mask.t()
        input_ = (((x+1)/2).unsqueeze(-1) * W_conn)*2 -1
        Disj_output= torch.amax(input_, 1).unsqueeze(-1)
        Conj_output=  torch.amin(input_, 1).unsqueeze(-1)
        output = (torch.cat((Disj_output, Conj_output), dim=-1) * W_op.unsqueeze(0)).sum(dim=2).squeeze()
        return output

    def clip(self):
        self.W_op.data.clamp_(-1.0, 1.0)
        self.W_conn.data.clamp_(-1.0, 1.0)
        # pass


