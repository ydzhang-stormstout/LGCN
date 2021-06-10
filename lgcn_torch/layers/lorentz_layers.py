# Lorentzian neural network layers
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.modules.module import Module
import numpy
from manifolds.lorentzian import Lorentzian
from utils.math_utils import arsinh

def get_dim_act_curv(args):
    """
    get dimension and activation in each layers
    :param args:
    :return:
    """
    if not args.act:
        act = lambda x: x
    else:
        act = getattr(F, args.act)
    acts = [act] * (args.num_layers - 1)
    dims = [args.feat_dim] + ([args.dim] * (args.num_layers - 1))
    if args.task in ['lp']:
        dims += [args.dim]
        acts += [act]
        n_curvatures = args.num_layers
    else:
        n_curvatures = args.num_layers - 1
    if args.c is None:
        curvatures = [nn.Parameter(torch.Tensor([1.])) for _ in range(n_curvatures)]
    else:
        curvatures = [torch.tensor([args.c]) for _ in range(n_curvatures)]
        if not args.cuda == -1:
            curvatures = [curv.to(args.device) for curv in curvatures]
    return dims, acts, curvatures

class LorentzLinear(nn.Module):
    # Lorentz Hyperbolic Graph Neural Layer
    def __init__(self, manifold, in_features, out_features, c, drop_out, use_bias):
        super(LorentzLinear, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.drop_out = drop_out
        self.use_bias = use_bias
        self.bias = nn.Parameter(torch.Tensor(out_features-1))   # -1 when use mine mat-vec multiply
        self.weight = nn.Parameter(torch.Tensor(out_features - 1, in_features))  # -1, 0 when use mine mat-vec multiply
        self.reset_parameters()

    def report_weight(self):
        print(self.weight)

    def reset_parameters(self):
        init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        init.constant_(self.bias, 0)
        # print('reset lorentz linear layer')

    def forward(self, x):
        drop_weight = F.dropout(self.weight, self.drop_out, training=self.training)
        mv = self.manifold.matvec_regular(drop_weight, x, self.bias, self.c, self.use_bias)
        return mv

    def extra_repr(self):
        return 'in_features={}, out_features={}, c={}'.format(
            self.in_features, self.out_features, self.c
        )


class LorentzAgg(Module):
    """
    Lorentz centroids aggregation layer
    """
    def __init__(self, manifold, c, use_att, in_features, dropout):
        super(LorentzAgg, self).__init__()
        self.manifold = manifold
        self.c = c
        self.use_att = use_att
        self.in_features = in_features
        self.dropout = dropout
        self.this_spmm = SpecialSpmm()
        if use_att:
            self.att = LorentzSparseSqDisAtt(manifold, c, in_features-1, dropout)


    def lorentz_centroid(self, weight, x, c):
        """
        Lorentz centroid
        :param weight: dense weight matrix. shape: [num_nodes, num_nodes]
        :param x: feature matrix [num_nodes, features]
        :return: the centroids of nodes [num_nodes, features]
        """
        if self.use_att:
            sum_x = self.this_spmm(weight[0], weight[1], weight[2], x)
        else:
            sum_x = torch.spmm(weight, x)
        x_inner = self.manifold.l_inner(sum_x, sum_x)
        coefficient = (c ** 0.5) / torch.sqrt(torch.abs(x_inner))
        return torch.mul(coefficient, sum_x.transpose(-2, -1)).transpose(-2, -1)

    def forward(self, x, adj):
        if self.use_att:
            adj = self.att(x, adj)
        output = self.lorentz_centroid(adj, x, self.c)
        return output

    def extra_repr(self):
        return 'c={}, use_att={}'.format(
                self.c, self.use_att
        )

    def reset_parameters(self):
        if self.use_att:
            self.att.reset_parameters()
        # print('reset agg finished')


class LorentzAct(Module):
    """
    Lorentz activation layer
    """
    def __init__(self, manifold, c_in, c_out, act):
        super(LorentzAct, self).__init__()
        self.manifold = manifold
        self.c_in = c_in
        self.c_out = c_out
        self.act = act

    def forward(self, x):
        xt = self.act(self.manifold.log_map_zero(x, c=self.c_in))
        xt = self.manifold.normalize_tangent_zero(xt, self.c_in)
        return self.manifold.exp_map_zero(xt, c=self.c_out)

    def extra_repr(self):
        return 'c_in={}, c_out={}'.format(
                self.c_in, self.c_out
        )


class LorentzGraphNeuralNetwork(nn.Module):
    def __init__(self, manifold, in_feature, out_features, c_in, c_out, drop_out, act, use_bias, use_att):
        super(LorentzGraphNeuralNetwork, self).__init__()
        self.c_in = c_in
        self.linear = LorentzLinear(manifold, in_feature, out_features, c_in, drop_out, use_bias)
        self.agg = LorentzAgg(manifold, c_in, use_att, out_features, drop_out)
        self.lorentz_act = LorentzAct(manifold, c_in, c_out, act)

    def forward(self, input):
        x, adj = input
        h = self.linear.forward(x) ## problem is h1+
        h = self.agg.forward(h, adj)
        h = self.lorentz_act.forward(h)
        output = h, adj
        return output

    def reset_parameters(self):
        self.linear.reset_parameters()
        self.agg.reset_parameters()

class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        device = b.device
        a = torch.sparse_coo_tensor(indices, values, shape, device=device)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)

class LorentzSparseSqDisAtt(nn.Module):
    def __init__(self, manifold, c, in_features, dropout):
        super(LorentzSparseSqDisAtt, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.manifold = manifold
        self.c = c
        self.weight_linear = LorentzLinear(manifold, in_features, in_features+1, c, dropout, True)

    def forward(self, x, adj):
        d = x.size(1) - 1
        x = self.weight_linear(x)
        index = adj._indices()
        _x = x[index[0, :]]
        _y = x[index[1, :]]
        _x_head = _x.narrow(1, 0, 1)
        _y_head = _y.narrow(1, 0, 1)
        _x_tail = _x.narrow(1, 1, d)
        _y_tail = _y.narrow(1, 1, d)
        l_inner = -_x_head.mul(_y_head).sum(-1) + _x_tail.mul(_y_tail).sum(-1)
        res = torch.clamp(-(self.c+l_inner), min=1e-10, max=1)
        res = torch.exp(-res)
        return (index, res, adj.size())

class LorentzGraphDecoder(nn.Module):
    # Lorentzian graph neural network decoder
    def __init__(self, manifold, in_feature, out_features, c_in, c_out, drop_out, act, use_bias, use_att):
        super(LorentzGraphDecoder, self).__init__()
        self.manifold = manifold
        self.c_in = c_in
        self.out_features = out_features + 1 # original output equal to num_classes
        self.in_features = in_feature
        self.linear = LorentzLinear(manifold, in_feature-1, self.out_features, c_in, drop_out, False)
        self.agg = LorentzAgg(manifold, c_in, use_att, self.out_features, drop_out)
        self.lorentz_act = LorentzAct(manifold, c_in, c_out, act)
        self.bias = nn.Parameter(torch.Tensor(self.out_features))
        init.constant_(self.bias, 0)

    def forward(self, input):
        x, adj = input
        # print('=====x', x.shape, self.in_features)
        h = self.linear.forward(x) ## problem is h1+
        h = self.agg.forward(h, adj)
        h = self.lorentz_act.forward(h)
        b = self.manifold.ptransp0(h, self.bias, self.c_in)
        b = self.manifold.exp_map_x(h, b, self.c_in)
        poincare_h = self.manifold.lorentz2poincare(h, self.c_in)
        output = poincare_h, adj
        return output

    def reset_parameters(self):
        self.linear.reset_parameters()
        self.agg.reset_parameters()
