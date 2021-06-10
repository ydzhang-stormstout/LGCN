# The hyperboloid manifold Class
# 2019.12.20

import torch
from manifolds.base import Manifold
from utils.math_utils import arcosh, artanh, tanh
import numpy as np

class Lorentzian(Manifold):
    """
    Hyperboloid Manifold class.
    for x in (d+1)-dimension Euclidean space
    -x0^2 + x1^2 + x2^2 + … + xd = -c, x0 > 0, c > 0
    negative curvature - 1 / c
    """

    def __init__(self):
        super(Lorentzian, self).__init__()
        self.name = 'Lorentzian'
        self.max_norm = 1000
        self.min_norm = 1e-8
        self.eps = {torch.float32: 1e-6, torch.float64: 1e-8}

    def l_inner(self, x, y, keep_dim=False):
        # input shape [node, features]
        d = x.size(-1) - 1
        xy = x * y
        xy = torch.cat((-xy.narrow(1, 0, 1), xy.narrow(1, 1, d)), dim=1)
        return torch.sum(xy, dim=1, keepdim=keep_dim)

    def sqdist(self, p1, p2, c):
        dist = self.lorentzian_distance(p1, p2, c)
        dist = torch.clamp(dist, min = self.eps[p1.dtype], max=50)
        return dist

    def induced_distance(self, x, y, c):
        xy_inner = self.l_inner(x, y)
        sqrt_c = c ** 0.5
        return sqrt_c * arcosh(-xy_inner / c + self.eps[x.dtype])

    def lorentzian_distance(self, x, y, c):
        # the squared Lorentzian distance
        xy_inner = self.l_inner(x, y)
        return -2 * (c + xy_inner)

    def egrad2rgrad(self, p, dp, c):
        """
        Transform the Euclidean gradient to Riemannian gradient
        :param p: vector in hyperboloid
        :param dp: gradient with Euclidean geometry
        :return: gradient with Riemannian geometry
        """
        dp.narrow(-1, 0, 1).mul_(-1)  # multiply g_l^-1
        dp.addcmul_(self.l_inner(p, dp, keep_dim=True).expand_as(p)/c, p)
        return dp

    def normalize(self, p, c):
        """
        Normalize vector to confirm it is located on the hyperboloid
        :param p: [nodes, features(d + 1)]
        :param c: parameter of curvature
        """
        d = p.size(-1) - 1
        narrowed = p.narrow(-1, 1, d)
        if self.max_norm:
            narrowed = torch.renorm(narrowed.view(-1, d), 2, 0, self.max_norm)
        first = c + torch.sum(torch.pow(narrowed, 2), dim=-1, keepdim=True)
        first = torch.sqrt(first)
        return torch.cat((first, narrowed), dim=1)

    def proj(self, p, c):
        return self.normalize(p, c)

    def normalize_tangent(self, p, p_tan, c):
        """
        Normalize tangent vectors to place the vectors satisfies <p, p_tan>_L=0
        :param p: the tangent spaces at p. size:[nodes, feature]
        :param p_tan: the tangent vector in tangent space at p
        """
        d = p_tan.size(1) - 1
        p_tail = p.narrow(1, 1, d)
        p_tan_tail = p_tan.narrow(1, 1, d)
        ptpt = torch.sum(p_tail * p_tan_tail, dim=1, keepdim=True)
        p_head = torch.sqrt(c + torch.sum(torch.pow(p_tail, 2), dim=1, keepdim=True) + self.eps[p.dtype])
        return torch.cat((ptpt / p_head, p_tan_tail), dim=1)

    def normalize_tangent_zero(self, p_tan, c):
        zeros = torch.zeros_like(p_tan)
        zeros[:, 0] = c ** 0.5
        return self.normalize_tangent(zeros, p_tan, c)

    def exp_map_x(self, p, dp, c, is_res_normalize=True, is_dp_normalize=True):
        if is_dp_normalize:
            dp = self.normalize_tangent(p, dp, c)
        dp_lnorm = self.l_inner(dp, dp, keep_dim=True)
        dp_lnorm = torch.sqrt(torch.clamp(dp_lnorm + self.eps[p.dtype], 1e-6))
        dp_lnorm_cut = torch.clamp(dp_lnorm, max=50)
        sqrt_c = c ** 0.5
        res = (torch.cosh(dp_lnorm_cut / sqrt_c) * p) + sqrt_c * (torch.sinh(dp_lnorm_cut / sqrt_c) * dp / dp_lnorm)
        if is_res_normalize:
            res = self.normalize(res, c)
        return res

    def exp_map_zero(self, dp, c, is_res_normalize=True, is_dp_normalize=True):
        zeros = torch.zeros_like(dp)
        zeros[:, 0] = c ** 0.5
        return self.exp_map_x(zeros, dp, c, is_res_normalize, is_dp_normalize)

    def log_map_x(self, x, y, c, is_tan_normalize=True):
        """
        Logarithmic map at x: project hyperboloid vectors to a tangent space at x
        :param x: vector on hyperboloid
        :param y: vector to project a tangent space at x
        :param normalize: whether normalize the y_tangent
        :return: y_tangent
        """
        xy_distance = self.induced_distance(x, y, c)
        tmp_vector = y + self.l_inner(x, y, keep_dim=True) / c * x
        tmp_norm = torch.sqrt(self.l_inner(tmp_vector, tmp_vector) + self.eps[x.dtype])
        y_tan = xy_distance.unsqueeze(-1) / tmp_norm.unsqueeze(-1) * tmp_vector
        if is_tan_normalize:
            y_tan = self.normalize_tangent(x, y_tan, c)
        return y_tan

    def log_map_zero(self, y, c, is_tan_normalize=True):
        zeros = torch.zeros_like(y)
        zeros[:, 0] = c ** 0.5
        return self.log_map_x(zeros, y, c, is_tan_normalize)

    def logmap0(self, p, c):
        return self.log_map_zero(p, c)

    def proj_tan(self, u, p, c):
        """
        project vector u into the tangent vector at p
        :param u: the vector in Euclidean space
        :param p: the vector on a hyperboloid
        """
        return u - self.l_inner(u, p, keep_dim=True) / self.l_inner(p, p, keep_dim=True) * p

    def proj_tan_zero(self, u, c):
        zeros = torch.zeros_like(u)
        # print(zeros)
        zeros[:, 0] = c ** 0.5
        return self.proj_tan(u, zeros, c)

    def proj_tan0(self, u, c):
        return self.proj_tan_zero(u, c)

    def normalize_input(self, x, c):
        # print('=====normalize original input===========')
        num_nodes = x.size(0)
        zeros = torch.zeros(num_nodes, 1, dtype=x.dtype, device=x.device)
        x_tan = torch.cat((zeros, x), dim=1)
        return self.exp_map_zero(x_tan, c)

    def matvec_regular(self, m, x, b, c, use_bias):
        d = x.size(1) - 1
        x_tan = self.log_map_zero(x, c)
        x_head = x_tan.narrow(1, 0, 1)
        x_tail = x_tan.narrow(1, 1, d)
        mx = x_tail @ m.transpose(-1, -2)
        if use_bias:
            mx_b = mx + b
        else:
            mx_b = mx
        mx = torch.cat((x_head, mx_b), dim=1)
        mx = self.normalize_tangent_zero(mx, c)
        mx = self.exp_map_zero(mx, c)
        cond = (mx==0).prod(-1, keepdim=True, dtype=torch.uint8)
        res = torch.zeros(1, dtype=mx.dtype, device=mx.device)
        res = torch.where(cond, res, mx)
        return res

    def lorentz_centroid(self, weight, x, c):
        sum_x = torch.spmm(weight, x)
        # print('weight x', sum_x)
        x_inner = self.l_inner(sum_x, sum_x)
        coefficient = (c ** 0.5) / torch.sqrt(torch.abs(x_inner))
        return torch.mul(coefficient, sum_x.transpose(-2, -1)).transpose(-2, -1)

    def lorentz2poincare(self, x, c):
        try:
            radius = torch.sqrt(c)
        except:
            radius = c ** 0.5
        d = x.size(-1) -1
        return (x.narrow(-1, 1, d) * radius) / (x.narrow(-1, 0, 1) + radius)

    def poincare2lorentz(self, x, c):
        x_norm_square = torch.sum(x * x, dim=1, keepdim=True)
        return torch.cat((1 + x_norm_square, 2 * x), dim=1) / (1 - x_norm_square + 1e-8)

    def ptransp0(self, y, v, c):
        # y: target point
        zeros = torch.zeros_like(v)
        zeros[:, 0] = c ** 0.5
        v = self.normalize_tangent_zero(v, c)
        return self.ptransp(zeros, y, v, c)
    
    def ptransp(self, x, y, v, c):
        # transport v from x to y
        K = 1. / c
        yv = self.l_inner(y, v, keep_dim=True)
        xy = self.l_inner(x, y, keep_dim=True)
        _frac = K * yv / (1 - K * xy)
        return v + _frac * (x + y)