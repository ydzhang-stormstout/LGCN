"""Graph encoders."""
import manifolds
import layers.lorentz_layers as loren_layers
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.math_utils as pmath
from layers.layers import Linear, get_dim_act


class Encoder(nn.Module):
    """
    Encoder abstract class.
    """
    def __init__(self, c):
        super(Encoder, self).__init__()
        self.c = c

    def encode(self, x, adj):
        if self.encode_graph:
            input = (x, adj)
            output, _ = self.layers.forward(input)
        else:
            output = self.layers.forward(x)
        return output

class MLP(Encoder):
    """
    Multi-layer perceptron.
    """
    def __init__(self, c, args):
        super(MLP, self).__init__(c)
        assert args.num_layers > 0
        dims, acts = get_dim_act(args)
        layers = []
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            layers.append(Linear(in_dim, out_dim, args.dropout, act, args.bias))
        self.layers = nn.Sequential(*layers)
        self.encode_graph = False

    def reset_parameters(self):
        for _layer in self.layers:
            _layer.reset_parameteres()

class LGCN(Encoder):
    def __init__(self, c, args):
        super(LGCN, self).__init__(c)
        self.manifold = getattr(manifolds, args.manifold)()
        assert args.num_layers > 1
        dims, acts, self.curvatures = loren_layers.get_dim_act_curv(args)
        self.curvatures.append(self.c)
        lgnn_layers = []
        for i in range(len(dims) - 1):
            c_in, c_out = self.curvatures[i], self.curvatures[i + 1]
            in_dim, out_dim = dims[i], dims[i + 1]
            in_dim = in_dim - 1 if i != 0 else in_dim   # for layer more than 2
            act = acts[i]
            lgnn_layers.append(
                loren_layers.LorentzGraphNeuralNetwork(
                    self.manifold, in_dim, out_dim, c_in, c_out, args.dropout, act, args.bias, args.use_att
                )
            )
        self.layers = nn.Sequential(*lgnn_layers)
        self.encode_graph = True

    def encode(self, x, adj):
        # print(f'lorentz tangent {x.shape}')
        x_loren = self.manifold.normalize_input(x, c=self.curvatures[0])
        return super(LGCN, self).encode(x_loren, adj)

    def reset_parameters(self):
        for tmp_layer in self.layers:
            tmp_layer.reset_parameters()

class Shallow(Encoder):
    """
    Shallow Embedding method.
    Learns embeddings or loads pretrained embeddings and uses an MLP for classification.
    """

    def __init__(self, c, args):
        super(Shallow, self).__init__(c)
        self.manifold = getattr(manifolds, args.manifold)()
        self.use_feats = args.use_feats
        self.pretrained_embeddings = args.pretrained_embeddings
        self.n_nodes = args.n_nodes
        self.weights = torch.Tensor(args.n_nodes, args.dim)
        layers = []
        if args.pretrained_embeddings is not None and args.num_layers > 0:
            # MLP layers after pre-trained embeddings
            dims, acts = get_dim_act(args)
            if self.use_feats:
                dims[0] = args.feat_dim + self.weights.shape[1]
            else:
                dims[0] = self.weights.shape[1]
            for i in range(len(dims) - 1):
                in_dim, out_dim = dims[i], dims[i + 1]
                act = acts[i]
                layers.append(Linear(in_dim, out_dim, args.dropout, act, args.bias))
        self.layers = nn.Sequential(*layers)
        self.reset_parameteres()
        self.encode_graph = False

    def reset_parameteres(self):
        if not self.pretrained_embeddings:
            weights = self.manifold.init_weights(self.weights, self.c)
            trainable = True
        else:
            weights = torch.Tensor(np.load(self.pretrained_embeddings))
            assert weights.shape[0] == self.n_nodes, "The embeddings you passed seem to be for another dataset."
            trainable = False
        self.lt = manifolds.ManifoldParameter(weights, trainable, self.manifold, self.c)
        self.all_nodes = torch.LongTensor(list(range(self.n_nodes)))
        for _layer in self.layers:
            _layer.reset_parameters()

    def encode(self, x, adj):
        h = self.lt[self.all_nodes, :]
        if self.use_feats:
            h = torch.cat((h, x), 1)
        return super(Shallow, self).encode(h, adj)
