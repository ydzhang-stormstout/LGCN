import numpy as np
import tensorflow as tf
from utils import hlayers as layers
from utils import util
from models.base_hgattn import BaseHGAttN

class SpHGAT(BaseHGAttN):
    def inference(inputs, nb_classes, nb_nodes, training, attn_drop, ffd_drop,
            bias_mat, hid_units, n_heads, activation=tf.nn.elu):
        this_layer = layers.sp_lgnn_head
        attns = []
        inputs = tf.transpose(tf.squeeze(inputs, 0))
        inputs = tf.transpose(util.tf_mat_exp_map_zero(inputs))
        inputs = tf.expand_dims(inputs, 0)

        # input layer
        for _ in range(n_heads[0]):
            att = this_layer(inputs, adj_mat=bias_mat,
                out_sz=hid_units[0], activation=activation, nb_nodes=nb_nodes,
                in_drop=ffd_drop, coef_drop=attn_drop)
            attns.append(att)
        h_1 = tf.concat(attns, axis=-1)
        # hidden layer
        for i in range(1, len(hid_units)):
            attns = []
            for _ in range(n_heads[i]):
                att = this_layer(h_1, adj_mat=bias_mat,
                    out_sz=hid_units[i], activation=activation, nb_nodes=nb_nodes,
                    in_drop=ffd_drop, coef_drop=attn_drop)
                attns.append(att)
            h_1 = tf.concat(attns, axis=-1)
        out = []
        # output layer
        for i in range(n_heads[-1]):
            att = this_layer(h_1, adj_mat=bias_mat,
                out_sz=nb_classes, activation=lambda x: x, nb_nodes=nb_nodes,
                in_drop=ffd_drop, coef_drop=attn_drop)
            out.append(att)
        logits = tf.add_n(out) / n_heads[-1]
        return logits 