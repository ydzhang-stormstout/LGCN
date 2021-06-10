import numpy as np
import tensorflow as tf
from utils import util
import numpy as np

np.set_printoptions(threshold=np.inf)

def sp_lgnn_head(seq, out_sz, adj_mat, activation, nb_nodes, in_drop=0.0, coef_drop=0.0, residual=False):
    with tf.name_scope('sp_lgnn'):
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)

        seq = tf.squeeze(seq, 0)
        seq = tf.transpose(seq)
        seq_size = seq.shape.as_list()
        with tf.name_scope("att") as scope:
            W = tf.get_variable(name=scope + 'W', shape=[out_sz, seq_size[0]], initializer=tf.contrib.layers.xavier_initializer())
        ## log map
        seq_log = util.tf_mat_log_map_zero(seq)
        seq_fts_log = tf.matmul(W, seq_log)
        seq_fts_exp = tf.transpose(util.tf_my_mat_exp_map_zero(tf.transpose(seq_fts_log)))
        # attention alpha
        adj_indices = adj_mat.indices
        adj_idx_x = adj_indices[:, 0]
        adj_idx_y = adj_indices[:, 1]
        fts_x = tf.gather(tf.transpose(seq_fts_exp), adj_idx_x)
        fts_y = tf.gather(tf.transpose(seq_fts_exp), adj_idx_y)
        # adj_indices
        sparse_distance = util.tf_my_poincare_list_distance(fts_x, fts_y)
        att = tf.SparseTensor(indices=adj_indices, values=-sparse_distance, dense_shape=adj_mat.dense_shape)
        coefs = tf.sparse.softmax(att)

        seq_fts = tf.transpose(seq_fts_exp)
        seq_fts = tf.expand_dims(seq_fts, 0)
        if coef_drop != 0.0:
            coefs = tf.SparseTensor(indices=coefs.indices,
                                    values=tf.nn.dropout(coefs.values, 1.0 - coef_drop),
                                    dense_shape=coefs.dense_shape)
        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

        coefs = tf.sparse.reshape(coefs, [nb_nodes, nb_nodes])
        seq_fts = tf.squeeze(seq_fts)
        seq_fts_lorentz = util.poincare2lorentz(seq_fts)
        vals = tf.sparse_tensor_dense_matmul(coefs, seq_fts_lorentz)
        vals = util.lorentz_centroid(vals, nb_nodes, out_sz)
        vals = util.lorentz2poincare(vals, nb_nodes)
        vals = util.tf_my_mat_log_map_zero(vals)

        vals = tf.expand_dims(vals, axis=0)
        vals.set_shape([1, nb_nodes, out_sz])
        ret_before = tf.contrib.layers.bias_add(vals)
        ret_before = tf.squeeze(ret_before)
        ret = util.tf_my_mat_exp_map_zero(activation(ret_before))
        ret = tf.expand_dims(ret, 0)
        return ret 