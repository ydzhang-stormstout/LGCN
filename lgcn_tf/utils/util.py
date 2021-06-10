import logging
import tensorflow as tf
import numpy as np
from numpy import linalg as LA
from numpy import random as np_random
import os
import random

PROJ_EPS = 1e-5
EPS = 1e-15
MAX_TANH_ARG = 15.0
clip_value = 0.98

def tf_project_hyp_vecs(x, c):
    # Projection op. Need to make sure hyperbolic embeddings are inside the unit ball.
    return tf.clip_by_norm(t=x, clip_norm=(1. - PROJ_EPS) / np.sqrt(c), axes=[1])

######################## x,y have shape [batch_size, emb_dim] in all tf_* functions ################

# # Real x, not vector!
def tf_atanh(x):
    return tf.atanh(tf.minimum(x, 1. - EPS)) # Only works for positive real x.

def tf_my_mob_vec_exp_map_zero(vec, c=1):
    norm = tf.norm(vec)
    coef = tf.tanh(norm) / norm
    return vec * coef

def tf_my_mob_addition(u, v, c=1):
    norm_u_sq = tf.norm(u, axis=1) ** 2
    norm_v_sq = tf.norm(v, axis=1) ** 2
    uv_dot_times = 4 * tf.reduce_mean(u * v, axis=1)
    denominator = 1 + uv_dot_times + norm_u_sq * norm_v_sq
    coef_1 = (1 + uv_dot_times + norm_v_sq) / denominator
    coef_2 = (1 - norm_u_sq) / denominator
    return tf.multiply(tf.expand_dims(coef_1, 1), u) + tf.multiply(tf.expand_dims(coef_2, 1), v)

def tf_my_prod_mob_addition(u, v, c):
    # input [nodes, features]
    norm_u_sq = tf.norm(u, axis=1) ** 2
    norm_v_sq = tf.norm(v, axis=1) ** 2
    uv_dot_times = 4 * tf.reduce_mean(u * v, axis=1) * c
    denominator = 1 + uv_dot_times + norm_u_sq * norm_v_sq * c * c
    coef_1 = (1 + uv_dot_times + c * norm_v_sq) / denominator
    coef_2 = (1 - c * norm_u_sq) / denominator
    return tf.multiply(tf.expand_dims(coef_1, 1), u) + tf.multiply(tf.expand_dims(coef_2, 1), v)

def tf_my_mat_log_map_zero(M, c=1):
    M = tf.transpose(M)
    M = M + EPS
    M = tf.clip_by_norm(M, clip_norm=clip_value, axes=0)
    m_norm = tf.norm(M, axis=0)
    atan_norm = tf_atanh(m_norm)
    M_cof = atan_norm / m_norm
    res = M * M_cof
    return tf.transpose(res)

def tf_my_prod_mat_log_map_zero(M, c):
    sqrt_c = tf.sqrt(c)
    # M = tf.transpose(M)
    M = M + EPS
    M = tf.clip_by_norm(M, clip_norm=clip_value, axes=0)
    m_norm = tf.norm(M, axis=0)
    atan_norm = tf.atanh(tf.clip_by_value(m_norm*sqrt_c, clip_value_min=-0.9, clip_value_max=0.9))
    M_cof = atan_norm / m_norm / sqrt_c
    res = M * M_cof
    return res


def tf_my_mat_exp_map_zero(vecs):
    vecs = vecs + EPS
    vecs = tf.transpose(vecs)
    vecs = tf.clip_by_norm(vecs, clip_norm=clip_value, axes=0)
    norms = tf.norm(vecs, axis=0)
    c_tanh = tf.tanh(norms)
    coef = c_tanh / norms
    res = vecs * coef
    return tf.transpose(res)

def tf_my_prod_mat_exp_map_zero(vecs, c):
    sqrt_c = tf.sqrt(c)
    vecs = vecs + EPS
    vecs = tf.transpose(vecs)
    vecs = tf.clip_by_norm(vecs, clip_norm=clip_value, axes=0)
    norms = tf.norm(vecs, axis=0)
    c_tanh = tf.tanh(norms*sqrt_c)
    coef = c_tanh / norms / sqrt_c
    res = vecs * coef
    return tf.transpose(res)


def tf_my_poincare_list_distance(mat_x, mat_y):
    # input: [nodes, features]
    norm_x_sq = tf.norm(mat_x, axis=1) ** 2
    norm_y_sq = tf.norm(mat_y, axis=1) ** 2
    dif_xy = tf.norm(mat_x-mat_y, axis=1) ** 2
    demoninator = tf.multiply(1-norm_x_sq, 1-norm_y_sq)
    res = tf.clip_by_value(1 + 2 * dif_xy/ demoninator, clip_value_min=1.000001, clip_value_max=100)
    return tf.acosh(res)


def tf_my_mob_mat_distance(mat_x, mat_y):
    # input shape: [features, nodes]
    mat = tf_my_mob_mat_addition(-mat_x, mat_y)
    # mat = mat + EPS
    mat_norm = tf.norm(mat, axis=2)
    mat_norm = tf.clip_by_value(mat_norm, clip_value_min=1e-8, clip_value_max=clip_value)
    res = 2. * tf.atanh(mat_norm)
    return res

def poincare2lorentz(mat):
    # input shape [nodes, features]
    norm_sq = tf.norm(mat, axis=1, keep_dims=True) ** 2
    res_t = tf.concat([1+norm_sq, 2*mat], 1) / (1 - norm_sq)
    res = res_t
    return res

def lorentz2poincare(mat, n):
    # shape [nodes, features]
    # print('l2p', mat.shape.as_list())
    # n = mat.shape.as_list()[0]
    d = mat.shape.as_list()[1] - 1
    vector = tf.slice(mat, [0, 1], [n, d])
    head = tf.slice(mat, [0, 0], [n, 1])
    return vector / (head+1)

def lorentz_inner(x, y, n, d):
    """
    lorentz inner product
    :param x, y: [nodes, features]
    :param n: number of nodes
    :param d: dimension of features, i.e., len(features) -1
    :return:
    """
    x_head = tf.slice(x, [0, 0], [n, 1])
    x_tail = tf.slice(x, [0, 1], [n, d])
    y_head = tf.slice(y, [0, 0], [n, 1])
    y_tail = tf.slice(y, [0, 1], [n, d])
    return tf.reduce_sum(-tf.multiply(x_head, y_head), axis=1) + tf.reduce_sum(tf.multiply(x_tail, y_tail), axis=1)

def lorentz_centroid(x, n, d):
    # input weighted matrix: [nodes, features]
    coef = tf.sqrt(tf.abs(lorentz_inner(x, x, n, d)))
    coef = tf.clip_by_value(coef, clip_value_min=1e-8, clip_value_max=coef)
    res = tf.divide(x, tf.expand_dims(tf.transpose(coef), 1))
    return res

def tf_mat_exp_map_zero(M, c=1.):
    M = M + EPS
    sqrt_c = tf.sqrt(c)
    M = tf.clip_by_norm(M, clip_norm=clip_value / sqrt_c, axes=0)
    norms = tf.norm(M, axis=0)
    c_tanh = tf.tanh(norms * sqrt_c)
    coef = c_tanh / norms / sqrt_c
    res = M * coef
    return res

# # each row
def tf_mat_log_map_zero(M, c=1):
    M = M + EPS
    M = tf.clip_by_norm(M, clip_norm=clip_value, axes=0)
    m_norm = tf.norm(M, axis=0)
    atan_norm = tf_atanh(m_norm)
    M_cof = atan_norm / m_norm
    res = M * M_cof
    return res


