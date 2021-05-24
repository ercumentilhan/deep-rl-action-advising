import numpy as np
import tensorflow as tf


def sample_noise(shape):
    noise = tf.compat.v1.random_normal(shape)
    return noise

def noisy_dense(x, size, name, evaluation, bias=True, activation_fn=tf.compat.v1.identity, sigma_zero=0.5):
    mu_init = tf.compat.v1.random_uniform_initializer(minval=-1 * 1 / np.power(x.get_shape().as_list()[1], 0.5),
                                            maxval=1 * 1 / np.power(x.get_shape().as_list()[1], 0.5))

    sigma_init = tf.compat.v1.constant_initializer(sigma_zero / np.power(x.get_shape().as_list()[1], 0.5))

    p = sample_noise([x.get_shape().as_list()[1], 1])
    q = sample_noise([1, size])

    f_p = tf.compat.v1.multiply(tf.compat.v1.sign(p), tf.compat.v1.pow(tf.compat.v1.abs(p), 0.5))
    f_q = tf.compat.v1.multiply(tf.compat.v1.sign(q), tf.compat.v1.pow(tf.compat.v1.abs(q), 0.5))

    w_epsilon = f_p * f_q
    b_epsilon = tf.compat.v1.squeeze(f_q)

    w_mu = tf.compat.v1.get_variable(name + "/w_mu", [x.get_shape()[1], size], initializer=mu_init)
    w_sigma = tf.compat.v1.get_variable(name + "/w_sigma", [x.get_shape()[1], size], initializer=sigma_init)
    w = tf.compat.v1.cond(
        tf.compat.v1.equal(
            evaluation,
            tf.compat.v1.constant(True)),
        lambda: w_mu,
        lambda: w_mu + tf.compat.v1.multiply(w_sigma, w_epsilon))

    ret = tf.compat.v1.matmul(x, w)
    if bias:
        b_mu = tf.compat.v1.get_variable(name + "/bias_mu", [size], initializer=mu_init)
        b_sigma = tf.compat.v1.get_variable(name + "/bias_sigma", [size], initializer=sigma_init)
        b = tf.compat.v1.cond(tf.compat.v1.equal(evaluation, tf.compat.v1.constant(True)), lambda: b_mu,
                    lambda: b_mu + tf.compat.v1.multiply(b_sigma, b_epsilon))

        return activation_fn(ret + b), w_sigma, b_sigma
    else:
        return activation_fn(ret)