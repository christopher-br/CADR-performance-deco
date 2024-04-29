# Copyright (c) 2020, Ioana Bica

import numpy as np
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()


def equivariant_layer(x, h_dim, layer_id, treatment_id):
    xm = tf.reduce_sum(x, axis=1, keepdims=True)

    l_gamma = tf.layers.dense(x, h_dim, activation=None,
                              name='eqv_%s_treatment_%s_gamma' % (str(layer_id), str(treatment_id)),
                              reuse=tf.AUTO_REUSE)
    l_lambda = tf.layers.dense(xm, h_dim, activation=None, use_bias=False,
                               name='eqv_%s_treatment_%s_lambda' % (str(layer_id), str(treatment_id)),
                               reuse=tf.AUTO_REUSE)
    out = l_gamma - l_lambda
    return out


def invariant_layer(x, h_dim, treatment_id):
    rep_layer_1 = tf.layers.dense(x, h_dim, activation=tf.nn.elu,
                                  name='inv_treatment_%s' % str(treatment_id),
                                  reuse=tf.AUTO_REUSE)
    rep_sum = tf.reduce_sum(rep_layer_1, axis=1)

    return rep_sum


def sample_Z(m, n):
    return np.random.uniform(0, 1., size=[m, n])


def sample_X(X, size):
    start_idx = np.random.randint(0, X.shape[0], size)
    return start_idx


def sample_dosages(batch_size, num_treatments, num_dosages):
    dosage_samples = np.random.uniform(0., 1., size=[batch_size, num_treatments, num_dosages])
    return dosage_samples

def get_model_predictions(sess, num_treatments, num_dosage_samples, test_data):
    batch_size = test_data['x'].shape[0]

    treatment_dosage_samples = sample_dosages(batch_size, num_treatments, num_dosage_samples)
    factual_dosage_position = np.random.randint(num_dosage_samples, size=[batch_size])
    treatment_dosage_samples[range(batch_size), test_data['t'], factual_dosage_position] = test_data['d']

    treatment_dosage_mask = np.zeros(shape=[batch_size, num_treatments, num_dosage_samples])
    treatment_dosage_mask[range(batch_size), test_data['t'], factual_dosage_position] = 1

    I_logits = sess.run('inference_outcomes:0',
                        feed_dict={'input_features:0': test_data['x'],
                                   'input_treatment_dosage_samples:0': treatment_dosage_samples})

    Y_pred = np.sum(treatment_dosage_mask * I_logits, axis=(1, 2))

    return Y_pred