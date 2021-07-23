import os
import random
import numpy as np
import tensorflow as tf

from constants.general import *

# Twin DQN is to be supervised trained with the original DQNs samples and targets - has dropout enabled by default
# for uncertainty estimations
class DQNTwin(object):
    def __init__(self, id, config, session, stats):

        # Extract relevant configuration:
        self.config = {}
        self.config['env_type'] = config['env_type']
        self.config['env_n_actions'] = config['env_n_actions']
        self.config['env_obs_dims'] = config['env_obs_dims']
        self.config['env_obs_form'] = config['env_obs_form']

        bc_config_params = [
            'dqn_twin_learning_rate',
            'dqn_twin_adam_eps',
            'dqn_twin_dropout_rate',
            'dqn_twin_dropout_uc_ensembles',
            'dqn_twin_adam_eps',
            'dqn_twin_huber_loss_delta',
            'dqn_twin_n_hidden_layers',
            'dqn_twin_hidden_size_1',
            'dqn_twin_hidden_size_2',
        ]
        for param in bc_config_params:
            self.config[param] = config[param]

        self.id = id
        self.session = session
        self.stats = stats

        # Scoped names
        self.name = self.id + '/' + 'ONLINE'

        self.tf_vars = {}
        self.tf_vars['obs'] = self.build_input_obs(self.name)

        self.tf_vars['dropout_rate'] = tf.compat.v1.placeholder(tf.float32, shape=(), name='DROPOUT_RATE')

        self.tf_vars['pre_fc_features'], self.tf_vars['mid_fc_features'], self.tf_vars['q_values'] = \
            self.build_network(self.name, self.tf_vars['obs'], self.config['dqn_twin_n_hidden_layers'],
                               self.config['dqn_twin_hidden_size_1'], self.config['dqn_twin_hidden_size_2'],
                               self.config['env_n_actions'])

        self.build_training_ops()

    # ==================================================================================================================

    def build_input_obs(self, name):
        if self.config['env_obs_form'] == NONSPATIAL:
            return tf.compat.v1.placeholder(tf.compat.v1.float32, [None, self.config['env_obs_dims'][0]],
                                            name=name + '_OBS')
        elif self.config['env_obs_form'] == SPATIAL:
            return tf.compat.v1.placeholder(tf.compat.v1.float32, [None, self.config['env_obs_dims'][0],
                                                                   self.config['env_obs_dims'][1],
                                                                   self.config['env_obs_dims'][2]], name=name + '_OBS')

    # ==================================================================================================================

    def conv_layers(self, scope, inputs):
        if self.config['env_type'] == GRIDWORLD or \
                self.config['env_type'] == MINATAR:
            layer_1 = tf.compat.v1.layers.conv2d(inputs=inputs,
                                                 filters=16,
                                                 kernel_size=(3, 3),
                                                 strides=(1, 1),
                                                 padding='VALID',
                                                 kernel_initializer=tf.keras.initializers.VarianceScaling(),
                                                 activation=tf.nn.relu,
                                                 name='CONV_LAYER_1')
            output = tf.compat.v1.layers.flatten(layer_1)
            return output

        elif self.config['env_type'] == ALE:
            with tf.compat.v1.variable_scope(scope, reuse=tf.compat.v1.AUTO_REUSE):
                layer_1 = tf.compat.v1.layers.conv2d(inputs=inputs,
                                                     filters=32,
                                                     kernel_size=(8, 8),
                                                     strides=(4, 4),
                                                     padding='VALID',
                                                     kernel_initializer=tf.keras.initializers.VarianceScaling(),
                                                     activation=tf.nn.relu,
                                                     name='CONV_LAYER_1')

                layer_2 = tf.compat.v1.layers.conv2d(inputs=layer_1,
                                                     filters=64,
                                                     kernel_size=(4, 4),
                                                     strides=(2, 2),
                                                     padding='VALID',
                                                     kernel_initializer=tf.keras.initializers.VarianceScaling(),
                                                     activation=tf.nn.relu,
                                                     name='CONV_LAYER_2')

                layer_3 = tf.compat.v1.layers.conv2d(inputs=layer_2,
                                                     filters=64,
                                                     kernel_size=(3, 3),
                                                     strides=(1, 1),
                                                     padding='VALID',
                                                     kernel_initializer=tf.keras.initializers.VarianceScaling(),
                                                     activation=tf.nn.relu,
                                                     name='CONV_LAYER_3')

                output = tf.compat.v1.layers.flatten(layer_3)
                return output

    # ==================================================================================================================

    def dense_layers(self, scope, inputs, hidden_number, hidden_size_1, hidden_size_2, output_size, head_id):
        hidden_sizes = (hidden_size_1, hidden_size_2)
        with tf.compat.v1.variable_scope(scope, reuse=tf.compat.v1.AUTO_REUSE):

            layer_1_in = inputs

            hidden_layers = []
            for i in range(hidden_number):
                if i == 0:
                    if self.config['env_obs_form'] == NONSPATIAL:
                        hidden_layer_in = layer_1_in
                    elif self.config['env_obs_form'] == SPATIAL:  # First dense layer dropout
                        hidden_layer_in = tf.compat.v1.nn.dropout(layer_1_in, name='DROPOUT_LAYER_' + str(i + 1),
                                                                  rate=self.tf_vars['dropout_rate'])

                    # If first layer Dropout is to be disabled always (will reduce variance):
                    # hidden_layer_in = layer_1_in
                else:
                    hidden_layer_in = tf.compat.v1.nn.dropout(hidden_layers[-1], name='DROPOUT_LAYER_' + str(i + 1),
                                                              rate=self.tf_vars['dropout_rate'])

                hidden_layers.append(tf.compat.v1.layers.dense(hidden_layer_in, hidden_sizes[i], use_bias=True,
                                                               kernel_initializer=tf.keras.initializers.VarianceScaling(),
                                                               activation=tf.nn.relu,
                                                               name='DENSE_LAYER_' + str(head_id) + '_' + str(i + 1)))

            layer_final_in = tf.compat.v1.nn.dropout(hidden_layers[-1], name='DROPOUT_LAYER_FINAL',
                                                     rate=self.tf_vars['dropout_rate'])

            layer_final = tf.compat.v1.layers.dense(layer_final_in, output_size, use_bias=True,
                                                    kernel_initializer=tf.keras.initializers.VarianceScaling(),
                                                    activation=None, name='DENSE_LAYER_' + str(head_id) + '_FINAL')
            return layer_final, layer_final_in

    # ==================================================================================================================

    def build_network(self, name, input, n_hidden_layers, dense_hidden_size_1, dense_hidden_size_2, output_size):
        pre_fc_features = None
        if self.config['env_obs_form'] == NONSPATIAL:
            pre_fc_features = input
        elif self.config['env_obs_form'] == SPATIAL:
            pre_fc_features = self.conv_layers(name, input)

        q_values, mid_fc_features = self.dense_layers(name,
                                                         inputs=pre_fc_features,
                                                         hidden_number=n_hidden_layers,
                                                         hidden_size_1=dense_hidden_size_1,
                                                         hidden_size_2=dense_hidden_size_2,
                                                         output_size=output_size,
                                                         head_id=1)

        return pre_fc_features, mid_fc_features, q_values

    # ==================================================================================================================

    def build_training_ops(self):
        self.tf_vars['action'] = tf.compat.v1.placeholder(tf.compat.v1.int32, [None], name='ACTIONS_' + str(self.id))

        self.tf_vars['td_target'] = tf.compat.v1.placeholder(tf.compat.v1.float32, [None],
                                                             name='LABELS_' + str(self.id))

        action_one_hot = tf.compat.v1.one_hot(self.tf_vars['action'], self.config['env_n_actions'], 1.0, 0.0)

        q_values_reduced = tf.compat.v1.reduce_sum(tf.compat.v1.math.multiply(self.tf_vars['q_values'],
                                                                              action_one_hot), reduction_indices=1)

        self.tf_vars['td_error'] = tf.compat.v1.abs(self.tf_vars['td_target'] - q_values_reduced)

        self.tf_vars['loss'] = tf.compat.v1.losses.huber_loss(labels=self.tf_vars['td_target'],
                                                              predictions=q_values_reduced,
                                                              delta=self.config['dqn_twin_huber_loss_delta'])

        optimizer = tf.compat.v1.train.AdamOptimizer(self.config['dqn_twin_learning_rate'],
                                                     epsilon=self.config['dqn_twin_adam_eps'])

        self.tf_vars['grads_update'] = optimizer.minimize(self.tf_vars['loss'])

    # ==================================================================================================================

    def train_model_with_feed_dict(self, feed_dict_in, is_batch):
        feed_dict = {
            self.tf_vars['obs']: feed_dict_in['obs'],
            self.tf_vars['action']: feed_dict_in['action'],
            self.tf_vars['td_target']: feed_dict_in['td_target'],
            self.tf_vars['dropout_rate']: self.config['dqn_twin_dropout_rate']
        }

        loss_batch, _, _ = \
            self.session.run([self.tf_vars['loss'], self.tf_vars['grads_update'], self.tf_vars['q_values']],
                             feed_dict=feed_dict)

        return loss_batch if is_batch else loss_batch[0]

    # ==================================================================================================================
    # Returns the following:
    # -- mean variance of the Q-values (regarded as uncertainty)
    # -- variance of the Q-values
    # -- Q-values

    def get_uncertainty(self, obs):
        if self.config['env_type'] == ALE:
            obs = np.moveaxis(np.asarray(obs, dtype=np.float32) / 255.0, 0, -1)

        obs_batch = [obs.astype(dtype=np.float32)] * self.config['dqn_twin_dropout_uc_ensembles']
        feed_dict = {self.tf_vars['obs']: obs_batch, self.tf_vars['dropout_rate']: self.config['dqn_twin_dropout_rate']}

        q_values = np.asarray(self.session.run(self.tf_vars['q_values'], feed_dict=feed_dict))
        q_values_vars = np.var(q_values, axis=0)

        return np.mean(q_values_vars), q_values_vars, q_values
