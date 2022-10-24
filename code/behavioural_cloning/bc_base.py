import os
import random
import numpy as np
import tensorflow as tf
from behavioural_cloning.bc_replay_buffer import ReplayBuffer

from constants.general import *
# from constants.env_info import *
# from constants.demonstrator_policy import *
# from constants.demonstrator_dataset import *
# from constants.config_sets import *

class BehaviouralCloning(object):
    def __init__(self, id, config, session, stats):

        # Extract relevant configuration:
        self.config = {}
        self.config['env_type'] = config['env_type']
        self.config['env_n_actions'] = config['env_n_actions']
        self.config['env_obs_dims'] = config['env_obs_dims']
        self.config['env_obs_form'] = config['env_obs_form']  # NONSPATIAL (0): Vector input, SPATIAL (1): Tensor input

        bc_config_params = [
            'bc_hidden_size',
            'bc_batch_size',
            'bc_learning_rate',
            'bc_adam_eps',
            'bc_dropout_rate',
            'bc_uc_ensembles',
        ]
        for param in bc_config_params:
            self.config[param] = config[param]

        self.id = id
        self.session = session
        self.stats = stats

        # Scoped names
        self.name = self.id + '/' + 'BC_ONLINE'

        self.tf_vars = {}
        self.tf_vars['obs'] = self.build_input_obs(self.name)

        self.replay_memory = None

        self.post_init_steps = 0
        self.training_steps = 0

        self.create_replay_memory()

        self.minibatch_keys = ('obs', 'action')

        self.dropout_rate_ph = tf.compat.v1.placeholder(tf.float32, shape=(), name='DROPOUT_RATE')

        self.tf_vars['pre_fc_features'], self.tf_vars['mid_fc_features'], self.tf_vars['action_logits'],\
            self.tf_vars['action_probs']= \
            self.build_network(self.name, self.tf_vars['obs'], self.config['bc_hidden_size'],
                               self.config['env_n_actions'])

        self.build_training_ops()

    # ==================================================================================================================

    def create_replay_memory(self):
        self.replay_memory = ReplayBuffer(1e6)

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

    def dense_layers(self, scope, inputs, hidden_size, output_size):
        if self.config['env_obs_form'] == NONSPATIAL:  # Dropout in final layer(s) only!
            with tf.compat.v1.variable_scope(scope, reuse=tf.compat.v1.AUTO_REUSE):
                layer_1_out = tf.compat.v1.layers.dense(inputs, hidden_size, use_bias=True,
                                                        kernel_initializer=tf.keras.initializers.VarianceScaling(),
                                                        activation=tf.nn.relu, name='DENSE_LAYER_1')

                layer_2_in = tf.compat.v1.nn.dropout(layer_1_out, name='DROPOUT_LAYER_2', rate=self.dropout_rate_ph)
                layer_2_out = tf.compat.v1.layers.dense(layer_2_in, output_size, use_bias=True,
                                                        kernel_initializer=tf.keras.initializers.VarianceScaling(),
                                                        activation=None, name='DENSE_LAYER_2')
            return layer_2_out, layer_1_out

        elif self.config['env_obs_form'] == SPATIAL:  # Full Dropout when following a ConvNet
            with tf.compat.v1.variable_scope(scope, reuse=tf.compat.v1.AUTO_REUSE):
                layer_1_in = tf.compat.v1.nn.dropout(inputs, name='DROPOUT_LAYER_1', rate=self.dropout_rate_ph)
                layer_1_out = tf.compat.v1.layers.dense(layer_1_in, hidden_size, use_bias=True,
                                                    kernel_initializer=tf.keras.initializers.VarianceScaling(),
                                                    activation=tf.nn.relu, name='DENSE_LAYER_1')

                layer_2_in = tf.compat.v1.nn.dropout(layer_1_out, name='DROPOUT_LAYER_2', rate=self.dropout_rate_ph)
                layer_2_out = tf.compat.v1.layers.dense(layer_2_in, output_size, use_bias=True,
                                                    kernel_initializer=tf.keras.initializers.VarianceScaling(),
                                                    activation=None, name='DENSE_LAYER_2')
            return layer_2_out, layer_1_out

    # ==================================================================================================================

    def fix_batch_form(self, var, is_batch):
        return var if is_batch else [var]

    # ==================================================================================================================

    def build_network(self, name, input, hidden_size, output_size):
        pre_fc_features = None
        if self.config['env_obs_form'] == NONSPATIAL:
            pre_fc_features = input
        elif self.config['env_obs_form'] == SPATIAL:
            pre_fc_features = self.conv_layers(name, input)

        action_logits, mid_fc_features = self.dense_layers(name, inputs=pre_fc_features, hidden_size=hidden_size,
                                                           output_size=output_size)
        action_probs = tf.compat.v1.nn.softmax(action_logits)
        return pre_fc_features, mid_fc_features, action_logits, action_probs

    # ==================================================================================================================

    def build_training_ops(self):
        self.tf_vars['action'] = tf.compat.v1.placeholder(tf.compat.v1.int32, [None], name='ACTIONS_' + str(self.id))
        action_one_hot = tf.compat.v1.one_hot(self.tf_vars['action'], self.config['env_n_actions'], 1.0, 0.0)

        loss_all = tf.compat.v1.nn.sigmoid_cross_entropy_with_logits(logits=self.tf_vars['action_logits'],
                                                                 labels=action_one_hot)

        self.tf_vars['loss_batch'] = tf.compat.v1.reduce_mean(loss_all, reduction_indices=1)
        self.tf_vars['loss'] = tf.compat.v1.reduce_mean(loss_all)

        optimizer = tf.compat.v1.train.AdamOptimizer(self.config['bc_learning_rate'], epsilon=self.config['bc_adam_eps'])
        self.tf_vars['grads_update'] = optimizer.minimize(self.tf_vars['loss'])

    # ==================================================================================================================

    def feedback_observe(self, obs, action):
        self.replay_memory.add(obs, action)

    # ==================================================================================================================

    def feedback_learn(self):
        self.post_init_steps += 1
        loss = self.train_model()
        return loss

    # ==================================================================================================================

    def train_model(self):
        self.training_steps += 1

        batch_size = min(self.config['bc_batch_size'], self.replay_memory.__len__())

        minibatch_ = self.replay_memory.sample(batch_size, in_numpy_form=True)
        minibatch = {}
        for i, key in enumerate(self.minibatch_keys):
            if key == 'obs':  # Float Correction
                if self.config['env_type'] == ALE:
                    # LazyFrames correction
                    minibatch[key] = np.moveaxis(np.asarray(minibatch_[i], dtype=np.float32) / 255.0, 1, -1)
                else:
                    minibatch[key] = np.asarray(minibatch_[i], dtype=np.float32)
            else:
                minibatch[key] = minibatch_[i]

        loss, loss_batch = self.get_grads_update(minibatch)
        return loss

    # ==================================================================================================================

    def get_action_logits(self, obs):
        if self.config['env_type'] == ALE:
            obs = np.moveaxis(np.asarray(obs, dtype=np.float32) / 255.0, 0, -1)
        feed_dict = {self.tf_vars['obs']: [obs.astype(dtype=np.float32)]}
        return self.session.run(self.tf_vars['action_logits'], feed_dict=feed_dict)

    # ==================================================================================================================

    def get_action_probs(self, obs, normalise=True):
        if self.config['env_type'] == ALE:
            if normalise:
                obs = np.moveaxis(np.asarray(obs, dtype=np.float32) / 255.0, 0, -1)
        feed_dict = {self.tf_vars['obs']: [obs.astype(dtype=np.float32)], self.dropout_rate_ph: 0.0}
        return self.session.run(self.tf_vars['action_probs'], feed_dict=feed_dict)[0]

    def get_action_probs_batch(self, obs_batch):
        feed_dict = {self.tf_vars['obs']: obs_batch.astype(dtype=np.float32), self.dropout_rate_ph: 0.0}
        return self.session.run(self.tf_vars['action_probs'], feed_dict=feed_dict)

    # ==================================================================================================================

    def get_latent_features(self, obs):
        if self.config['env_type'] == ALE:
            obs = np.moveaxis(np.asarray(obs, dtype=np.float32) / 255.0, 0, -1)
        feed_dict = {self.tf_vars['obs']: [obs.astype(dtype=np.float32)]}
        return self.session.run(self.tf_vars['latent_features'], feed_dict=feed_dict)

    # ==================================================================================================================

    def get_loss(self, minibatch):
        feed_dict, is_batch = self.arrange_feed_dict(minibatch)
        loss_batch = self.session.run(self.tf_vars['loss'], feed_dict=feed_dict)
        return loss_batch if is_batch else loss_batch[0]

    # ==================================================================================================================

    def get_grads_update(self, minibatch):
        feed_dict, is_batch = self.arrange_feed_dict(minibatch)
        loss_batch, _, action_logits, loss_batch_all = \
            self.session.run([self.tf_vars['loss'], self.tf_vars['grads_update'], self.tf_vars['action_logits'],
                              self.tf_vars['loss_batch']],
                             feed_dict=feed_dict)
        return loss_batch if is_batch else loss_batch[0], loss_batch_all

    # ==================================================================================================================

    def arrange_feed_dict(self, minibatch):
        is_batch = isinstance(minibatch['action'], list) or isinstance(minibatch['action'], np.ndarray)

        obs_batch = minibatch['obs'] if isinstance(minibatch['obs'], list) \
            else minibatch['obs'].astype(dtype=np.float32)

        obs_batch = self.fix_batch_form(obs_batch, is_batch)
        action_batch = self.fix_batch_form(minibatch['action'], is_batch)

        feed_dict = {self.tf_vars['obs']: obs_batch,
                     self.tf_vars['action']: action_batch,
                     self.dropout_rate_ph: self.config['bc_dropout_rate']}

        return feed_dict, is_batch

    # ==================================================================================================================

    def get_action(self, obs):
        action_probs = self.get_action_probs(obs)
        return np.argmax(action_probs)

    # ==================================================================================================================

    def restore(self, model_path):
        print('Restoring...')
        print('Scope: {}'.format(self.id))
        print('# of variables: {}'.format(len(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES,
                                                                          scope=self.id))))
        loader = tf.compat.v1.train.Saver(var_list=tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES,
                                                                               scope=self.id))
        loader.restore(self.session, model_path)

    # ==================================================================================================================

    def get_uncertainty(self, obs, normalise=True):
        if self.config['env_type'] == ALE:
            if normalise:
                obs = np.moveaxis(np.asarray(obs, dtype=np.float32) / 255.0, 0, -1)

        obs_batch = [obs.astype(dtype=np.float32)] * self.config['bc_uc_ensembles']
        feed_dict = {self.tf_vars['obs']: obs_batch, self.dropout_rate_ph: self.config['bc_dropout_rate']}

        probs = np.asarray(self.session.run(self.tf_vars['action_probs'], feed_dict=feed_dict))
        probs_vars = np.var(probs, axis=0)

        return np.mean(probs_vars)

    def get_uncertainty_batch(self, obs_batch_in):
        obs_batch = np.repeat(obs_batch_in, self.config['bc_uc_ensembles'], axis=0)
        feed_dict = {self.tf_vars['obs']: obs_batch, self.dropout_rate_ph: self.config['bc_dropout_rate']}

        probs = np.asarray(self.session.run(self.tf_vars['action_probs'], feed_dict=feed_dict))

        n_obs = int(probs.shape[0] / self.config['bc_uc_ensembles'])
        probs_vars = []
        for i in range(n_obs):
            i_start = i * self.config['bc_uc_ensembles']
            i_end = i_start + self.config['bc_uc_ensembles']
            probs_vars.append(np.mean(np.var(probs[i_start:i_end, :], axis=0)))

        return probs_vars

    # ==================================================================================================================

    # def get_batch_uncertainty(self, obs_batch, normalise=True):
    #     obs_batch = [obs.astype(dtype=np.float32)] * self.config['bc_uc_ensembles']
    #     feed_dict = {self.tf_vars['obs']: obs_batch, self.dropout_rate_ph: self.config['bc_dropout_rate']}
    #
    #     probs = np.asarray(self.session.run(self.tf_vars['action_probs'], feed_dict=feed_dict))
    #     probs_vars = np.var(probs, axis=0)
    #
    #     return np.mean(probs_vars)

