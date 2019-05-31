# -*- coding: utf-8 -*-
"""
Created on Feb 26, 2017
@author: Weiping Song
"""
import logging
import os
from typing import Optional, List

import numpy as np
import pandas as pd
import tensorflow as tf

from nnrecsys.legacy_gru4rec.data import SessionsIterator
from nnrecsys.legacy_gru4rec.evaluation import evaluate_sessions_batch


class GRU4Rec:
    def __init__(self, sess: tf.Session,
                 session_key: str,
                 item_key: str,
                 item_idx_key: str,
                 time_key: str,
                 layers: int,
                 rnn_size: int,
                 epochs: int,
                 batch_size: int,
                 item_id2idx_map: pd.Series,
                 lr: float,
                 optimizer: str,
                 momentum: float,
                 grad_cap: float,
                 decay_steps: int,
                 summary_steps: int,
                 decay: float,
                 sigma: float,
                 log_dir: str,
                 checkpoint_dir: str,
                 loss: str,
                 init_as_normal: bool,
                 hidden_act: str,
                 final_act: str,
                 dropout: float,
                 monitor_grads: bool,
                 restore_from_model: Optional[str] = None):
        self.sess = sess

        self.n_items = len(item_id2idx_map)
        self.item_id2idx_map = item_id2idx_map
        self.session_key = session_key
        self.item_key = item_key
        self.item_idx_key = item_idx_key
        self.time_key = time_key

        self.log_dir = log_dir

        self.layers = layers
        self.rnn_size = rnn_size
        self.batch_size = batch_size
        self.n_epochs = epochs
        self.learning_rate = lr
        self.optimizer = optimizer
        self.momentum = momentum

        self.init_as_normal = init_as_normal
        self.grad_cap = grad_cap
        self.monitor_grads = monitor_grads

        self.decay = decay
        self.decay_steps = decay_steps
        self.summary_steps = summary_steps

        self.sigma = sigma
        self.dropout_p_hidden = dropout

        if hidden_act == 'tanh':
            self.hidden_act = self.tanh
        elif hidden_act == 'relu':
            self.hidden_act = self.relu
        else:
            raise NotImplementedError

        if loss == 'cross-entropy':
            if final_act == 'tanh':
                self.final_activation = self.softmaxth
            else:
                self.final_activation = self.softmax
            self.loss_function = self.cross_entropy
        elif loss == 'bpr':
            if final_act == 'linear':
                self.final_activation = self.linear
            elif final_act == 'relu':
                self.final_activation = self.relu
            else:
                self.final_activation = self.tanh
            self.loss_function = self.bpr
        elif loss == 'top1':
            if final_act == 'linear':
                self.final_activation = self.linear
            elif final_act == 'relu':
                self.final_activation = self.relu
            else:
                self.final_activation = self.tanh
            self.loss_function = self.top1
        else:
            raise NotImplementedError

        self.checkpoint_dir = checkpoint_dir
        if not os.path.isdir(self.checkpoint_dir):
            raise Exception("[!] Checkpoint Dir not found")

        self.build_model()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)

        self.current_session = np.ones(self.batch_size) * -1

        self.predict_state = [np.zeros([self.batch_size, self.rnn_size], dtype=np.float32) for _ in
                              range(self.layers)]
        if restore_from_model:
            # use self.predict_state to hold hidden states during prediction.
            ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                self.saver.restore(sess, '{}/gru-model-{}'.format(self.checkpoint_dir, restore_from_model))

        self.train_writer = tf.summary.FileWriter(self.log_dir + '/train', self.sess.graph)
        self.test_writer = tf.summary.FileWriter(self.log_dir + '/test')

    # Activation functions
    @staticmethod
    def linear(X):
        return X

    @staticmethod
    def tanh(X):
        return tf.nn.tanh(X)

    @staticmethod
    def softmax(X):
        return tf.nn.softmax(X)

    @staticmethod
    def softmaxth(X):
        return tf.nn.softmax(tf.tanh(X))

    @staticmethod
    def relu(X):
        return tf.nn.relu(X)

    @staticmethod
    def sigmoid(X):
        return tf.nn.sigmoid(X)

    # Loss functions
    @staticmethod
    def cross_entropy(yhat):
        return tf.reduce_mean(-tf.log(tf.diag_part(yhat) + 1e-24))

    @staticmethod
    def bpr(yhat):
        yhatT = tf.transpose(yhat)
        return tf.reduce_mean(-tf.log(tf.nn.sigmoid(tf.diag_part(yhat) - yhatT) + 1e-24))

    def top1(self, yhat):
        yhatT = tf.transpose(yhat)
        term1 = tf.reduce_mean(tf.nn.sigmoid(-tf.diag_part(yhat) + yhatT) + tf.nn.sigmoid(yhatT ** 2), axis=0)
        term2 = tf.nn.sigmoid(tf.diag_part(yhat) ** 2) / self.batch_size
        return tf.reduce_mean(term1 - term2)

    def build_model(self, x, y):
        self.is_training = tf.placeholder(tf.bool, name='is_training')

        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        with tf.variable_scope('initializer'):
            sigma = self.sigma if self.sigma != 0 else np.sqrt(6.0 / (self.n_items + self.rnn_size))
            if self.init_as_normal:
                initializer = tf.random_normal_initializer(mean=0, stddev=sigma)
            else:
                initializer = tf.random_uniform_initializer(minval=-sigma, maxval=sigma)

        with tf.variable_scope('embedding'):
            embedding = tf.get_variable('embedding', [self.n_items, self.rnn_size], initializer=initializer)
            tf.summary.histogram('embeddings', embedding)
            inputs = tf.nn.embedding_lookup(embedding, x)

        with tf.variable_scope('gru_layer'):
            dropout_prob = tf.cond(self.is_training, lambda: self.dropout_p_hidden, lambda: 1.0)

            def rnn_cell():
                cell = tf.nn.rnn_cell.GRUCell(self.rnn_size, activation=self.hidden_act)
                drop_cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=dropout_prob)
                return drop_cell

            stacked_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell()] * self.layers)
            output, state = tf.nn.dynamic_rnn(stacked_cell, inputs=x, dtype=tf.float32, sequence_length=length(x))

            tf.summary.histogram('gru_activations', output)
            for variable in stacked_cell.variables:
                tf.summary.histogram('gru_vars/' + variable.name, variable)
            tf.summary.histogram('state', state)
            self.final_state = state

        # Consider ranking of only other examples of the minibatch as negative samples when training.
        with tf.name_scope('train_fc'):
            softmax_W = tf.get_variable('softmax_w', [self.n_items, self.rnn_size], initializer=initializer)
            softmax_b = tf.get_variable('softmax_b', [self.n_items], initializer=tf.initializers.constant(0))
            tf.summary.histogram('weights', softmax_W)
            tf.summary.histogram('biases', softmax_b)

            sampled_W = tf.nn.embedding_lookup(softmax_W, self.Y)
            sampled_b = tf.nn.embedding_lookup(softmax_b, self.Y)
            sampled_logits = tf.matmul(output, sampled_W, transpose_b=True) + sampled_b
            tf.summary.histogram('logits', sampled_logits)
            self.train_yhat = self.final_activation(sampled_logits)  # (batch_size, batch_size)

        # Consider ranking of all examples when predicting
        with tf.name_scope('predict_fc'):
            all_logits = tf.matmul(output, softmax_W, transpose_b=True) + softmax_b
            tf.summary.histogram('logits', all_logits)
            self.predict_yhat = self.final_activation(all_logits)
            tf.summary.histogram('weights', softmax_W)
            tf.summary.histogram('biases', softmax_b)

        with tf.name_scope('loss'):
            self.loss = self.loss_function(self.train_yhat)
            tf.summary.scalar('loss', self.loss)

        # Try different optimizers.
        with tf.name_scope('optimizer'):
            self.lr = tf.maximum(1e-5, tf.train.exponential_decay(self.learning_rate, self.global_step,
                                                                  self.decay_steps,
                                                                  self.decay, staircase=True))
            tf.summary.scalar('lr', self.lr)
            if self.optimizer == 'adagrad':
                optimizer = tf.train.AdagradOptimizer(self.lr)
            elif self.optimizer == 'adam':
                optimizer = tf.train.AdamOptimizer(self.lr)
            elif self.optimizer == 'adadelta':
                optimizer = tf.train.AdadeltaOptimizer(self.lr)
            elif self.optimizer == 'rmsprop':
                optimizer = tf.train.RMSPropOptimizer(self.lr, momentum=self.momentum)
            else:
                raise ValueError('Unknown optimizer')

        with tf.name_scope('gradients'):
            grads_and_vars = optimizer.compute_gradients(self.loss)

            tf.summary.merge([tf.summary.histogram("%s-grad" % g[1].name.replace(':', '_'), g[0])
                              for g in grads_and_vars])

            mean_grad_norm = tf.reduce_mean([tf.norm(grad) for grad, var in grads_and_vars])
            self.mean_grad_norm_summary = tf.summary.scalar('mean_grad_norm', mean_grad_norm)

            tf.summary.merge([tf.summary.scalar("%s-grad" % g[1].name.replace(':', '_'), tf.norm(g[0]))
                              for g in grads_and_vars])

            if self.grad_cap > 0:
                capped_gvs = [(tf.clip_by_norm(grad, self.grad_cap), var) for grad, var in grads_and_vars]
            else:
                capped_gvs = grads_and_vars

            mean_grad_norm = tf.reduce_mean([tf.norm(grad) for grad, var in capped_gvs])
            self.clipped_grad_norm_summary = tf.summary.scalar('clipped_mean_grad_norm', mean_grad_norm)

            self.train_op = optimizer.apply_gradients(capped_gvs, global_step=self.global_step)

        self.merged_summary = tf.summary.merge_all()

    def fit_epoch(self, train_iterator: SessionsIterator, epoch: int) -> List[float]:
        state = [np.zeros([self.batch_size, self.rnn_size], dtype=np.float32) for _ in range(self.layers)]

        losses = []
        for batches, exhausted_sessions_indices in train_iterator.generate():
            # prepare inputs, targeted outputs and hidden states
            for in_idx, out_idx in batches:
                feed_dict = {self.X: in_idx, self.Y: out_idx, self.is_training: True}

                for j in range(self.layers):
                    feed_dict[self.state[j]] = state[j]

                fetches = [self.loss,
                           self.final_state,
                           self.global_step,
                           self.lr,
                           self.train_op]
                loss, state, step, lr, _ = self.sess.run(fetches, feed_dict)

                if self.monitor_grads:
                    mean_grads, clipped_grads = self.sess.run([self.mean_grad_norm_summary,
                                                               self.clipped_grad_norm_summary], feed_dict)
                    self.train_writer.add_summary(mean_grads, step)
                    self.train_writer.add_summary(clipped_grads, step)

                if np.isnan(loss):
                    raise ValueError('{}, Nan error!'.format(epoch))
                losses.append(loss)

                if step == 1 or step % self.summary_steps == 0:
                    summary = self.sess.run(self.merged_summary, feed_dict)
                    self.train_writer.add_summary(summary, step)
                    logging.info('Step {}: lr: {}, train loss {}'.format(step, lr, np.mean(losses)))

            if len(exhausted_sessions_indices):
                for k in range(self.layers):
                    state[k][exhausted_sessions_indices] = 0
        return losses

    def validate(self, val_iterator: SessionsIterator) -> List[float]:
        state = [np.zeros([self.batch_size, self.rnn_size], dtype=np.float32) for _ in range(self.layers)]

        val_loss = []
        for batches, exhausted_sessions_indices in val_iterator.generate():
            for in_idx, out_idx in batches:
                feed_dict = {self.X: in_idx, self.Y: out_idx, self.is_training: False}

                for j in range(self.layers):
                    feed_dict[self.state[j]] = state[j]

                fetches = [self.loss, self.final_state]
                loss, state = self.sess.run(fetches, feed_dict)
                val_loss.append(loss)

            if len(exhausted_sessions_indices):
                for k in range(self.layers):
                    state[k][exhausted_sessions_indices] = 0

        return val_loss

    def fit(self, train_data: pd.DataFrame, val_data: pd.DataFrame):
        train_iterator = SessionsIterator(train_data, self.session_key, self.item_idx_key, self.batch_size)
        val_iterator = SessionsIterator(val_data, self.session_key, self.item_idx_key, self.batch_size)

        logging.info('fitting model...')
        for epoch in range(self.n_epochs):
            train_loss = self.fit_epoch(train_iterator,  epoch)
            mean_train_loss = np.mean(train_loss)
            self.add_external_summaries(self.train_writer, {'mean_loss': mean_train_loss}, epoch)

            val_loss = self.validate(val_iterator),
            mean_val_loss = np.mean(val_loss)
            self.add_external_summaries(self.test_writer, {'mean_loss': mean_val_loss}, epoch)

            logging.info('Epoch {}\tloss: {:.6f}\tval_loss: {:.6f}'
                         .format(epoch, mean_train_loss, mean_val_loss))

            # Evaluate on dev set
            recall, mrr = evaluate_sessions_batch(self, val_data,
                                                  self.session_key,
                                                  self.item_key)
            self.add_external_summaries(self.test_writer, {'recall': recall, 'mrr': mrr}, epoch)
            logging.info('Epoch {}, Recall@20: {}\tMRR@20: {}'.format(epoch, recall, mrr))

            self.saver.save(self.sess, '{}/gru-model'.format(self.checkpoint_dir), global_step=epoch)

    def add_external_summaries(self, writer, summaries: dict, step: int):
        for name, value in summaries.items():
            summary = tf.Summary()
            summary.value.add(tag=name, simple_value=value)
            writer.add_summary(summary, step)

    def predict_next_batch(self, session_ids: np.array, input_item_ids: np.array):
        """
        Gives predicton scores for a selected set of items. Can be used in batch mode to predict for multiple independent events (i.e. events of different sessions) at once and thus speed up evaluation.

        If the session ID at a given coordinate of the session_ids parameter remains the same during subsequent calls of the function, the corresponding hidden state of the network will be kept intact (i.e. that's how one can predict an item to a session).
        If it changes, the hidden state of the network is reset to zeros.

        Parameters
        --------
        session_ids : 1D array
            Contains the session IDs of the events of the batch. Its length must equal to the prediction batch size (batch param).
        input_item_ids : 1D array
            Contains the item IDs of the events of the batch. Every item ID must be must be in the training data of the network. Its length must equal to the prediction batch size (batch param).

        Returns
        --------
        out : pandas.DataFrame
            Prediction scores for selected items for every event of the batch.
            Columns: events of the batch; rows: items. Rows are indexed by the item IDs.

        """
        session_change = np.arange(self.batch_size)[session_ids != self.current_session]
        if len(session_change) > 0:  # change internal states with session changes
            for i in range(self.layers):
                self.predict_state[i][session_change] = 0.0
            self.current_session = session_ids.copy()

        in_idxs = self.item_id2idx_map[input_item_ids].values
        fetches = [self.predict_yhat, self.final_state]
        feed_dict = {self.X: in_idxs, self.is_training: False}
        for i in range(self.layers):
            feed_dict[self.state[i]] = self.predict_state[i]
        preds, self.predict_state = self.sess.run(fetches, feed_dict)
        preds = np.asarray(preds).T
        return pd.DataFrame(data=preds, index=self.item_id2idx_map.index)
