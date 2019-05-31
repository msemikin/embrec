# -*- coding: utf-8 -*-
"""
Created on Feb 26 2017
Author: Weiping Song
"""
import os
import tensorflow as tf
import pandas as pd
import numpy as np
import argparse
import logging
import yaml

from tensorflow.python import debug as tf_debug

from nnrecsys.legacy_gru4rec import model, evaluation
from nnrecsys.legacy_gru4rec.monitor import monitor_gpu

logging.basicConfig(format='[%(asctime)s - %(levelname)s] %(message)s')
logging.root.setLevel(logging.INFO)


PATH_TO_TRAIN = 'data/train.txt'
PATH_TO_VAL = 'data/val.txt'
PATH_TO_TEST = 'data/test.txt'
PATH_TO_ITEM_ID2IDX_MAP = 'data/item_id2idx_map.pkl'

ITEM_KEY = 'item_id'
ITEM_IDX_KEY = 'item_idx'
SESSION_KEY = 'session_id'
TIME_KEY = 'timestamp'


def parse_args():
    parser = argparse.ArgumentParser(description='GRU4Rec args')
    parser.add_argument('--evaluate', default=False)
    parser.add_argument('--experiment', required=True, type=str)
    parser.add_argument('--restore_model', default=False, type=str)

    parser.add_argument('--layers', default=1, type=int)
    parser.add_argument('--rnn_size', default=100, type=int)
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--batch_size', default=50, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--optimizer', default='rmsprop', type=str)
    parser.add_argument('--hidden_act', default='relu', type=str)
    parser.add_argument('--final_act', default='softmax', type=str)
    parser.add_argument('--loss', default='cross-entropy', type=str)
    parser.add_argument('--dropout', default=1.0, type=float)
    parser.add_argument('--momentum', default=0, type=float)
    parser.add_argument('--grad_cap', default=0, type=float)
    parser.add_argument('--debug', default=False, type=bool)
    parser.add_argument('--monitor_gpu', default=False, type=bool)

    return parser.parse_args()


if __name__ == '__main__':
    cli = parse_args()
    logging.info(cli)
    experiment_dir = os.path.join('experiments', cli.experiment)

    if os.path.exists(experiment_dir):
        raise ValueError('Such experiment name already exists')

    logdir = os.path.join(experiment_dir, 'log')
    checkpoint_dir = os.path.join(experiment_dir, 'checkpoint')

    os.makedirs(logdir)
    os.makedirs(checkpoint_dir)

    with open(os.path.join(logdir, 'args.yaml'), 'w') as f:
        yaml.dump(vars(cli), f)

    logging.info('Loading train')
    train_data = pd.read_csv(PATH_TO_TRAIN, dtype={ITEM_KEY: np.int64})
    logging.info('Train shape: {}'.format(train_data.shape))

    logging.info('Loading val')
    val_data = pd.read_csv(PATH_TO_VAL, dtype={ITEM_KEY: np.int64})
    logging.info('Val shape: {}'.format(val_data.shape))

    item_ids = sorted(set(train_data[ITEM_KEY].unique().tolist() + val_data[ITEM_KEY].unique().tolist()))
    item_id2idx_map = pd.Series(np.arange(len(item_ids)), index=item_ids)
    logging.info('N items: {}'.format(len(item_ids)))

    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True

    with tf.Session(config=gpu_config) as sess:
        if cli.debug:
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)

        gru = model.GRU4Rec(sess, session_key='session_id',
                            item_key='item_id',
                            item_idx_key='item_idx',
                            time_key='timestamp',
                            layers=cli.layers,
                            rnn_size=cli.rnn_size,
                            epochs=cli.epochs,
                            batch_size=cli.batch_size,
                            item_id2idx_map=item_id2idx_map,
                            lr=cli.lr,
                            optimizer=cli.optimizer,
                            decay_steps=20000,
                            summary_steps=1000,
                            decay=0.96,
                            sigma=0,
                            dropout=cli.dropout,
                            grad_cap=cli.grad_cap,
                            log_dir=logdir,
                            checkpoint_dir=checkpoint_dir,
                            loss=cli.loss,
                            hidden_act=cli.hidden_act,
                            final_act=cli.final_act,
                            init_as_normal=False,
                            momentum=cli.momentum,
                            restore_from_model=cli.restore_model,
                            monitor_grads=False)
        if cli.monitor_gpu:
            monitor_gpu(logdir)

        if cli.evaluate:
            logging.info('Evaluating')

            logging.info('Loading test')
            test_data = pd.read_csv(PATH_TO_TEST, dtype={ITEM_KEY: np.int64})
            logging.info('Test shape: {}'.format(test_data.shape))

            res = evaluation.evaluate_sessions_batch(gru, test_data, SESSION_KEY, ITEM_KEY)
            logging.info('Recall@20: {}\tMRR@20: {}'.format(res[0], res[1]))
        else:
            logging.info('Training')
            gru.fit(train_data, val_data)

