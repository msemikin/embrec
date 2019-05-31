#!/usr/bin/env python
# coding: utf-8
import argparse

import tensorflow as tf
import logging
import os

from nnrecsys.data.yoochoose.input import get_feature_columns, train_input_fn
from nnrecsys.data.yoochoose import constants
from nnrecsys.models.rnn import model_fn
from nnrecsys.training.hooks import ValidationMetricHook
from nnrecsys.utils import file_len

logging.getLogger().setLevel(logging.INFO)


dir_path = os.path.dirname(os.path.realpath(__file__))


def get_estimator(config):
    n_items = file_len(constants.VOCABULARY_FILE)
    return tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=os.path.join(dir_path, '../experiments', config['experiment']),
        params={
            'feature_columns': get_feature_columns(config),
            'k': 20,
            'n_items': n_items,
            **config
        })


def fit(config: dict, reporter=None) -> float:
    estimator = get_estimator(config)

    os.makedirs(estimator.eval_dir(), exist_ok=True)
    
    early_stopping = tf.contrib.estimator.stop_if_no_decrease_hook(
        estimator,
        metric_name='loss',
        max_steps_without_decrease=10_000_000 / config['batch_size'],
        min_steps=2_000_000 / config['batch_size'])

    eval_hooks = [ValidationMetricHook(
        estimator,
        lambda global_step, metrics: reporter(recall_at_k=metrics['recall_at_k'],
                                              neg_loss=-metrics['loss']))] if reporter else []

    eval_result, export_result = tf.estimator.train_and_evaluate(
        estimator,
        train_spec=tf.estimator.TrainSpec(lambda: train_input_fn(constants.TRAIN_PATH,
                                                                 constants.VOCABULARY_FILE,
                                                                 config['batch_size']).repeat(),
                                          hooks=[early_stopping]),
        eval_spec=tf.estimator.EvalSpec(lambda: train_input_fn(constants.VAL_PATH,
                                                               constants.VOCABULARY_FILE,
                                                               config['batch_size']),
                                        hooks=eval_hooks))

    return eval_result


def evaluate(config):
    estimator = get_estimator(config)
    estimator.evaluate(lambda: train_input_fn(constants.TEST_PATH, constants.VOCABULARY_FILE, batch_size=config['batch_size']))


def parse_args():
    parser = argparse.ArgumentParser(description='GRU4Rec args')
    parser.add_argument('--experiment', required=True, type=str)
    parser.add_argument('--evaluate', type=str)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--rnn_layers', default=1, type=int)
    parser.add_argument('--rnn_units', default=100, type=int)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--hidden_activation', default='relu', type=str)
    parser.add_argument('--dropout', default=1, type=float)

    return parser.parse_args()


def main():
    cli = parse_args()

    config = {'experiment': cli.experiment,
              'batch_size': cli.batch_size,
              'rnn_layers': cli.rnn_layers,
              'rnn_units': cli.rnn_units,
              'lr': cli.lr,
              'hidden_activation': cli.hidden_activation,
              'dropout': cli.dropout}

    if cli.evaluate:
        logging.info('Evaluating')
        evaluate(config)
    else:
        logging.info('Fitting')
        fit(config)


if __name__ == '__main__':
    main()
