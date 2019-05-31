import argparse
import copy
import json
import os
from glob import glob

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorboard.plugins.projector import ProjectorConfig, visualize_embeddings
from tensorflow.python.training.summary_io import SummaryWriter
from tensorflow.python import debug as tf_debug
from tqdm import tqdm

from nnrecsys.data.amazon import constants
from nnrecsys.data.amazon.input import input_fn, predict_input_fn
from nnrecsys.data.amazon.parquet import load_parquet
from nnrecsys.models.embrec import model_fn
from nnrecsys.training.hooks import ValidationMetricHook

tf.logging.set_verbosity(tf.logging.INFO)


def get_estimator(config, run_params, model_dir, profile=False):
    params = copy.copy(config)

    if config['text_embeddings']:
        if config['text_embeddings_type'] == 'elmo':
            if config['pca']:
                text_embeddings_path = constants.CATALOG_TEXT_PCA_EMBEDDINGS_PATH
            else:
                text_embeddings_path = constants.CATALOG_TEXT_EMBEDDINGS_PATH
        elif config['text_embeddings_type'] == 'tfidf':
            if config['pca']:
                text_embeddings_path = constants.CATALOG_TEXT_TFIDF_PCA_EMBEDDINGS_PATH
            else:
                text_embeddings_path = constants.CATALOG_TEXT_TFIDF_EMBEDDINGS_PATH
        elif config['text_embeddings_type'] == 'word2vec':
            text_embeddings_path = constants.CATALOG_TEXT_WORD2VEC_EMBEDDINGS_PATH
        elif config['text_embeddings_type'] == 'img_text':
            if config['pca']:
                text_embeddings_path = constants.IMG_TEXT_PCA_EMBEDDINGS_PATH
            else:
                raise ValueError('Cannot use img_text without pca')
        else:
            raise ValueError('Unknown type of text embeddings', config['text_embeddings_type'])

        text_embeddings = np.load(text_embeddings_path)
        print('❗ Loaded text embeddings from {} of shape: {}'.format(
            text_embeddings_path,
            text_embeddings.shape
        ))
        params['text_embeddings'] = text_embeddings
        params['text_embedding_dim'] = text_embeddings.shape[1]
    if config['image_embeddings']:
        image_embeddings = np.load(constants.IMAGE_PCA_EMBEDDINGS_PATH if config['pca'] else
                                   constants.IMAGE_EMBEDDINGS_PATH)
        print('❗ Loaded image embeddings from {} of shape: {}'.format(
            constants.IMAGE_EMBEDDINGS_PATH,
            image_embeddings.shape
        ))
        params['image_embeddings'] = image_embeddings
        params['image_embedding_dim'] = image_embeddings.shape[1]

    params['run_params'] = run_params

    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True

    checkpoint_steps = (run_params['n_samples'] / config['batch_size'] if 'checkpoint_steps' not in config or config['checkpoint_steps'] is None
                        else config['checkpoint_steps'])
    estimator_config = (tf.estimator.RunConfig() if profile
                        else tf.estimator.RunConfig(
        save_checkpoints_steps=checkpoint_steps,
        session_config=session_config))

    estimator = tf.estimator.Estimator(
        config=estimator_config,
        model_fn=model_fn,
        model_dir=model_dir,
        params=params)

    os.makedirs(estimator.eval_dir(), exist_ok=True)
    return estimator


def get_run_params():
    catalog = load_parquet(constants.CLEAN_METADATA_PATH)
    train_reviews = load_parquet(constants.TRAIN_REVIEWS_PATH)

    run_params = {
        'n_samples': train_reviews.reviewerID.nunique(),
        'catalog_size': len(catalog),
        'item_probs': get_item_probs(catalog, events_path=constants.TRAIN_PATH),
    }

    return run_params


def init_embedding_viz(estimator, config):
    projector_config = ProjectorConfig()

    cwd = os.getcwd()

    tensor_names = []
    if config['image_embeddings']:
        tensor_names.append('image_embeddings')
    if config['text_embeddings']:
        tensor_names.append('text_embeddings')
    if config['identity_embeddings']:
        tensor_names.append('identity_embeddings')

    for tensor_name in tensor_names:
        embedding = projector_config.embeddings.add()
        embedding.tensor_name = tensor_name
        embedding.sprite.image_path = os.path.join(cwd, constants.IMAGE_SPRITE_FILE)
        embedding.sprite.single_image_dim.extend([32, 32])
        embedding.metadata_path = os.path.join(cwd, constants.EMBEDDING_METADATA_TSV)

    summary_writer = SummaryWriter(estimator.model_dir)
    visualize_embeddings(summary_writer, projector_config)


def extract_reviewed_items(events_path):
    for path in tqdm(glob(os.path.join(events_path, '*.tfrecord')), desc='Extracting reviewed items'):
        record_iterator = tf.python_io.tf_record_iterator(path=path)
        for string_record in record_iterator:
            example = tf.train.SequenceExample()
            example.ParseFromString(string_record)

            for value in dict(example.feature_lists.feature_list)['identity'].feature:
                item_idx, = list(value.int64_list.value)
                yield item_idx


def get_item_probs(catalog, events_path):
    reviewed_items = pd.Series(list(extract_reviewed_items(events_path)))
    item_probs = reviewed_items.value_counts() / len(reviewed_items)
    item_probs = item_probs.reindex(np.arange(len(catalog)), fill_value=0).values
    assert len(item_probs) == len(catalog)
    return item_probs


def get_train_hooks(estimator, run_params, config, debug, early_stop):
    train_hooks = []

    if early_stop:
        print('Using early stopping')
        steps_per_epoch = run_params['n_samples'] / config['batch_size']
        early_stopping = tf.contrib.estimator.stop_if_no_increase_hook(
            estimator,
            metric_name='hitrate_at_50',
            max_steps_without_increase=3 * steps_per_epoch,
            min_steps=3 * steps_per_epoch)
        train_hooks.append(early_stopping)

    # profiler = tf.train.ProfilerHook(save_steps=1, output_dir=estimator.model_dir)

    if debug:
        train_hooks.append(tf_debug.LocalCLIDebugHook())

    return train_hooks


def get_eval_hooks(estimator, reporter):
    eval_hooks = []
    if reporter is not None:
        val_metric_hook = ValidationMetricHook(estimator,
                                               lambda global_step, metrics: reporter(auc=metrics['auc'],
                                                                                     neg_loss=-metrics['loss']))
        eval_hooks.append(val_metric_hook)

    logging_hook = tf.train.LoggingTensorHook({'loss': 'loss',
                                               'global_step': 'global_step'}, every_n_iter=100)
    eval_hooks.append(logging_hook)
    return eval_hooks


def get_model_dir(experiment, experiment_folder=None):
    return os.path.join(constants.ROOT_DIR,
                        'experiments',
                        experiment_folder if experiment_folder else constants.DATASET_NAME,
                        experiment)


def read_config(model_dir):
    with open(os.path.join(model_dir, 'args.json')) as f:
        return json.load(f)


def write_config(model_dir, config):
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, 'args.json'), 'w') as f:
        json.dump(config, f, indent=2, sort_keys=True)


def fit(config: dict, reporter=None, debug=False, train_steps=None, early_stop=False):
    tf.random.set_random_seed(42)
    run_params = get_run_params()

    model_dir = get_model_dir(config['experiment'], config['experiment_folder'])
    if os.path.exists(model_dir):
        config = read_config(model_dir)
        print('Loaded config from', model_dir)
    else:
        write_config(model_dir, config)

    estimator = get_estimator(config, run_params, model_dir)
    init_embedding_viz(estimator, config)

    def train_input_fn():
        return input_fn(constants.TRAIN_PATH,
                        config['batch_size'],
                        config['review_embeddings'],
                        run_params['catalog_size'],
                        config['n_negatives'],
                        run_params['item_probs'],
                        config['filter_positives']).repeat()

    def eval_input_fn():
        return input_fn(constants.VAL_PATH,
                        config['eval_batch_size'],
                        config['review_embeddings'],
                        run_params['catalog_size'],
                        config['n_negatives'],
                        run_params['item_probs'],
                        config['filter_positives'])

    if not early_stop:
        estimator.train(train_input_fn, steps=train_steps)
        eval_result = estimator.evaluate(eval_input_fn, steps=train_steps)
    else:
        train_hooks = get_train_hooks(estimator, run_params, config, debug, early_stop)
        eval_hooks = get_eval_hooks(estimator, reporter)

        eval_result, export_result = tf.estimator.train_and_evaluate(
            estimator,
            train_spec=tf.estimator.TrainSpec(train_input_fn,
                                              hooks=train_hooks,
                                              max_steps=train_steps),
            eval_spec=tf.estimator.EvalSpec(eval_input_fn,
                                            hooks=eval_hooks,
                                            throttle_secs=0,
                                            steps=None))

    return eval_result


def profile(config: dict):
    run_params = get_run_params()
    model_dir = get_model_dir(config['experiment'], config['experiment_folder'])
    with tf.contrib.tfprof.ProfileContext('/tmp/train_dir') as pctx:
        estimator = get_estimator(config, run_params, model_dir, profile=True)
        estimator.train(lambda: input_fn(constants.TRAIN_PATH,
                                         config['batch_size'],
                                         config['review_embeddings'],
                                         run_params['catalog_size'],
                                         config['n_negatives'],
                                         run_params['item_probs'],
                                         config['filter_positives']).repeat(),
                        steps=500)


def evaluate(config: dict):
    run_params = get_run_params()
    model_dir = get_model_dir(config['experiment'], config['experiment_folder'])

    config = read_config(model_dir)
    config['input_fc'] = True
    print('Loaded config from', model_dir)

    estimator = get_estimator(config, run_params, model_dir)
    os.makedirs(estimator.eval_dir(), exist_ok=True)
    result = estimator.evaluate(lambda: input_fn(constants.TEST_PATH,
                                                 config['batch_size'],
                                                 config['review_embeddings'],
                                                 run_params['catalog_size'],
                                                 config['n_negatives'],
                                                 run_params['item_probs'],
                                                 config['filter_positives']),
                                steps=None)
    print(result)


def predict(experiment, experiment_folder, batch_size, events_path):
    run_params = get_run_params()
    model_dir = get_model_dir(experiment, experiment_folder)

    config = read_config(model_dir)
    print('Loaded config from', model_dir)

    estimator = get_estimator(config, run_params, model_dir)
    result = estimator.predict(lambda: predict_input_fn(events_path, batch_size))
    return result


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser(description='GRU4Rec args')
    parser.add_argument('--experiment', required=True, type=str)
    parser.add_argument('--experiment_folder', default=None, type=str)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--eval_batch_size', default=32, type=int)
    parser.add_argument('--checkpoint_steps', default=None, type=int)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--dropout', default=1, type=float)
    parser.add_argument('--n_negatives', default=None, type=int)
    parser.add_argument('--filter_positives', default=True, type=str2bool)
    parser.add_argument('--steps', default=None, type=int)
    parser.add_argument('--reg_weight', default=0, type=float)
    parser.add_argument('--grad_norm', default=0., type=float)

    parser.add_argument('--pca', default=False, type=str2bool)
    parser.add_argument('--text_embeddings', default=True, type=str2bool)
    parser.add_argument('--text_embeddings_type', default='elmo', type=str)
    parser.add_argument('--image_embeddings', default=True, type=str2bool)
    parser.add_argument('--identity_embeddings', default=False, type=str2bool)
    parser.add_argument('--identity_embedding_dim', default=512, type=int)
    parser.add_argument('--review_embeddings', default=False, type=str2bool)
    parser.add_argument('--input_fc', default=True, type=str2bool)

    parser.add_argument('--content_embedding_dim', default=128, type=int)
    parser.add_argument('--rnn_layers', default=1, type=int)
    parser.add_argument('--rnn_units', default=128, type=int)

    parser.add_argument('--loss', default='bpr', type=str)

    parser.add_argument('--evaluate', default=False, type=str2bool)
    parser.add_argument('--debug', default=False, type=str2bool)
    parser.add_argument('--profile', default=False, type=str2bool)
    parser.add_argument('--early_stop', default=True, type=str2bool)

    return parser.parse_args()


def main():
    cli = parse_args()

    config = {'experiment': cli.experiment,
              'experiment_folder': cli.experiment_folder,
              'batch_size': cli.batch_size,
              'eval_batch_size': cli.eval_batch_size,
              'checkpoint_steps': cli.checkpoint_steps,
              'lr': cli.lr,
              'dropout': cli.dropout,
              'n_negatives': cli.n_negatives,
              'filter_positives': cli.filter_positives,
              'reg_weight': cli.reg_weight,
              'grad_norm': cli.grad_norm,

              'pca': cli.pca,
              'text_embeddings': cli.text_embeddings,
              'text_embeddings_type': cli.text_embeddings_type,
              'image_embeddings': cli.image_embeddings,
              'identity_embeddings': cli.identity_embeddings,
              'identity_embedding_dim': cli.identity_embedding_dim,
              'review_embeddings': cli.review_embeddings,
              'input_fc': cli.input_fc,

              'content_embedding_dim': cli.content_embedding_dim,
              'rnn_layers': cli.rnn_layers,
              'rnn_units': cli.rnn_units,

              'loss': cli.loss,

              'k': 20}

    print(config)

    if cli.profile:
        print('----------------------------')
        print('Profiling')
        print('----------------------------')
        profile(config)
    elif cli.evaluate:
        print('----------------------------')
        print('Evaluating')
        print('----------------------------')
        evaluate(config)
    else:
        print('----------------------------')
        print('Fitting')
        print('----------------------------')
        fit(config, debug=cli.debug, train_steps=cli.steps, early_stop=cli.early_stop)


if __name__ == '__main__':
    main()
