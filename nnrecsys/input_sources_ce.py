import numpy as np
import ray
from hyperopt import hp
from ray.tune import run_experiments
from ray.tune.schedulers import AsyncHyperBandScheduler, HyperBandScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch
from hyperopt.pyll.base import scope

from nnrecsys.amazon_emb import fit, get_run_params
from nnrecsys.data.amazon.constants import DATASET_NAME

EXPERIMENT = 'input_sources_ce_' + DATASET_NAME

run_params = get_run_params()
n_samples = run_params['n_samples']


def trainable(config, reporter):
    sub_config = {
        'identity': config['identity_embeddings'],
        'image': config['image_embeddings'],
        'text': config['text_embeddings']
    }
    experiment = ':'.join('{}={}'.format(k, v) for k, v in sub_config.items())

    fit({'experiment': experiment,
         'experiment_folder': EXPERIMENT,
         **config}, reporter=reporter, train_steps=10000)


def main():
    ray.init(num_cpus=16, num_gpus=1)

    space = {
        'batch_size': 32,
        'lr': 0.000234,
        'dropout': 0.85,
        'reg_weight': 0,
        'n_negatives': -1,
        'filter_positives': True,
        'grad_norm': 0.0,

        'text_embeddings': ray.tune.grid_search([True, False]),
        'text_embeddings_type': 'elmo',
        'image_embeddings': ray.tune.grid_search([True, False]),
        'identity_embeddings': ray.tune.grid_search([True, False]),
        'review_embeddings': False,

        'content_embedding_dim': 128,
        'rnn_layers': 1,
        'rnn_units': 128,

        'loss': 'crossentropy',
        'scores': 'multiplicative',

        'k': 20,
        'identity_embedding_dim': 512
    }

    space['eval_batch_size'] = space['batch_size']

    config = {
        EXPERIMENT: {
            "config": space,
            "run": trainable,
            "stop": {"auc": 1},
            "num_samples": 1,
            "local_dir": "~/ray_results",
            "resources_per_trial": {
                'gpu': 0.25,
            }
        }
    }

    ray.tune.run_experiments(config, raise_on_failed_trial=False, resume=True)


if __name__ == '__main__':
    print(DATASET_NAME)
    main()
