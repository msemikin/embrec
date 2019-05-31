import numpy as np
import ray
from hyperopt import hp
from ray.tune import run_experiments
from ray.tune.schedulers import AsyncHyperBandScheduler, HyperBandScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch
from hyperopt.pyll.base import scope

from nnrecsys.amazon_emb import fit, get_run_params
from nnrecsys.data.amazon.constants import DATASET_NAME


EXPERIMENT = 'tune_bpr_max_' + DATASET_NAME

run_params = get_run_params()
n_samples = run_params['n_samples']


class NNRecSys(ray.tune.Trainable):
    def _setup(self, config):
        sub_config = {
            'lr': round(config['lr'], 6),
            'dropout': round(config['dropout'], 6),
            'reg_weight': round(config['reg_weight'], 6),
            'n_negatives': config['n_negatives'],
        }
        self.experiment = ':'.join('{}={}'.format(k, v) for k, v in sub_config.items())
        self.iteration = 0

    def _train(self):
        eval_result = fit({'experiment': self.experiment,
                           'experiment_folder': EXPERIMENT,
                           **self.config},
                          train_steps=(n_samples / self.config['batch_size']) / 4)
        iteration = self.iteration
        self.iteration += 1
        return {**eval_result, 'iteration': iteration, 'neg_loss': -eval_result['loss']}

    def _save(self, checkpoint_dir):
        pass

    def _restore(self, checkpoint):
        pass


def main():
    ray.init(num_gpus=1, ignore_reinit_error=True)

    space = {
        'batch_size': 32,
        'lr': hp.loguniform('learning_rate', np.log(1e-6), np.log(1e-1)),
        'dropout': hp.uniform('dropout', 0.5, 1),
        'reg_weight': hp.uniform('reg_weight', 0, 1),
        'n_negatives': hp.choice('n_negatives', [-1, scope.int(hp.quniform('n_negative_samples', 1000, n_samples, 1))]),
        'filter_positives': True,
        'grad_norm': 0.0,

        'text_embeddings': True,
        'text_embeddings_type': 'elmo',
        'image_embeddings': False,
        'identity_embeddings': False,
        'review_embeddings': False,

        'content_embedding_dim': 128,
        'rnn_layers': 1,
        'rnn_units': 128,

        'loss': 'bpr_max',
        'scores': 'multiplicative',

        'k': 20,
        'identity_embedding_dim': 512
    }

    space['eval_batch_size'] = space['batch_size']

    config = {
        EXPERIMENT: {
            "config": space,
            "run": NNRecSys,
            "stop": {"auc": 1},
            "num_samples": 100,
            "local_dir": "~/ray_results",
            "resources_per_trial": {
                'gpu': 0.5
            }
        }
    }

    algo = HyperOptSearch(space, reward_attr="auc")
    scheduler = AsyncHyperBandScheduler(reward_attr="auc", grace_period=3, max_t=40)
    ray.tune.run_experiments(config, search_alg=algo, scheduler=scheduler, raise_on_failed_trial=False)


if __name__ == '__main__':
    main()
