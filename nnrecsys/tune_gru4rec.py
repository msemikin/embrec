import numpy as np
import ray
from hyperopt import hp
from ray.tune import run_experiments
from ray.tune.schedulers import AsyncHyperBandScheduler, HyperBandScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch
from hyperopt.pyll.base import scope

from nnrecsys.amazon_emb import fit, get_run_params
from nnrecsys.data.amazon.constants import DATASET_NAME

EXPERIMENT = 'tune_gru4rec_' + DATASET_NAME

run_params = get_run_params()
n_samples = run_params['n_samples']


class NNRecSys(ray.tune.Trainable):
    def _setup(self, config):
        sub_config = {
            'lr': round(config['lr'], 6),
            'dropout': round(config['dropout'], 6),
            'reg_weight': round(config['reg_weight'], 6),
            'input_fc': config['input_fc'],
            'identity_embedding_dim': config['identity_embedding_dim']
        }
        self.experiment = ':'.join('{}={}'.format(k, v) for k, v in sub_config.items())
        self.iteration = 0

    def _train(self):
        eval_result = fit({'experiment': self.experiment,
                           'experiment_folder': EXPERIMENT,
                           **self.config},
                          train_steps=(n_samples / self.config['batch_size']))
        iteration = self.iteration
        self.iteration += 1
        return {**eval_result, 'iteration': iteration, 'neg_loss': -eval_result['loss']}

    def _save(self, checkpoint_dir):
        pass

    def _restore(self, checkpoint):
        pass


def main():
    ray.init(num_cpus=16, num_gpus=1)

    space = {
        'batch_size': 32,
        'lr': hp.loguniform('learning_rate', np.log(1e-5), np.log(1e-2)),
        'dropout': hp.uniform('dropout', 0.5, 1),
        'reg_weight': hp.choice('r_weight', [0.0, hp.loguniform('reg_weight', np.log(1e-7), np.log(1e-4))]),
        'n_negatives': -1,
        'filter_positives': True,
        'grad_norm': 0.0,

        'text_embeddings': False,
        'text_embeddings_type': 'elmo',
        'image_embeddings': False,
        'identity_embeddings': True,
        'review_embeddings': False,
        'input_fc': hp.choice('input_fc', [True, False]),

        'content_embedding_dim': 128,
        'rnn_layers': 1,
        'rnn_units': 128,

        'loss': 'crossentropy',
        'scores': 'multiplicative',

        'k': 20,
        'identity_embedding_dim': hp.choice('identity_embedding_dim', [128, 256]),

        'checkpoint_steps': None
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
                'gpu': 0.5,
                'cpu': 4
            }
        }
    }

    algo = HyperOptSearch(space, reward_attr="hitrate_at_50")
    scheduler = AsyncHyperBandScheduler(reward_attr="hitrate_at_50", grace_period=2, max_t=30)
    ray.tune.run_experiments(config, search_alg=algo, scheduler=scheduler, raise_on_failed_trial=False, resume=True)


if __name__ == '__main__':
    print(DATASET_NAME)
    main()
