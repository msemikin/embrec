import tensorflow as tf

from nnrecsys.amazon_emb import get_run_params
from nnrecsys.data.amazon import constants
from nnrecsys.data.amazon.input import input_fn
from nnrecsys.models.poprec import model_fn


def evaluate():
    run_params = get_run_params()
    batch_size = 128
    params = {
        'k': 20,
        'run_params': run_params
    }

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        params=params)

    result = estimator.evaluate(lambda: input_fn(constants.TEST_PATH,
                                                 batch_size,
                                                 include_reviews=False,
                                                 catalog_size=run_params['catalog_size'],
                                                 n_negatives=None,
                                                 item_probs=run_params['item_probs'],
                                                 include_item_probs=True),
                                steps=None)
    print(result)
    return result


if __name__ == '__main__':
    evaluate()
