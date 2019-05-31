import tensorflow as tf

from nnrecsys.models.embrec import get_metrics


def model_fn(features, labels, mode, params):
    item_probs = features['item_probs']
    sequence_length = features['seq_length']

    input_shape = tf.shape(features['identity'])
    batch_size = input_shape[0]
    seq_len = input_shape[1]
    catalog_scores = tf.broadcast_to(item_probs,
                                     shape=[batch_size, seq_len, tf.shape(item_probs)[-1]])

    metrics = get_metrics(catalog_scores, labels['identity'], sequence_length,
                          params['k'], params['run_params']['catalog_size'])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=tf.constant(0.0, dtype=tf.float32), eval_metric_ops=metrics)
