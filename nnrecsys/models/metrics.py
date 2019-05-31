import tensorflow_ranking as tfr
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import metrics


def mean_reciprocal_rank(labels, predictions, weights=None, topn=None, name=None):
    """Computes mean reciprocal rank (MRR).

    Args:
      labels: A `Tensor` of the same shape as `predictions`. A value >= 1 means a
        relevant example.
      predictions: A `Tensor` with shape [batch_size, list_size]. Each value is
        the ranking score of the corresponding example.
      weights: A `Tensor` of the same shape of predictions or [batch_size, 1]. The
        former case is per-example and the latter case is per-list.
      name: A string used as the name for this metric.

    Returns:
      A metric for the weighted mean reciprocal rank of the batch.
    """
    with ops.name_scope(name, 'mean_reciprocal_rank',
                        (labels, predictions, weights)):
        _, list_size = array_ops.unstack(array_ops.shape(predictions))
        labels, predictions, weights, topn = tfr.metrics._prepare_and_validate_params(
            labels, predictions, weights, topn)
        sorted_labels, = tfr.utils.sort_by_scores(predictions, [labels], topn=topn)
        # Relevance = 1.0 when labels >= 1.0 to accommodate graded relevance.
        relevance = math_ops.to_float(math_ops.greater_equal(sorted_labels, 1.0))
        reciprocal_rank = 1.0 / math_ops.to_float(math_ops.range(1, topn + 1))
        # MRR has a shape of [batch_size, 1]
        mrr = math_ops.reduce_max(
            relevance * reciprocal_rank, axis=1, keepdims=True)
        return metrics.mean(mrr * array_ops.ones_like(weights), weights)
