from nnrecsys.models.metrics import mean_reciprocal_rank
import tensorflow as tf


def model_fn(features, labels, mode, params):
    print(features)
    input_layer, sequence_length = tf.contrib.feature_column.sequence_input_layer(features, params['feature_columns'])

    with tf.name_scope('encoder'):
        def rnn_cell():
            with tf.name_scope('recurrent_layer'):
                cell = tf.nn.rnn_cell.GRUCell(params['rnn_units'], activation=params['hidden_activation'])
                drop_cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=params['dropout'])
            return drop_cell

        stacked_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell() for _ in range(params['rnn_layers'])])

        x, states = tf.nn.dynamic_rnn(stacked_cell,
                                      inputs=input_layer,
                                      dtype=tf.float32,
                                      sequence_length=sequence_length)

        tf.summary.histogram('rnn_outputs', x)
        tf.summary.histogram('rnn_state', states)

        for variable in stacked_cell.variables:
            tf.summary.histogram('gru_vars/' + variable.name, variable)

    logits = tf.layers.dense(x, params['n_items'], activation=None)

    if mode == tf.estimator.ModeKeys.PREDICT:
        scores, predicted_items = tf.nn.top_k(logits,
                                              k=params['k'],
                                              sorted=True,
                                              name='top_k')
        predictions = {
            'scores': scores,
            'item_ids': predicted_items,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    padding_mask = tf.sequence_mask(sequence_length, dtype=tf.float32)
    loss = tf.contrib.seq2seq.sequence_loss(logits, labels, weights=padding_mask, name='seq_loss')

    recall_at_k = tf.metrics.recall_at_k(labels, logits, name='recall_at_k', k=params['k'])

    reshaped_logits = tf.reshape(logits, (-1, logits.shape[-1]))
    reshaped_labels = tf.reshape(labels, (-1,))
    one_hot_labels = tf.one_hot(reshaped_labels, depth=logits.shape[-1])

    mrr = mean_reciprocal_rank(one_hot_labels, reshaped_logits, topn=params['k'], name='mrr_at_k')
    metrics = {'recall_at_k': recall_at_k, 'mrr': mrr}

    tf.summary.scalar('recall_at_k', recall_at_k[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
