import tensorflow as tf

from nnrecsys.models.embrec import input_layer, get_item_embeddings, enrich_input, recurrence, crossentropy_loss, \
    cosine_loss, bpr_max_loss, bpr_loss


def model_fn(features, labels, mode, params):
    run_params = params['run_params']
    embedding_sources, scaffold = input_layer(params)
    input_ids = features['identity']
    sequence_length = features['seq_length']

    input_item_embeddings = get_item_embeddings(embedding_sources, params['content_embedding_dim'], input_ids)

    enriched_input = enrich_input(input_item_embeddings, features, params)

    rnn_outputs = recurrence(enriched_input, sequence_length, params)  # [batch_size, seq_len, rnn_units]

    embedding_size = params['content_embedding_dim'] * len(embedding_sources)
    predicted_embeddings = tf.layers.dense(rnn_outputs,
                                           embedding_size,
                                           activation=None)  # [batch_size, seq_len, emb_size]

    catalog_ids = tf.range(0, run_params['catalog_size'], dtype=tf.int32)
    catalog_embeddings = get_item_embeddings(embedding_sources,
                                             params['content_embedding_dim'],
                                             catalog_ids)  # [catalog_size, emb_size]

    catalog_scores = tf.tensordot(predicted_embeddings, catalog_embeddings, axes=[[-1], [-1]])
    tf.summary.histogram('catalog_scores', catalog_scores)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predicted_scores, predicted_items = tf.nn.top_k(catalog_scores,
                                                        k=params['k'],
                                                        sorted=True,
                                                        name='top_k')
        predictions = {
            'scores': catalog_scores,
            'item_ids': predicted_items
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    positive_embeddings = get_item_embeddings(embedding_sources,
                                              params['content_embedding_dim'],
                                              labels['identity'])  # [batch_size, seq_len, emb_size]
    negative_embeddings = get_item_embeddings(embedding_sources,
                                              params['content_embedding_dim'],
                                              labels['negatives'])  # [n_negatives, emb_size]
    positive_scores = tf.reduce_sum(tf.multiply(predicted_embeddings, positive_embeddings), axis=-1)
    negative_scores = tf.tensordot(predicted_embeddings, negative_embeddings, axes=[[-1], [-1]])
    # tf.summary.histogram('positive_scores', positive_scores)
    # tf.summary.histogram('negative_scores', negative_scores)

    if params['loss'] == 'bpr':
        loss = bpr_loss(positive_scores, negative_scores, sequence_length)
    elif params['loss'] == 'bpr_max':
        loss = bpr_max_loss(positive_scores, negative_scores, sequence_length, params['reg_weight'])
    elif params['loss'] == 'cosine':
        loss = cosine_loss(predicted_embeddings, positive_embeddings, sequence_length)
    elif params['loss'] == 'crossentropy':
        loss = crossentropy_loss(positive_scores, negative_scores, sequence_length)
    else:
        raise ValueError('Unknown loss', params['loss'])
    loss = tf.identity(loss, name="loss")

    metrics = get_metrics(catalog_scores, labels['identity'], sequence_length,
                          params['k'], params['run_params']['catalog_size'])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    optimizer = tf.train.AdamOptimizer(learning_rate=params['lr'])
    grad_vars = optimizer.compute_gradients(loss=loss)

    gradients, variables = zip(*grad_vars)

    if params['grad_norm']:
        gradients, _ = tf.clip_by_global_norm(gradients, params['grad_norm'])

    with tf.name_scope('optimizer'):
        for gradient, variable in grad_vars:
            tf.summary.histogram("gradients/" + variable.name, tf.norm(gradient))
            tf.summary.histogram("variables/" + variable.name, tf.norm(variable))

    train_op = optimizer.apply_gradients(zip(gradients, variables), global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, scaffold=scaffold)


def get_metrics(scores, positives, sequence_length, k, catalog_size):
    with tf.name_scope('metrics'):
        sequence_mask = tf.sequence_mask(sequence_length)
        hitrate = tf.metrics.recall_at_k(tf.cast(positives, tf.int64),
                                         scores,
                                         weights=sequence_mask,
                                         name='hitrate_at_50',
                                         k=50)
        tf.summary.scalar('hitrate_at_50', hitrate[1])

        metrics = {'hitrate_at_50': hitrate}
    return metrics
