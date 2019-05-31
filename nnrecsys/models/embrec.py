import tensorflow as tf


def model_fn(features, labels, mode, params):
    run_params = params['run_params']
    embedding_sources, scaffold = input_layer(params)
    input_ids = features['identity']
    sequence_length = features['seq_length']

    input_item_embeddings = get_item_embeddings(embedding_sources, params['content_embedding_dim'], input_ids, params['input_fc'])

    enriched_input = enrich_input(input_item_embeddings, features, params)

    rnn_outputs = recurrence(enriched_input, sequence_length, params)  # [batch_size, seq_len, rnn_units]

    embedding_size = params['content_embedding_dim'] * len(embedding_sources)
    predicted_embeddings = tf.layers.dense(rnn_outputs,
                                           embedding_size,
                                           activation=None)  # [batch_size, seq_len, emb_size]

    catalog_ids = tf.range(0, run_params['catalog_size'], dtype=tf.int32)
    catalog_embeddings = get_item_embeddings(embedding_sources,
                                             params['content_embedding_dim'],
                                             catalog_ids, params['input_fc'])  # [catalog_size, emb_size]
    tf.summary.histogram('catalog_embeddings', catalog_embeddings)
    # unit_catalog_embeddings = tf.nn.l2_normalize(catalog_embeddings, axis=-1)
    # catalog_distances = tf.tensordot(unit_catalog_embeddings, unit_catalog_embeddings, axes=[-1, -1])
    # print(catalog_distances.shape)
    # tf.summary.histogram('catalog_distances', catalog_distances)

    catalog_scores = tf.tensordot(predicted_embeddings, catalog_embeddings, axes=[[-1], [-1]])
    tf.summary.histogram('catalog_scores', catalog_scores)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predicted_scores, predicted_items = tf.nn.top_k(catalog_scores,
                                                        k=params['k'],
                                                        sorted=True,
                                                        name='top_k')
        last_predicted_scores = get_last_seq_item(predicted_scores, sequence_length)
        last_predicted_items = get_last_seq_item(predicted_items, sequence_length)
        predictions = {
            'scores': last_predicted_scores,
            'item_ids': last_predicted_items,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    positive_embeddings = get_item_embeddings(embedding_sources,
                                              params['content_embedding_dim'],
                                              labels['identity'],
                                              params['input_fc'])  # [batch_size, seq_len, emb_size]
    positive_scores = tf.reduce_sum(tf.multiply(predicted_embeddings, positive_embeddings), axis=-1)
    tf.summary.histogram('positive_scores', positive_scores)

    if 'negatives' in labels:
        negative_embeddings = get_item_embeddings(embedding_sources,
                                                  params['content_embedding_dim'],
                                                  labels['negatives'])  # [n_negatives, emb_size]
        negative_scores = tf.tensordot(predicted_embeddings, negative_embeddings, axes=[[-1], [-1]])
        tf.summary.histogram('negative_scores', negative_scores)
        if params['loss'] == 'bpr':
            loss = bpr_loss(positive_scores, negative_scores, sequence_length)
        elif params['loss'] == 'bpr_max':
            loss = bpr_max_loss(positive_scores, negative_scores, sequence_length, params['reg_weight'])
        elif params['loss'] == 'crossentropy':
            loss = crossentropy_loss(positive_scores, negative_scores, sequence_length)
        else:
            raise ValueError('Loss not supported with negatives:', params['loss'])
    else:
        if params['loss'] == 'cosine':
            loss = cosine_loss(predicted_embeddings, positive_embeddings, sequence_length)
        else:
            raise ValueError('Loss not supported without negatives', params['loss'])

    loss = tf.identity(loss, name="loss")

    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss += tf.reduce_sum(reg_losses)

    metrics = get_metrics(catalog_scores, labels['identity'], sequence_length,
                          params['k'], params['run_params']['catalog_size'])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=tf.constant(.0, dtype=tf.float32), eval_metric_ops=metrics)

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


# Embeddings


def get_text_embeddings(params):
    text_embeddings = tf.get_variable('text_embeddings',
                                      trainable=False,
                                      shape=[params['run_params']['catalog_size'], params['text_embedding_dim']],
                                      dtype=tf.float32,
                                      initializer=tf.zeros_initializer)
    tf.summary.histogram('text_embeddings', text_embeddings)

    def init_embeddings(_, session):
        session.run(text_embeddings.initializer,
                    {text_embeddings.initial_value: params['text_embeddings']})

    return text_embeddings, init_embeddings


def get_image_embeddings(params):
    image_embeddings = tf.get_variable('image_embeddings',
                                       trainable=False,
                                       shape=[params['run_params']['catalog_size'], params['image_embedding_dim']],
                                       dtype=tf.float32,
                                       initializer=tf.zeros_initializer)
    tf.summary.histogram('image_embeddings', image_embeddings)

    def init_embeddings(_, session):
        session.run(image_embeddings.initializer,
                    {image_embeddings.initial_value: params['image_embeddings']})

    return image_embeddings, init_embeddings


def get_identity_embeddings(params):
    item_embeddings = tf.get_variable('item_embeddings',
                                      trainable=True,
                                      shape=[params['run_params']['catalog_size'], params['identity_embedding_dim']],
                                      dtype=tf.float32,
                                      initializer=tf.random_normal_initializer,
                                      regularizer=tf.contrib.layers.l2_regularizer(scale=params['reg_weight']))
    tf.summary.histogram('item_embeddings', item_embeddings)

    def nop(*_):
        pass

    return item_embeddings, nop


def embed_source(embedding_source, output_size, ids, input_fc):
    x = embedding_source if ids is None else tf.nn.embedding_lookup(embedding_source, ids)
    if input_fc:
        scope_name = embedding_source.name.split(':')[0]
        x = tf.layers.dense(x, output_size, 'relu', reuse=tf.AUTO_REUSE, name=scope_name + '_fc_relu')
        x = tf.layers.dense(x, output_size, reuse=tf.AUTO_REUSE, activation=None, name=scope_name + '_fc')
    return x


def get_item_embeddings(embedding_sources, output_size, ids=None, input_fc=True):
    embeddings = tf.concat([embed_source(emb, output_size, ids, input_fc)
                            for emb in embedding_sources], axis=-1)
    return embeddings


# Layers


def input_layer(params):
    embeddings = []
    if params['text_embeddings'] is not False:
        embeddings.append(get_text_embeddings)
    if params['image_embeddings'] is not False:
        embeddings.append(get_image_embeddings)
    if params['identity_embeddings'] is not False:
        embeddings.append(get_identity_embeddings)

    embedding_sources, init_fns = zip(*(embedding(params) for embedding in embeddings))

    def init_fn(*args):
        for fn in init_fns:
            fn(*args)

    scaffold = tf.train.Scaffold(init_fn=init_fn)
    return embedding_sources, scaffold


def enrich_input(input_item_embeddings, features, params):
    if params['review_embeddings']:
        review_embeddings = features['review']
        review_embeddings = tf.layers.dense(review_embeddings,
                                            params['content_embedding_dim'],
                                            activation='relu',
                                            name='reviews_fc_relu')
        return tf.concat([input_item_embeddings, review_embeddings], axis=-1)
    return input_item_embeddings


def recurrence(source_embeddings, sequence_length, params):
    with tf.name_scope('recurrence'):
        def rnn_cell():
            with tf.name_scope('recurrent_layer'):
                cell = tf.nn.rnn_cell.GRUCell(params['rnn_units'], activation='relu')
                drop_cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=params['dropout'])
            return drop_cell

        stacked_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell() for _ in range(params['rnn_layers'])])

        x, states = tf.nn.dynamic_rnn(stacked_cell,
                                      inputs=source_embeddings,
                                      dtype=tf.float32,
                                      sequence_length=sequence_length,
                                      swap_memory=True)

        tf.summary.histogram('rnn_outputs', x)
        tf.summary.histogram('rnn_state', states)

        for variable in stacked_cell.variables:
            tf.summary.histogram('gru_vars/' + variable.name.replace(':', '_'), variable)
    return x


# Metrics

def ranking_auc(last_item_scores, last_positives, catalog_size):
    with tf.name_scope('ranking_auc'):
        n_negatives = catalog_size - 1
        positive_scores = tf.expand_dims(get_last_seq_item_positive_scores(last_item_scores, last_positives),
                                         axis=1)  # [batch_size, 1]
        correct_ranking = positive_scores > last_item_scores  # [batch_size, catalog_size]
        total_correct_ranking = tf.reduce_sum(tf.cast(correct_ranking, tf.int32), axis=1)  # [batch_size]
        return tf.metrics.mean(total_correct_ranking / n_negatives)


def get_last_seq_item(tensor, seq_length):
    batch_size = tf.shape(seq_length)[0]
    indices = tf.concat([tf.expand_dims(tf.range(batch_size), axis=1),
                         tf.expand_dims(seq_length - 1, axis=1)], axis=1)
    return tf.gather_nd(tensor, indices)


def get_metrics(scores, positives, sequence_length, k, catalog_size):
    with tf.name_scope('metrics'):
        sequence_mask = tf.sequence_mask(sequence_length)
        recall_at_k = tf.metrics.recall_at_k(tf.cast(positives, tf.int64),
                                             scores,
                                             weights=sequence_mask,
                                             name='recall_at_k',
                                             k=k)

        with tf.name_scope('last_scores'):
            last_item_scores = get_last_seq_item(scores, sequence_length)  # [batch_size, catalog_size]
            last_positives = get_last_seq_item(positives, sequence_length)  # [batch_size]

        last_item_hitrate = tf.metrics.recall_at_k(tf.cast(last_positives, tf.int64),
                                                   last_item_scores,
                                                   name='hitrate_at_50',
                                                   k=50)

        auc = ranking_auc(last_item_scores, last_positives, catalog_size)

        tf.summary.scalar('auc', auc[1])
        tf.summary.scalar('hitrate_at_50', last_item_hitrate[1])

        metrics = {'avg_recall_at_k': recall_at_k,
                   'auc': auc,
                   'hitrate_at_50': last_item_hitrate}
    return metrics


def get_last_seq_item_positive_scores(scores, positives):
    batch_size = tf.shape(scores)[0]
    indices = tf.concat([tf.expand_dims(tf.range(batch_size), axis=1),
                         tf.expand_dims(positives, axis=1)], axis=1)
    return tf.gather_nd(scores, indices)


# Losses


def bpr_loss(positive_scores, negative_scores, sequence_length):
    """
    :param positive_scores: shape [batch_size, seq_len]
    :param negative_scores: shape [batch_size, seq_len, n_negatives]
    :param sequence_length: shape [batch_size]
    :return:
    """
    with tf.name_scope('BPR_loss'):
        # [batch_size, seq_len, n_negatives]
        losses = -tf.log_sigmoid(tf.expand_dims(positive_scores, axis=-1) - negative_scores)
        sequence_mask = tf.expand_dims(tf.sequence_mask(sequence_length), axis=-1)
        return tf.losses.compute_weighted_loss(losses, weights=sequence_mask)


def bpr_max_loss(positive_scores, negative_scores, sequence_length, reg_weight):
    """
    :param positive_scores: shape [batch_size, seq_len]
    :param negative_scores: shape [batch_size, seq_len, n_negatives]
    :param sequence_length: shape [batch_size]
    :param reg_weight:
    :return:
    """
    with tf.name_scope('BPR_max_loss'):
        # [batch_size, seq_len, n_negatives]
        softmax_weights = tf.nn.softmax(negative_scores, axis=-1)
        tf.summary.histogram('softmax', softmax_weights)
        sequence_mask = tf.sequence_mask(sequence_length)

        # [batch_size, seq_len, n_negatives]
        correct_ranking_probs = tf.nn.sigmoid(tf.expand_dims(positive_scores, axis=-1) - negative_scores)
        tf.summary.histogram('correct rankings', correct_ranking_probs)

        # [batch_size, seq_len]
        correct_ranking_max_probs = tf.reduce_sum(correct_ranking_probs * softmax_weights, axis=-1)
        tf.summary.histogram('probs', correct_ranking_max_probs)

        losses = -tf.log(correct_ranking_max_probs + 1e-24)

        tf.summary.histogram('losses', losses)

        # [batch_size, seq_len]
        reg = reg_weight * tf.reduce_sum(softmax_weights * tf.square(negative_scores), axis=-1)

        return tf.losses.compute_weighted_loss(losses + reg, weights=sequence_mask)


def get_positive_indices(positives):
    batch_size = tf.shape(positives)[0]
    seq_size = tf.shape(positives)[1]

    seq_ranges = tf.expand_dims(tf.broadcast_to(tf.range(seq_size), shape=(batch_size, seq_size)), axis=-1)
    batch_ranges = tf.expand_dims(tf.expand_dims(tf.range(batch_size), axis=1), axis=1)
    batch_ranges = tf.zeros_like(seq_ranges) + batch_ranges

    element_indices = tf.expand_dims(positives, axis=-1)
    indices = tf.concat([batch_ranges, seq_ranges, element_indices], axis=-1)
    return indices


def get_positive_scores(scores, positives):
    indices = get_positive_indices(positives)
    return tf.gather_nd(scores, indices)


def cosine_loss(predicted_embeddings, positive_embeddings, sequence_length):
    with tf.name_scope('Cosine_loss'):
        unit_predicted_embeddings = tf.nn.l2_normalize(predicted_embeddings, axis=-1)
        unit_target_embeddings = tf.nn.l2_normalize(positive_embeddings, axis=-1)

        sequence_mask = tf.sequence_mask(sequence_length, dtype=tf.float32)  # [batch_size, seq_len]
        reshaped_mask = tf.expand_dims(sequence_mask, axis=-1)  # [batch_size, seq_len, 1]

        return tf.losses.cosine_distance(unit_predicted_embeddings,
                                         unit_target_embeddings,
                                         weights=reshaped_mask,
                                         axis=-1)


def crossentropy_loss(positive_scores, negative_scores, sequence_length):
    """
    :param positive_scores: shape [batch_size, seq_len]
    :param negative_scores: shape [batch_size, seq_len, n_negatives]
    :param sequence_length: shape [batch_size]
    :return:
    """
    with tf.name_scope('Crossentropy_loss'):
        sequence_mask = tf.sequence_mask(sequence_length, dtype=tf.float32)  # [batch_size, seq_len]

        # [batch_size, seq_len, n_negatives + 1]
        scores = tf.concat([tf.expand_dims(positive_scores, axis=-1), negative_scores], axis=-1)

        neg_log_softmax_scores = -tf.nn.log_softmax(scores, axis=-1)
        losses = neg_log_softmax_scores[:, :, 0]  # [batch_size, seq_len]
        return tf.losses.compute_weighted_loss(losses, sequence_mask)
