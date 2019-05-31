import time

import tensorflow as tf
import pandas as pd
from nnrecsys.data.amazon.input import input_fn
from nnrecsys.data.amazon import constants
from nnrecsys.amazon_emb import get_item_probs, get_estimator, get_run_params


catalog = pd.read_parquet(constants.CLEAN_METADATA_PATH)
item_probs = get_item_probs(catalog, constants.TRAIN_PATH)
ds = input_fn(constants.TRAIN_PATH, 128, False, 1024, item_probs)
next_batch = ds.make_one_shot_iterator().get_next()

space = {
    'experiment': 'test',
    'batch_size': 128,
    'eval_batch_size': 128,
    'lr': 0.1,
    'dropout': 1,
    'n_negatives': 1024,

    'text_embeddings': True,
    'text_embeddings_type': 'elmo',
    'image_embeddings': False,
    'identity_embeddings': False,
    'review_embeddings': False,

    'content_embedding_dim': 128,
    'rnn_layers': 1,
    'rnn_units': 128,

    'loss': 'crossentropy',
    'scores': 'additive',

    'k': 20,
    'identity_embedding_dim': 512
}

estimator = get_estimator(space, get_run_params(), profile=True)
model_spec = estimator.model_fn(next_batch[0], next_batch[1], tf.estimator.ModeKeys.TRAIN, estimator.params)

builder = tf.profiler.ProfileOptionBuilder
opts = builder(builder.time_and_memory()).order_by('micros').build()

with tf.contrib.tfprof.ProfileContext('/tmp/train_dir',
                                      trace_steps=[],
                                      dump_steps=[]) as pctx:
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        model_spec.scaffold.init_fn(sess)
        for i in range(200):
            t0 = time.time()
            print(i)
            pctx.trace_next_step()
            pctx.dump_next_step()
            _ = sess.run(model_spec.train_op)
            pctx.profiler.profile_operations(options=opts)
            print('Time elapsed: {} seconds'.format(time.time() - t0))

