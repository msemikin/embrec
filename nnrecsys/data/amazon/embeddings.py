import math
import numpy as np
import os
from glob import glob
from itertools import chain
import tensorflow as tf
import tensorflow_hub as hub
from tqdm import tqdm


def extract_elmo_embeddings(start_batch, texts, batch_size):
    g = tf.Graph()
    with g.as_default():
        module_url = "https://tfhub.dev/google/elmo/2"
        embed = hub.Module(module_url)

        text_input = tf.placeholder(dtype=tf.string, shape=[None])
        embedding = embed(text_input)
        init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
    g.finalize()

    session = tf.Session(graph=g)
    session.run(init_op)

    for i in tqdm(range(start_batch, math.ceil(len(texts) / batch_size))):
        chunk = texts.iloc[(i * batch_size):((i + 1) * batch_size)]

        embeddings = session.run(embedding, feed_dict={text_input: chunk.values})
        yield i, embeddings


def gather_batches_from_dir(batches_dir):
    files = glob(os.path.join(batches_dir, '*.npy'))
    batches = []

    for i in tqdm(range(len(files))):
        fname = os.path.join(batches_dir, '{}.npy'.format(i))
        batches.append(np.load(fname))

    result = np.array(list(chain.from_iterable(batches)))
    print('Gathered array of shape', result.shape)
    return result
