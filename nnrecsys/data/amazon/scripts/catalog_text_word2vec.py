from itertools import chain

from nnrecsys.data.amazon import constants
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import math
from tqdm import tqdm

from nnrecsys.data.amazon.parquet import load_parquet


def compute_word2vec():
    metadata = load_parquet(constants.CLEAN_METADATA_PATH)
    texts = metadata.texts
    batch_size = 128

    g = tf.Graph()
    with g.as_default():
        module_url = "https://tfhub.dev/google/Wiki-words-250/1"
        embed = hub.Module(module_url)

        text_input = tf.placeholder(dtype=tf.string, shape=[None])
        embedding = embed(text_input)
        init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
    g.finalize()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(graph=g, config=config)
    session.run(init_op)

    results = []
    for i in tqdm(range(math.ceil(len(texts) / batch_size))):
        chunk = texts.iloc[(i * batch_size):((i + 1) * batch_size)]

        embeddings = session.run(embedding, feed_dict={text_input: chunk.values})
        results.append(embeddings)

    result = np.array(list(chain.from_iterable(results)))
    print(result.shape)
    np.save(constants.CATALOG_TEXT_WORD2VEC_EMBEDDINGS_PATH, result)


if __name__ == '__main__':
    compute_word2vec()

