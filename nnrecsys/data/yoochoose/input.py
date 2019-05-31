import tensorflow as tf

from nnrecsys.utils import file_len
from nnrecsys.data.yoochoose import constants


def train_input_fn(train_path, vocabulary_path, batch_size):
    buckets = list(range(1, 21)) + list(range(25, 40, 5))

    def bucket():
        return tf.data.experimental.bucket_by_sequence_length(
            lambda x: tf.shape(x)[0],
            bucket_batch_sizes=[batch_size] * (len(buckets) + 1),
            bucket_boundaries=buckets,
        )

    table = tf.contrib.lookup.index_table_from_file(vocabulary_path)

    sources = (tf.data.TextLineDataset(train_path)
               .map(lambda string: tf.string_split([string]).values[:-1])
               .apply(bucket())
               .map(lambda items: {'items': items}))

    targets = (tf.data.TextLineDataset(train_path)
               .map(lambda string: tf.string_split([string]).values[1:])
               .map(table.lookup)
               .apply(bucket()))

    return (tf.data.Dataset.zip((sources, targets))
            .shuffle(buffer_size=10000))


def get_feature_columns(config):
    n_items = file_len(constants.VOCABULARY_FILE)

    items = tf.contrib.feature_column.sequence_categorical_column_with_vocabulary_file(
        key='items', vocabulary_file=config.VOCABULARY_FILE, vocabulary_size=n_items)

    items_embedding = tf.feature_column.embedding_column(items, config['rnn_units'])
    feature_columns = [items_embedding]
    return feature_columns
