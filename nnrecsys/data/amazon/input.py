import math
import os
from glob import glob

import tensorflow as tf
import pandas as pd
from tqdm import tqdm

NUM_PARALLEL_CALLS = 2
CYCLE_LENGTH = 16
BLOCK_LENGTH = 2


def bucket(buckets, batch_size):
    return tf.data.experimental.bucket_by_sequence_length(
        lambda source, target: source['seq_length'],
        bucket_batch_sizes=[batch_size] * (len(buckets) + 1),
        bucket_boundaries=buckets,
    )


def extract_seq_lengths(events_path):
    for path in tqdm(glob(os.path.join(events_path, '*.tfrecord')), desc='Extracting seq_length'):
        record_iterator = tf.python_io.tf_record_iterator(path=path)
        for string_record in record_iterator:
            example = tf.train.SequenceExample()
            example.ParseFromString(string_record)
            seq_len, = dict(example.context.feature)['seq_length'].int64_list.value
            yield seq_len


def get_buckets(events_path, batch_size):
    seq_lengths = pd.Series(list(extract_seq_lengths(events_path)))
    seq_length_counts = seq_lengths.value_counts().sort_index()
    buckets = []

    running_sum = 0
    for i, count in seq_length_counts.iteritems():
        if running_sum + count >= batch_size:
            buckets.append(i)
            running_sum = 0
        else:
            running_sum += count

    print('\n\n============== Buckets ===============')
    for low, high in zip([0] + buckets, buckets + [float('inf')]):
        bucket_count = seq_length_counts[(seq_length_counts.index >= low) &
                                         (seq_length_counts.index < high)].sum()
        print('[{}; {}): {}'.format(low, high, bucket_count))
    print('======================================\n\n')
    return buckets


def get_input_dataset(events_path, include_reviews):
    context_description = {
        'seq_length': tf.io.FixedLenFeature([1], tf.int64)
    }

    feature_description = {
        'identity': tf.io.FixedLenSequenceFeature([1], tf.int64),
    }

    if include_reviews:
        feature_description['review'] = tf.io.FixedLenSequenceFeature([1], tf.string)

    def _parse_seq(example_proto):
        context, sequence = tf.io.parse_single_sequence_example(example_proto,
                                                                context_features=context_description,
                                                                sequence_features=feature_description)
        identity = tf.reshape(sequence['identity'], [-1])
        seq_length = context['seq_length'][0]
        seq = ({'identity': tf.cast(identity[:-1], tf.int32),
                'seq_length': tf.cast(seq_length, tf.int32)},
               {'identity': tf.cast(identity[1:], tf.int32)})

        if include_reviews:
            review = sequence['review']
            seq[0]['review'] = review[:-1]
        return seq

    dataset = (tf.data.Dataset.list_files(os.path.join(events_path, '*.tfrecord'), seed=42)
               .interleave(tf.data.TFRecordDataset,
                           cycle_length=CYCLE_LENGTH,
                           block_length=BLOCK_LENGTH,
                           num_parallel_calls=NUM_PARALLEL_CALLS)
               .map(_parse_seq, num_parallel_calls=NUM_PARALLEL_CALLS))

    return dataset


def input_fn(events_path, batch_size, include_reviews, catalog_size,
             n_negatives=-1, item_probs=None, filter_positives=True, include_item_probs=False):

    item_probs = tf.constant(item_probs, dtype=tf.float32)

    buckets = get_buckets(events_path, batch_size)
    dataset = (get_input_dataset(events_path, include_reviews)
               .shuffle(buffer_size=10000)
               .apply(bucket(buckets, batch_size)))

    if n_negatives == -1:
        dataset = dataset.map(get_all_negatives(catalog_size, filter_positives),
                              num_parallel_calls=NUM_PARALLEL_CALLS * 2)
    elif n_negatives is not None:
        dataset = dataset.map(sample_negatives(item_probs, n_negatives, filter_positives),
                              num_parallel_calls=NUM_PARALLEL_CALLS * 2)
    if include_item_probs:
        dataset = dataset.map(lambda features, labels: ({**features, 'item_probs': item_probs}, labels))

    return dataset.cache().prefetch(100)


def predict_input_fn(events_path, batch_size):
    buckets = get_buckets(events_path, batch_size)
    dataset = get_input_dataset(events_path, False).apply(bucket(buckets, batch_size))
    return dataset


def sample_negatives(probs, n_negatives, filter_positives):
    def transform(features, labels):
        if filter_positives:
            positives = tf.reshape(labels['identity'], shape=[-1])  # [batch_size * seq_len]
            catalog_size = tf.shape(probs)[0]
            positive_one_hot = tf.one_hot(positives, on_value=0., off_value=1., depth=catalog_size, dtype=tf.float32)
            positive_many_hot = tf.reduce_prod(positive_one_hot, axis=0)
            neg_probs = probs * positive_many_hot
        else:
            neg_probs = probs

        log_probs = tf.log(neg_probs)
        negatives = tf.random.multinomial([log_probs], num_samples=n_negatives, output_dtype=tf.int32)
        negatives = tf.reshape(negatives, shape=[-1])

        return features, {**labels, 'negatives': negatives}

    return transform


def get_all_negatives(catalog_size, filter_positives):
    def transform(features, labels):
        if not filter_positives:
            return features, {'negatives': None}
        positives = tf.reshape(labels['identity'], shape=[-1])
        positive_zero_hot = tf.one_hot(positives, on_value=0, off_value=1, depth=catalog_size, dtype=tf.int32)
        positive_zero_hot = tf.reduce_prod(positive_zero_hot, axis=0)
        negatives = tf.where(tf.cast(positive_zero_hot, tf.bool))
        negatives = tf.reshape(negatives, shape=[-1])
        return features, {**labels, 'negatives': negatives}

    return transform
