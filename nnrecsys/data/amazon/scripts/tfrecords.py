import math
import shutil
from itertools import zip_longest

import tensorflow as tf

from tqdm import tqdm
import numpy as np
import os

from nnrecsys.data.amazon import constants
from nnrecsys.data.amazon.parquet import load_parquet


SAMPLES_PER_SHARD = 1000


def _make_sequence_example(user_reviews, id2idx, review_embeddings=None):
    identity_seq = [id2idx[item] for item in user_reviews.asin]
    identity_features = [tf.train.Feature(int64_list=tf.train.Int64List(value=[identity]))
                         for identity in identity_seq]

    seq_length_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[len(user_reviews) - 1]))

    context = tf.train.Features(feature={
        'seq_length': seq_length_feature,
    })

    feature_list = {
        'identity': tf.train.FeatureList(feature=identity_features),
    }

    if review_embeddings is not None:
        feature_list['review'] = _make_review_feature_list(user_reviews, review_embeddings)

    feature_lists = tf.train.FeatureLists(feature_list=feature_list)
    return tf.train.SequenceExample(context=context, feature_lists=feature_lists)


def _make_review_feature_list(user_reviews, review_embeddings=None):
    review_embeddings = review_embeddings[user_reviews.index]
    review_text_features = [tf.train.Feature(float_list=tf.train.FloatList(value=review_embedding))
                            for review_embedding in review_embeddings]
    return tf.train.FeatureList(feature=review_text_features)


def _group_by_user(df, desc):
    unique_values = len(df.reviewerID.unique())
    return tqdm(grouper(df.groupby('reviewerID'), SAMPLES_PER_SHARD, (None, None)),
                total=math.ceil(unique_values / SAMPLES_PER_SHARD), desc=desc)


def grouper(iterable, n, fillvalue=None):
    """Collect data into fixed-length chunks or blocks"""
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def prepare_tf_records():
    for p in [constants.VAL_PATH, constants.TRAIN_PATH, constants.TEST_PATH]:
        if os.path.exists(p):
            shutil.rmtree(p)
            print('Cleared', p)
        os.makedirs(p, exist_ok=True)

    metadata = load_parquet(constants.CLEAN_METADATA_PATH)
    train_reviews = load_parquet(constants.TRAIN_REVIEWS_PATH)
    test_reviews = load_parquet(constants.TEST_REVIEWS_PATH)

    if constants.INCLUDE_REVIEWS:
        review_embeddings = np.load(constants.REVIEW_EMBEDDINGS_PATH)
        assert len(review_embeddings) == len(test_reviews)
    else:
        review_embeddings = None

    id2idx = {item: idx for idx, item in enumerate(metadata.asin)}
    for i, group in enumerate(_group_by_user(train_reviews, 'Train users')):
        fname = os.path.join(constants.TRAIN_PATH, '{}.tfrecord'.format(i))
        with tf.python_io.TFRecordWriter(fname) as train_writer:
            for user_id, reviews in group:
                if user_id is None:
                    break
                train = reviews.iloc[:-2]
                sequence_example = _make_sequence_example(train, id2idx, review_embeddings)
                train_writer.write(sequence_example.SerializeToString())

    for i, group in enumerate(_group_by_user(test_reviews, 'Test users')):
        val_fname = os.path.join(constants.VAL_PATH, '{}.tfrecord'.format(i))
        test_fname = os.path.join(constants.TEST_PATH, '{}.tfrecord'.format(i))
        with tf.python_io.TFRecordWriter(val_fname) as val_writer, \
                tf.python_io.TFRecordWriter(test_fname) as test_writer:
            for user_id, reviews in group:
                if user_id is None:
                    break
                val = reviews.iloc[:-1]
                val_sequence_example = _make_sequence_example(val, id2idx, review_embeddings)
                val_writer.write(val_sequence_example.SerializeToString())

                test = reviews
                test_sequence_example = _make_sequence_example(test, id2idx, review_embeddings)
                test_writer.write(test_sequence_example.SerializeToString())


if __name__ == '__main__':
    prepare_tf_records()
