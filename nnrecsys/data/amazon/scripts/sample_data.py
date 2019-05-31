import os
import tensorflow as tf
import shutil

import pandas as pd
from nnrecsys.data.amazon import constants, parquet
from nnrecsys.data.amazon.scripts.tfrecords import _group_by_user, _make_sequence_example

SAMPLE_SIZE = 100


def main():
    if os.path.exists(constants.SAMPLE_REVIEWS_PATH):
        os.remove(constants.SAMPLE_REVIEWS_PATH)
        print('Removed', constants.SAMPLE_REVIEWS_PATH)

    train_reviews = pd.read_parquet(constants.TRAIN_REVIEWS_PATH)
    sample_users = train_reviews.reviewerID.drop_duplicates().sample(SAMPLE_SIZE, random_state=42).values
    sample_reviews = train_reviews[train_reviews.reviewerID.isin(sample_users)]

    print('Sample: items={}, users={}, reviews={}'.format(len(sample_reviews.asin.unique()),
                                                          len(sample_reviews.reviewerID.unique()),
                                                          len(sample_reviews)))

    parquet.save_parquet(constants.SAMPLE_REVIEWS_PATH, sample_reviews)

    metadata = pd.read_parquet(constants.CLEAN_METADATA_PATH)

    if os.path.exists(constants.SAMPLE_PATH):
        shutil.rmtree(constants.SAMPLE_PATH)
        print('Removed', constants.SAMPLE_PATH)
    os.mkdir(constants.SAMPLE_PATH)

    id2idx = {item: idx for idx, item in enumerate(metadata.asin)}
    for i, group in enumerate(_group_by_user(sample_reviews, 'Sample users')):
        fname = os.path.join(constants.SAMPLE_PATH, '{}.tfrecord'.format(i))
        with tf.python_io.TFRecordWriter(fname) as train_writer:
            for user_id, reviews in group:
                if user_id is None:
                    break
                sequence_example = _make_sequence_example(reviews, id2idx)
                train_writer.write(sequence_example.SerializeToString())


if __name__ == '__main__':
    main()
