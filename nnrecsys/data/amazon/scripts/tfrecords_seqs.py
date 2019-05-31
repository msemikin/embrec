import os
import shutil
import numpy as np
import tensorflow as tf

from nnrecsys.data.amazon import constants
from nnrecsys.data.amazon.parquet import load_parquet
from nnrecsys.data.amazon.scripts.tfrecords import _group_by_user, _make_sequence_example


def prepare_tfrecords_seqs():
    for p in [constants.VAL_PATH, constants.TRAIN_PATH, constants.TEST_PATH]:
        if os.path.exists(p):
            shutil.rmtree(p)
            print('Cleared', p)
        os.makedirs(p, exist_ok=True)

    metadata = load_parquet(constants.CLEAN_METADATA_PATH)
    train_reviews = load_parquet(constants.TRAIN_REVIEWS_PATH)
    val_reviews = load_parquet(constants.VAL_REVIEWS_PATH)
    test_reviews = load_parquet(constants.TEST_REVIEWS_PATH)

    if constants.INCLUDE_REVIEWS:
        review_embeddings = np.load(constants.REVIEW_EMBEDDINGS_PATH)
        # assert len(review_embeddings) == len(test_reviews)
    else:
        review_embeddings = None

    id2idx = {item: idx for idx, item in enumerate(metadata.asin)}
    save_tf_records(train_reviews, constants.TRAIN_PATH, id2idx, review_embeddings, 'Train users')
    save_tf_records(val_reviews, constants.VAL_PATH, id2idx, review_embeddings, 'Val users')
    save_tf_records(test_reviews, constants.TEST_PATH, id2idx, review_embeddings, 'Test users')


def save_tf_records(reviews, path, id2idx, review_embeddings, tqdm_desc):
    for i, group in enumerate(_group_by_user(reviews, tqdm_desc)):
        fname = os.path.join(path, '{}.tfrecord'.format(i))
        with tf.python_io.TFRecordWriter(fname) as writer:
            for user_id, reviews in group:
                if user_id is None:
                    break
                sequence_example = _make_sequence_example(reviews, id2idx, review_embeddings)
                writer.write(sequence_example.SerializeToString())


if __name__ == '__main__':
    prepare_tfrecords_seqs()
