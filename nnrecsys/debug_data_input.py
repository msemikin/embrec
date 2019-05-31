from collections import Counter

import tensorflow as tf
import pandas as pd
from nnrecsys.data.amazon.input import input_fn
from nnrecsys.data.amazon import constants
from nnrecsys.amazon_emb import get_item_probs, get_estimator, get_run_params


catalog = pd.read_parquet(constants.CLEAN_METADATA_PATH)
item_probs = get_item_probs(catalog, constants.TRAIN_PATH)
ds = input_fn(constants.TRAIN_PATH, 128, False, 1024, item_probs).repeat()
next_batch = ds.make_one_shot_iterator().get_next()

seq_lengths = []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(200):
        features, labels = sess.run(next_batch)
        seq_lengths += features['seq_length'].tolist()


print(Counter(seq_lengths))
