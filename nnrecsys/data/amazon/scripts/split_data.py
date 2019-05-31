import gzip
import os

import pandas as pd
from tqdm import tqdm

from nnrecsys.data.amazon import constants
from nnrecsys.data.amazon.parquet import save_parquet

MAX_REVIEW_SIZE = 250
MAX_SEQ_LEN = 50


def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)


def get_df(path):
    i = 0
    df = {}
    for d in tqdm(parse(path), desc=path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')


def get_core_reviews(reviews):
    item_counts = reviews.asin.value_counts()
    popular_items = item_counts[item_counts >= 5].index

    user_counts = reviews.reviewerID.value_counts()
    popular_users = user_counts[user_counts >= 5].index

    print('\nBefore: items={}, users={}, reviews={}'.format(len(item_counts), len(user_counts), len(reviews)))

    filtered = reviews[reviews.asin.isin(popular_items) &
                       reviews.reviewerID.isin(popular_users)] \
        .sort_values(['reviewerID', 'unixReviewTime']) \
        .copy()

    filtered = truncate_user_seqs(filtered)
    filtered['review_text'] = get_review_text(filtered)

    user_counts = filtered.reviewerID.value_counts()
    train_users = user_counts[user_counts > 3].index
    test_users = user_counts[user_counts > 2].index

    test = filtered[filtered.reviewerID.isin(test_users)] \
        .reset_index(drop=True) \
        .copy()

    train = test[test.reviewerID.isin(train_users)]

    print('Core: items={}, users={}, reviews={}'.format(len(filtered.asin.unique()),
                                                        len(filtered.reviewerID.unique()),
                                                        len(filtered)))

    print('Train: items={}, users={}, reviews={}'.format(len(train.asin.unique()),
                                                         len(train.reviewerID.unique()),
                                                         len(train)))

    print('Test: items={}, users={}, reviews={}'.format(len(test.asin.unique()),
                                                        len(test.reviewerID.unique()),
                                                        len(test)))

    return train, test


def truncate_user_seqs(reviews):
    n_truncated = 0

    def truncate_seq(seq):
        if len(seq) <= MAX_SEQ_LEN:
            return seq

        n_to_truncate = len(seq) - MAX_SEQ_LEN
        nonlocal n_truncated
        n_truncated += n_to_truncate
        return seq.iloc[n_to_truncate:]

    result = reviews.groupby('reviewerID').apply(truncate_seq)
    print('Truncated {} / {} events for users with # of purchases > {}'.format(n_truncated,
                                                                               len(reviews),
                                                                               MAX_SEQ_LEN))
    return result.reset_index(drop=True)


def split_data():
    for p in [constants.CLEAN_METADATA_PATH, constants.TRAIN_REVIEWS_PATH, constants.TEST_REVIEWS_PATH]:
        if os.path.exists(p):
            os.remove(p)
            print('Removed', p)

    reviews = get_df(os.path.join(constants.DATA_ROOT, constants.RAW_REVIEWS_FILE))
    print(reviews.head())
    print()
    print(reviews.dtypes)

    if not os.path.exists(constants.SPLIT_FOLDER):
        os.makedirs(constants.SPLIT_FOLDER)

    train_reviews, test_reviews = get_core_reviews(reviews)

    metadata = get_df(os.path.join(constants.DATA_ROOT, constants.RAW_METADATA_FILE))
    print('Metadata shape:', metadata.shape)
    metadata = metadata[metadata.asin.isin(test_reviews.asin)].reset_index(drop=True)
    print('Metadata shape after filtering:', metadata.shape)
    metadata['texts'] = get_catalog_desc_text(metadata)

    save_parquet(constants.CLEAN_METADATA_PATH, metadata)
    save_parquet(constants.TRAIN_REVIEWS_PATH, train_reviews)
    save_parquet(constants.TEST_REVIEWS_PATH, test_reviews)


def get_review_text(reviews):
    text = reviews.summary + ' ' + reviews.reviewText
    return truncate_text_feature(text)


def get_catalog_desc_text(metadata):
    catalog = (metadata.title.fillna('') + ' ' + metadata.description.fillna(''))
    return truncate_text_feature(catalog)


def truncate_text_feature(text):
    counts_before = text.apply(lambda t: len(t.split()))
    truncated = text.apply(lambda t: ' '.join(t.split()[:MAX_REVIEW_SIZE]))
    counts_after = truncated.apply(lambda t: len(t.split()))
    print('Max words in review before: {}, after: {}'.format(counts_before.max(), counts_after.max()))
    print('Truncated {} / {} texts'.format((counts_before > MAX_REVIEW_SIZE).sum(), len(counts_before)))
    return truncated


if __name__ == '__main__':
    split_data()
