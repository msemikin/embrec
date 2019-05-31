import os

from nnrecsys.data.amazon import constants
from nnrecsys.data.amazon.parquet import save_parquet
from nnrecsys.data.amazon.scripts.split_data import get_core_reviews, get_df, get_catalog_desc_text


def split_seqs():
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

    _, reviews = get_core_reviews(reviews)
    metadata = get_df(os.path.join(constants.DATA_ROOT, constants.RAW_METADATA_FILE))

    print('Metadata shape:', metadata.shape)
    metadata = metadata[metadata.asin.isin(reviews.asin)].reset_index(drop=True)
    print('Metadata shape after filtering:', metadata.shape)
    metadata['texts'] = get_catalog_desc_text(metadata)

    users = reviews.reviewerID.drop_duplicates()
    holdout_users = users.sample(frac=0.2, random_state=42)

    train_users = users[~users.isin(holdout_users)]
    assert not holdout_users.isin(train_users.values).any()
    assert len(holdout_users) + len(train_users) == len(users)

    val_users = holdout_users.sample(frac=0.5, random_state=43)
    test_users = holdout_users[~holdout_users.isin(val_users)]
    assert len(test_users) + len(val_users) == len(holdout_users)
    assert not val_users.isin(test_users.values).any()

    train_reviews = reviews[reviews.reviewerID.isin(train_users)]
    val_reviews = reviews[reviews.reviewerID.isin(val_users)]
    test_reviews = reviews[reviews.reviewerID.isin(test_users)]
    assert len(train_reviews) + len(val_reviews) + len(test_reviews) == len(reviews)

    print("\n\n-----------------------")
    print("Users split:")
    print('Train:', len(train_users), '\tVal:', len(val_users), '\tTest', len(test_users))
    print("-----------------------\n\n")

    save_parquet(constants.CLEAN_METADATA_PATH, metadata)
    save_parquet(constants.TRAIN_REVIEWS_PATH, train_reviews)
    save_parquet(constants.VAL_REVIEWS_PATH, val_reviews)
    save_parquet(constants.TEST_REVIEWS_PATH, test_reviews)


if __name__ == '__main__':
    split_seqs()
