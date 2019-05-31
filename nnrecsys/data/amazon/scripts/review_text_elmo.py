import os

import numpy as np

from nnrecsys.data.amazon import constants
from nnrecsys.data.amazon.embeddings import extract_elmo_embeddings
from nnrecsys.data.amazon.parquet import load_parquet

START_BATCH = 0


def extract_review_embeddings():
    os.makedirs(constants.REVIEW_EMBEDDINGS_FOLDER_PATH, exist_ok=True)

    reviews = load_parquet(constants.TEST_REVIEWS_PATH)
    review_text = reviews.review_text

    for i, embeddings in extract_elmo_embeddings(START_BATCH, review_text, batch_size=32):
        fname = os.path.join(constants.REVIEW_EMBEDDINGS_FOLDER_PATH, '{}.npy'.format(i))
        np.save(fname, embeddings)


if __name__ == '__main__':
    extract_review_embeddings()
