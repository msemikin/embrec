import os

import numpy as np

from nnrecsys.data.amazon import constants
from nnrecsys.data.amazon.embeddings import extract_elmo_embeddings
from nnrecsys.data.amazon.parquet import load_parquet

START_BATCH = 0


def extract_catalog_embeddings():
    os.makedirs(constants.CATALOG_TEXT_EMBEDDINGS_FOLDER_PATH, exist_ok=True)

    metadata = load_parquet(constants.CLEAN_METADATA_PATH)

    for i, embeddings in extract_elmo_embeddings(START_BATCH, metadata.texts, batch_size=32):
        fname = os.path.join(constants.CATALOG_TEXT_EMBEDDINGS_FOLDER_PATH, '{}.npy'.format(i))
        np.save(fname, embeddings)


if __name__ == '__main__':
    extract_catalog_embeddings()

