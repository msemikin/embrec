import numpy as np

from nnrecsys.data.amazon import constants
from nnrecsys.data.amazon.parquet import load_parquet
from sklearn.feature_extraction.text import TfidfVectorizer


def extract_tfidf():
    metadata = load_parquet(constants.CLEAN_METADATA_PATH)
    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
    tfidf_matrix = tfidf.fit_transform(metadata.texts)
    print(tfidf_matrix.shape)
    np.save(constants.CATALOG_TEXT_TFIDF_EMBEDDINGS_PATH, tfidf_matrix.todense())


if __name__ == '__main__':
    extract_tfidf()
