from nnrecsys.data.amazon import constants
import numpy as np
from sklearn.decomposition import TruncatedSVD


def get_pca(path, save_path):
    svd = TruncatedSVD(n_components=125, random_state=42)
    embeddings = np.load(path)
    print('Before, shape: ', embeddings.shape)
    reduced = svd.fit_transform(embeddings)
    print('After, shape: ', reduced.shape)
    np.save(save_path, reduced)


if __name__ == '__main__':
    # get_pca(constants.CATALOG_TEXT_EMBEDDINGS_PATH, constants.CATALOG_TEXT_PCA_EMBEDDINGS_PATH)
    get_pca(constants.CATALOG_TEXT_TFIDF_EMBEDDINGS_PATH, constants.CATALOG_TEXT_TFIDF_PCA_EMBEDDINGS_PATH)
    get_pca(constants.IMAGE_EMBEDDINGS_PATH, constants.IMAGE_PCA_EMBEDDINGS_PATH)
