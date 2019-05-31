from nnrecsys.data.amazon import constants
import numpy as np

from nnrecsys.data.amazon.embeddings import gather_batches_from_dir

if __name__ == '__main__':
    catalog_embeddings = gather_batches_from_dir(constants.CATALOG_TEXT_EMBEDDINGS_FOLDER_PATH)
    np.save(constants.CATALOG_TEXT_EMBEDDINGS_PATH, catalog_embeddings)

