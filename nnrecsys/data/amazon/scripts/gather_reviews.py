from nnrecsys.data.amazon import constants
import numpy as np

from nnrecsys.data.amazon.embeddings import gather_batches_from_dir

if __name__ == '__main__':
    review_embeddings = gather_batches_from_dir(constants.REVIEW_EMBEDDINGS_FOLDER_PATH)
    np.save(constants.REVIEW_EMBEDDINGS_PATH, review_embeddings)

