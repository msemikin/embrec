import array
import os

from tqdm import tqdm
from nnrecsys.data.amazon import constants
from nnrecsys.data.amazon.parquet import load_parquet
import numpy as np


def read_image_features(path):
    with open(path, 'rb') as f:
        while True:
            asin = f.read(10)
            if asin == b'':
                break
            a = array.array('f')
            a.fromfile(f, 4096)
            yield asin.decode('utf8'), a.tolist()


def main():
    if os.path.exists(constants.IMAGE_EMBEDDINGS_PATH):
        raise ValueError('{} already exists', constants.IMAGE_EMBEDDINGS_PATH)

    metadata = load_parquet(constants.CLEAN_METADATA_PATH)
    filtered_asin = metadata.asin.tolist()
    filtered_asin_set = set(filtered_asin)

    image_features = tqdm(read_image_features(constants.RAW_IMAGE_EMBEDDINGS_FILE),
                          desc='Reading images file')
    image_features = {asin: embedding for asin, embedding in image_features if asin in filtered_asin_set}

    print('Total products: {}, Products with images: {}'.format(len(filtered_asin), len(image_features)))

    if len(image_features) != len(filtered_asin):
        print('Cannot find image features for {} items, using 0 vectors'.format(
            len(filtered_asin) - len(image_features)))

    aligned_embeddings = [np.array(image_features[item]) if item in image_features else np.zeros(4096, dtype=float)
                          for item in filtered_asin]
    aligned_embeddings = np.array(aligned_embeddings)

    np.save(constants.IMAGE_EMBEDDINGS_PATH, aligned_embeddings)


if __name__ == '__main__':
    main()

