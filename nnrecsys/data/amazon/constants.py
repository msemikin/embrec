import os


ROOT_DIR = os.path.expanduser('~/nnrecsys')
DATASET_NAME = 'clothing'

DATA_ROOT = os.path.join(ROOT_DIR, 'data/amazon', DATASET_NAME)
INCLUDE_REVIEWS = False
RAW_REVIEWS_FILE = 'reviews.json.gz'
RAW_METADATA_FILE = 'metadata.json.gz'
RAW_IMAGE_EMBEDDINGS_FILE = '/data/public_datasets/amazon/image_features.b'

SPLIT_FOLDER = os.path.join(DATA_ROOT, 'split')

CLEAN_METADATA_PATH = os.path.join(SPLIT_FOLDER, 'metadata.parquet.gzip')
TRAIN_REVIEWS_PATH = os.path.join(SPLIT_FOLDER, 'train_reviews.parquet.gzip')
VAL_REVIEWS_PATH = os.path.join(SPLIT_FOLDER, 'val_reviews.parquet.gzip')
TEST_REVIEWS_PATH = os.path.join(SPLIT_FOLDER, 'test_reviews.parquet.gzip')
SAMPLE_REVIEWS_PATH = os.path.join(SPLIT_FOLDER, 'sample_reviews.parquet.gzip')

TRAIN_PATH = os.path.join(SPLIT_FOLDER, 'train')
VAL_PATH = os.path.join(SPLIT_FOLDER, 'val')
TEST_PATH = os.path.join(SPLIT_FOLDER, 'test')
SAMPLE_PATH = os.path.join(SPLIT_FOLDER, 'sample')

CATALOG_TEXT_EMBEDDINGS_PATH = os.path.join(SPLIT_FOLDER, 'catalog_elmo.npy')
CATALOG_TEXT_PCA_EMBEDDINGS_PATH = os.path.join(SPLIT_FOLDER, 'catalog_elmo_pca.npy')
CATALOG_TEXT_TFIDF_EMBEDDINGS_PATH = os.path.join(SPLIT_FOLDER, 'catalog_tfidf.npy')
CATALOG_TEXT_TFIDF_PCA_EMBEDDINGS_PATH = os.path.join(SPLIT_FOLDER, 'catalog_tfidf_pca_125.npy')
CATALOG_TEXT_WORD2VEC_EMBEDDINGS_PATH = os.path.join(SPLIT_FOLDER, 'catalog_word2vec.npy')
CATALOG_TEXT_EMBEDDINGS_FOLDER_PATH = os.path.join(SPLIT_FOLDER, 'catalog_elmo')

REVIEW_EMBEDDINGS_PATH = os.path.join(SPLIT_FOLDER, 'reviews_elmo.npy')
REVIEW_EMBEDDINGS_FOLDER_PATH = os.path.join(SPLIT_FOLDER, 'reviews_elmo')

IMAGE_EMBEDDINGS_PATH = os.path.join(SPLIT_FOLDER, 'images.npy')
IMAGE_PCA_EMBEDDINGS_PATH = os.path.join(SPLIT_FOLDER, 'images_pca_125.npy')
IMAGE_SPRITE_FILE = os.path.join(SPLIT_FOLDER, 'sprite.jpg')
EMBEDDING_METADATA_TSV = os.path.join(SPLIT_FOLDER, 'catalog.tsv')

IMG_TEXT_PCA_EMBEDDINGS_PATH = os.path.join(SPLIT_FOLDER, 'catalog_tfidf_images_pca.npy')

os.makedirs(SPLIT_FOLDER, exist_ok=True)

