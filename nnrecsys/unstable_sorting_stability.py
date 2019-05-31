from nnrecsys.amazon_poprec import evaluate
from nnrecsys.data.amazon.scripts.split_data import split_data
from nnrecsys.data.amazon.scripts.tfrecords import prepare_tf_records
from tqdm import tqdm
import pickle

if __name__ == '__main__':
    for i in tqdm(range(100)):
        split_data()
        prepare_tf_records()
        result = evaluate()
        with open('/opt/home/msemikin/data/results/{}.pkl'.format(i), 'wb') as f:
            pickle.dump(result, f)
