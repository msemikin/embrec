#!/usr/bin/env python
# coding: utf-8


import pandas as pd


VOCABULARY_FILE = 'data/items.txt'
TRAIN_PATH = 'data/train.txt'
VAL_PATH = 'data/val.txt'
TEST_PATH = 'data/test.txt'

if __name__ == '__main__':
    print('---------Generating train and validation---------')
    data = pd.read_csv('data/yoochoose-clicks.dat', names=['session_id', 'timestamp', 'item_id', 'category'])
    print('Clicks shape:', data.shape)

    all_item_ids = sorted(data.item_id.unique().tolist())
    data.sort_values(['session_id', 'timestamp'], inplace=True)

    print('Creating vocabulary')
    data.item_id.drop_duplicates().to_csv(VOCABULARY_FILE, index=False)

    print('Creating session rows')
    sessions = data.groupby('session_id').agg({'item_id': lambda series: series.tolist()})
    sessions = sessions[sessions.item_id.apply(len) > 1]
    sessions = sessions.item_id.apply(lambda items: ' '.join(map(str, items)))

    print('# sessions', len(sessions))

    train_size = 0.9
    boundary = int(train_size * len(sessions))
    train = sessions.iloc[:boundary]
    val = sessions.iloc[boundary:]

    print('# train sessions:', len(train), '\t# val sessions:', len(val))

    train.to_csv(TRAIN_PATH, index=False)
    val.to_csv(VAL_PATH, index=False)

    print('---------Generating test---------')
    test_clicks = pd.read_csv('data/yoochoose-test.dat', names=['session_id', 'timestamp', 'item_id', 'category'])
    print('Test before filtering:', test_clicks.shape)

    test_clicks = test_clicks[test_clicks.item_id.isin(set(all_item_ids))]
    print('Test after filtering:', test_clicks.shape)

    test_clicks.sort_values(['session_id', 'timestamp'], inplace=True)
    test_sessions = test_clicks.groupby('session_id').agg({'item_id': lambda series: series.tolist()})
    test_sessions = test_sessions[test_sessions.item_id.apply(len) > 1]
    test_sessions = test_sessions.item_id.apply(lambda items: ' '.join(map(str, items)))

    print('# test sessions', len(test_sessions))
    test_sessions.to_csv(TEST_PATH, index=False)

