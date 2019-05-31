# -*- coding: utf-8 -*-
"""
Created on Feb 27 2017
Author: Weiping Song
"""
import numpy as np
from tqdm import tqdm


def evaluate_sessions_batch(model, test_data, session_key, item_key, cut_off=20):
    '''
    Evaluates the GRU4Rec network wrt. recommendation accuracy measured by recall@N and MRR@N.

    Parameters
    --------
    model : A trained GRU4Rec model.
    train_data : It contains the transactions of the train set. In evaluation phrase, this is used to build item-to-id map.
    test_data : It contains the transactions of the test set. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
    cut-off : int
        Cut-off value (i.e. the length of the recommendation list; N for recall@N and MRR@N). Defauld value is 20.
    session_key : string
        Header of the session ID column in the input file (default: 'SessionId')
    item_key : string
        Header of the item ID column in the input file (default: 'ItemId')
    time_key : string
        Header of the timestamp column in the input file (default: 'Time')

    Returns
    --------
    out : tuple
        (Recall@N, MRR@N)

    '''
    batch_size = model.batch_size

    # Build itemidmap from train data.

    session_offsets = np.zeros(test_data[session_key].nunique()+1, dtype=np.int32)
    session_offsets[1:] = test_data.groupby(session_key).size().cumsum()
    n_sessions = len(session_offsets) - 1

    evaluation_point_count = 0
    mrr, recall = 0.0, 0.0
    if n_sessions < batch_size:
        batch_size = len(session_offsets) - 1
    batch_session_indices = np.arange(batch_size).astype(np.int32)
    max_batch_session_idx = batch_session_indices.max()
    batch_session_start_indices = session_offsets[batch_session_indices]
    batch_session_end_indices = session_offsets[batch_session_indices+1]

    in_idx = np.zeros(batch_size, dtype=np.int32)

    np.random.seed(42)

    with tqdm(total=n_sessions, desc='evaluation') as pbar:
        pbar.update(batch_size)
        while True:
            valid_mask = batch_session_indices >= 0
            if valid_mask.sum() == 0:
                break

            start_valid = batch_session_start_indices[valid_mask]
            min_session_len = (batch_session_end_indices[valid_mask]-start_valid).min()
            in_idx[valid_mask] = test_data[item_key].values[start_valid]
            for i in range(min_session_len-1):
                out_idx = test_data[item_key].values[start_valid+i+1]
                preds = model.predict_next_batch(batch_session_indices, in_idx)
                preds.fillna(0, inplace=True)
                in_idx[valid_mask] = out_idx
                batch_positives_scores = np.diag(preds.ix[in_idx].values)[valid_mask]
                batch_scores = preds.values.T[valid_mask].T
                batch_invalidly_ranked_counts = (batch_scores > batch_positives_scores).sum(axis=0) + 1
                rank_ok = batch_invalidly_ranked_counts < cut_off
                recall += rank_ok.sum()
                mrr += (1.0 / batch_invalidly_ranked_counts[rank_ok]).sum()
                evaluation_point_count += len(batch_invalidly_ranked_counts)

            batch_session_start_indices = batch_session_start_indices+min_session_len-1
            exhausted_sessions_indices = np.argwhere(valid_mask & (batch_session_end_indices - batch_session_start_indices <= 1))
            for idx in exhausted_sessions_indices:
                pbar.update()
                max_batch_session_idx += 1
                if max_batch_session_idx >= n_sessions:
                    batch_session_indices[idx] = -1
                else:
                    batch_session_indices[idx] = max_batch_session_idx
                    batch_session_start_indices[idx] = session_offsets[max_batch_session_idx]
                    batch_session_end_indices[idx] = session_offsets[max_batch_session_idx+1]

    return recall/evaluation_point_count, mrr/evaluation_point_count
