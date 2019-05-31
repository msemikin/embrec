import pandas as pd
import numpy as np
from tqdm import tqdm


def get_session_offsets(data: pd.DataFrame, session_key):
    offset_sessions = np.zeros(data[session_key].nunique() + 1, dtype=np.int32)
    offset_sessions[1:] = data.groupby(session_key).size().cumsum()
    return offset_sessions


class SessionsIterator:
    def __init__(self, train_data: pd.DataFrame, session_key: str, item_idx_key: str, batch_size: int):
        self.train_data = train_data
        self.session_key = session_key
        self.item_idx_key = item_idx_key
        self.batch_size = batch_size
        self.session_offsets = get_session_offsets(self.train_data, self.session_key)
        self.n_sessions = len(self.session_offsets) - 1

    def generate(self):
        session_indices = np.arange(self.n_sessions)
        batch_sessions_indices = np.arange(self.batch_size)
        max_batch_session_idx = batch_sessions_indices.max()

        batch_session_start_indices = self.session_offsets[session_indices[batch_sessions_indices]]
        batch_session_end_indices = self.session_offsets[session_indices[batch_sessions_indices] + 1]

        finished = False
        with tqdm(total=self.n_sessions) as pbar:
            pbar.update(self.batch_size)
            while not finished:
                min_session_len = (batch_session_end_indices - batch_session_start_indices).min()

                batches = []
                out_idx = self.train_data[self.item_idx_key].values[batch_session_start_indices]
                for i in range(min_session_len - 1):
                    in_idx = out_idx
                    out_idx = self.train_data[self.item_idx_key].values[batch_session_start_indices + i + 1]
                    batches.append((in_idx, out_idx))

                batch_session_start_indices = batch_session_start_indices + min_session_len - 1
                exhausted_sessions_indices = np.argwhere((batch_session_end_indices - batch_session_start_indices) <= 1)

                for idx in exhausted_sessions_indices:
                    pbar.update()
                    max_batch_session_idx += 1
                    if max_batch_session_idx >= self.n_sessions:
                        finished = True
                        break

                    batch_sessions_indices[idx] = max_batch_session_idx
                    batch_session_start_indices[idx] = self.session_offsets[session_indices[max_batch_session_idx]]
                    batch_session_end_indices[idx] = self.session_offsets[session_indices[max_batch_session_idx] + 1]

                yield batches, exhausted_sessions_indices
