import pandas as pd


def save_parquet(fname, df):
    df.to_parquet(fname, engine='fastparquet', compression='gzip')


def load_parquet(fname):
    return pd.read_parquet(fname, engine='fastparquet')
