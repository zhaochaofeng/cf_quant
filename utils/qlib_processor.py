"""
    自定义 qlib Processor
"""

import pandas as pd
import numpy as np
from qlib.data.dataset.utils import fetch_df_by_index
from qlib.data.dataset.processor import Processor, get_group_columns


class Winsorize(Processor):
    """ 去极值 """

    def __init__(self, fit_start_time, fit_end_time, k=3, fields_group=None):
        # NOTE: correctly set the `fit_start_time` and `fit_end_time` is very important !!!
        # `fit_end_time` **must not** include any information from the test data!!!
        self.fit_start_time = fit_start_time
        self.fit_end_time = fit_end_time
        self.fields_group = fields_group
        self.k = k

    def fit(self, df: pd.DataFrame = None):
        df = fetch_df_by_index(df, slice(self.fit_start_time, self.fit_end_time), level="datetime")
        # 指定组 （feature/lable）对应的列名集合
        self.cols = get_group_columns(df, self.fields_group)
        self.mean_train = np.nanmean(df[self.cols].values, axis=0)
        self.std_train = np.nanstd(df[self.cols].values, axis=0)

    def __call__(self, df):
        def normalize(x, mean_train=self.mean_train, std_train=self.std_train):
            lower_bound = mean_train - self.k * std_train
            upper_bound = mean_train + self.k * std_train
            return np.clip(x, lower_bound, upper_bound)

        df.loc(axis=1)[self.cols] = normalize(df[self.cols].values)
        return df


