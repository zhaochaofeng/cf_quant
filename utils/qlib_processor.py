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

        # 确保数据类型兼容
        original_dtype = df[self.cols].dtypes
        normalized_values = normalize(df[self.cols].values)
        
        # 如果原始数据是 float32，则将结果转换为相同的类型
        if isinstance(original_dtype, pd.Series):
            # 处理多列可能有不同的 dtype 的情况
            for i, col in enumerate(self.cols):
                if df[col].dtype == np.float32:
                    normalized_values[:, i] = normalized_values[:, i].astype(np.float32)
        else:
            # 单一 dtype 情况
            if df[self.cols].dtype == np.float32:
                normalized_values = normalized_values.astype(np.float32)
        
        df.loc[:, self.cols] = normalized_values
        return df