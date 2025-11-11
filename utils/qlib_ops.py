'''
    自定义 qlib 操作算子
    注：在qlib.init()初始化时注册，如：
        _setup_kwargs = {'custom_ops': [CMean]}
        qlib.init(provider_uri=provider_uri, **_setup_kwargs)
'''

import sys
import pandas as pd
import numpy as np
from qlib.data.ops import Expression, ExpressionOps
from qlib.data.ops import Operators
from qlib.data.ops import (
    Mean, Std, If, Feature
)

class LastValue(ExpressionOps):
    """ 获取特征序列最后一个值的操作符。
        如 获取 expanding 滚动均值/标准差 最后一个值得到全局均值/标准差
          'LastValue(Mean($close, 0))', 'LastValue(Std($close, 0))'
    Parameters
    ----------
    feature : Expression
        特征表达式
    Returns
    ----------
    Expression
        返回整个序列的最后一个值
    """

    def __init__(self, feature):
        self.feature = feature

    def __str__(self):
        return "{}({})".format(type(self).__name__, self.feature)

    def _load_internal(self, instrument, start_index, end_index, *args):
        # 加载整个特征序列
        series = self.feature.load(instrument, start_index, end_index, *args)
        # 获取最后一个非NaN值
        if len(series) > 0:
            # 使用 iloc[-1] 获取最后一个值
            return pd.Series([series.iloc[-1]] * len(series), index=series.index)
        else:
            return pd.Series([np.nan] * len(series), index=series.index)

    def get_longest_back_rolling(self):
        return self.feature.get_longest_back_rolling()

    def get_extended_window_size(self):
        return self.feature.get_extended_window_size()


class ConstantOps(ExpressionOps):
    """ 常数类型的计算
        如：计算整个序列的均值、方差
        返回相同序列长度的常数序列
    """
    def __init__(self, feature, func):
        self.feature = feature
        self.func = func

    def __str__(self):
        return '{}({})'.format(type(self).__name__, self.feature)

    def _load_internal(self, instrument, start_index, end_index, *args):
        # 加载整个特征序列
        series = self.feature.load(instrument, start_index, end_index, *args)
        if len(series) > 0:
            s = getattr(series, self.func)()
            return pd.Series([s] * len(series), index=series.index)
        else:
            return pd.Series([np.nan] * len(series), index=series.index)

    def get_longest_back_rolling(self):
        return self.feature.get_longest_back_rolling()

    def get_extended_window_size(self):
        return self.feature.get_extended_window_size()


class CMean(ConstantOps):
    """ 求序列均值 """
    def __init__(self, feature):
        super(CMean, self).__init__(feature, 'mean')


class CStd(ConstantOps):
    """ 求序列标准差 """
    def __init__(self, feature):
        super(CStd, self).__init__(feature, 'std')


def standardize(feature: Expression):
    """ 标准化 """
    # mean = LastValue(Mean(feature, 0))
    # std = LastValue(Std(feature, 0))
    mean = CMean(feature)
    std = CStd(feature)
    x = (feature - mean) / std
    return x


def winsorize(feature: Expression, k: int = 3):
    """ 去极值 """
    # mean = LastValue(Mean(feature, 0))
    # std = LastValue(Std(feature, 0))
    mean = CMean(feature)
    std = CStd(feature)
    lower_bound = mean - k * std
    upper_bound = mean + k * std
    x = If(feature > upper_bound, upper_bound, If(feature < lower_bound, lower_bound, feature))
    return x


def main():
    """ 测试函数 """
    import qlib
    from qlib.constant import REG_CN
    from qlib.data import D
    qlib.init(provider_uri='~/.qlib/qlib_data/custom_data_hfq', region=REG_CN)
    # 注册自定义操作
    Operators.register([CMean, CStd])
    instruments = ['SZ000001']

    feature = Feature('close')
    feature = winsorize(feature, 1)
    feature = standardize(feature)

    feas = D.features(instruments, ['$close', feature],
                      start_time='2025-07-10', end_time='2025-08-11', freq='day')
    feas.columns = ['close', 'feature']
    print(feas.head())


if __name__ == '__main__':
    sys.exit(main())