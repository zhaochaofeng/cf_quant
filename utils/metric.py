"""
    评价函数。用于评估指标的计算
"""
import pandas as pd
from qlib.contrib.eva.alpha import calc_ic


def ic(pred: pd.Series, label: pd.Series, date_col="datetime", dropna=False)-> (pd.Series, pd.Series):
    """ 计算IC, RIC"""
    return calc_ic(pred, label, date_col=date_col, dropna=dropna)


def group_return(factor: pd.Series, close: pd.Series, n: int = 10, k: int=1):
    """ 计算分组收益 """
    com_idx = factor.index.intersection(close.index)
    factor = factor.loc[com_idx]
    close = close.loc[com_idx]
    factor.sort_index(inplace=True)
    close.sort_index(inplace=True)

    ret = close.groupby('instrument', group_keys=False).apply(lambda x: x.shift(-k-1) / x.shift(-1) - 1)
    ret.name = 'ret'
    labels = pd.qcut(factor, n, labels=False, duplicates='drop')
    # 返回年化收益率
    return ret.groupby(labels).mean() * 252



