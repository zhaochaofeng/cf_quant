"""
    评价函数。用于评估指标的计算
"""
import pandas as pd
from qlib.contrib.eva.alpha import calc_ic
import qlib
from qlib.data import D
from qlib.contrib.evaluate import backtest_daily
from qlib.contrib.evaluate import risk_analysis
from qlib.contrib.strategy import TopkDropoutStrategy


def ic_ric(pred: pd.Series, label: pd.Series, date_col="datetime", dropna=False)-> (pd.Series, pd.Series):
    """ 计算IC, RIC"""
    return calc_ic(pred, label, date_col=date_col, dropna=dropna)


def ic_ric_period(pred: pd.Series, label: pd.Series, period="year", dropna=False) -> (pd.Series, pd.Series):
    """ 指定计算周期计算IC, RIC
        period: 计算周期。day, month, year
    """
    ic, ric = ic_ric(pred, label, dropna=dropna)
    df = pd.DataFrame({"ic": ic, "ric": ric})

    # df = pd.DataFrame({"pred": pred, "label": label})
    if period == "year":
        df['year'] = df.index.get_level_values('datetime').year
    elif period == "month":
        df['month'] = df.index.get_level_values('datetime').month
    elif period == "day":
        df['day'] = df.index.get_level_values('datetime').day
    else:
        raise ValueError("period must be day, month, year")
    ic = df.groupby(period, group_keys=False)['ic'].mean()
    ric = df.groupby(period, group_keys=False)['ric'].mean()
    return ic, ric


def group_return(factor: pd.Series, close: pd.Series, n: int = 10, k: int=1):
    """ 计算分组收益。将 因子值按照分位数分组，计算每组收益率年化收益率
        n: 分组数
        k: 收益率计算日频跨度
    """
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



