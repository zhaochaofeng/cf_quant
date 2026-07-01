"""
    流动性因子 (Liquidity Factors)
    
    基于 BARRA CNE6 模型实现：
    - STOM: 月度换手率对数均值
    - STOQ: 季度换手率对数均值
    - STOA: 年度换手率对数均值
    - ATVR: 换手率波动率（加权换手率之和）
"""

import pandas as pd

from utils.dt import time_decorator
from .utils import cal_liquidity, rolling_with_func, SENTINEL
from .utils import factor_output

@time_decorator
@factor_output
def STOM(df) -> pd.Series:
    """
    Formulation: STOM = ln(mean(amount / circ_mv))
    Description：【月度换手率因子】过去1个月（21个交易日）的日换手率对数均值。
        换手率 = 成交额($amount) / 流通市值($circ_mv)，反映股票的交易活跃度。
        该因子用于衡量股票的短期流动性风险。
    """
    df = df.sort_index()
    
    # 计算日换手率: 成交额 / 流通市值
    share_turnover = df['$amount'] / df['$circ_mv']

    # 将 NaN 替换为 SENTINEL
    share_turnover = share_turnover.where(pd.notnull(share_turnover), SENTINEL)

    # 滚动计算月度流动性因子 (21天窗口)
    # NaN已通过SENTINEL填充，窗口始终满，min_periods=window确保只计算完整窗口
    # 有效数据比例由cal_liquidity内部的min_valid_ratio控制
    stom_series = share_turnover.groupby(level='instrument').rolling(
        window=21, min_periods=21
    ).apply(
        lambda x: cal_liquidity(x, days_per_month=21, sentinel=SENTINEL, min_valid_ratio=0.8),
        raw=True
    )
    # 删除重复 instrument 索引
    stom = stom_series.reset_index(level=0, drop=True)

    return stom


@time_decorator
@factor_output
def STOQ(df) -> pd.Series:
    """
    Formulation: STOQ = ln(mean(amount / circ_mv))
    Description：【季度换手率因子】过去1个季度（63个交易日）的日换手率对数均值。
        换手率 = 成交额($amount) / 流通市值($circ_mv)，反映股票的中期流动性特征。
        该因子用于衡量股票的中期流动性风险。
    """
    # 确保索引排序
    df = df.sort_index()
    
    # 计算日换手率: 成交额 / 流通市值
    share_turnover = df['$amount'] / df['$circ_mv']
    share_turnover = share_turnover.where(pd.notnull(share_turnover), SENTINEL)

    # 滚动计算季度流动性因子 (63天窗口)
    stoq_series = share_turnover.groupby(level='instrument').rolling(
        window=63, min_periods=63
    ).apply(
        lambda x: cal_liquidity(x, days_per_month=21, sentinel=SENTINEL, min_valid_ratio=0.8),
        raw=True
    )
    
    # 重置多余的 groupby 层级
    stoq = stoq_series.reset_index(level=0, drop=True)

    return stoq


@time_decorator
@factor_output
def STOA(df) -> pd.Series:
    """
    Formulation: STOA = ln(mean(amount / circ_mv))
    Description：【年度换手率因子】过去1年（252个交易日）的日换手率对数均值。
        换手率 = 成交额($amount) / 流通市值($circ_mv)，反映股票的长期流动性特征。
        该因子用于衡量股票的长期流动性风险。
    """
    # 确保索引排序
    df = df.sort_index()
    
    # 计算日换手率: 成交额 / 流通市值
    share_turnover = df['$amount'] / df['$circ_mv']

    # 将 NaN 替换为 SENTINEL
    share_turnover = share_turnover.where(pd.notnull(share_turnover), SENTINEL)

    # 滚动计算年度流动性因子 (252天窗口)
    stoa_series = share_turnover.groupby(level='instrument').rolling(
        window=252, min_periods=252
    ).apply(
        lambda x: cal_liquidity(x, days_per_month=21, sentinel=SENTINEL, min_valid_ratio=0.8),
        raw=True
    )
    stoa = stoa_series.reset_index(level=0, drop=True)

    return stoa


@time_decorator
@factor_output
def ATVR(df) -> pd.Series:
    """
    Formulation: ATVR = sum(turnover_rate * weight)
    Description：【换手率波动率因子】过去1年（252个交易日）的日换手率加权之和，
        使用半衰期为63天的指数权重。该因子用于衡量股票的换手率波动情况。
        与 STOM/STOQ/STOA 不同，ATVR 直接使用换手率率（%）而非成交额/市值比。
    """
    # 确保索引排序
    df = df.sort_index()
    turnover_rate = df['$amount'] / df['$circ_mv']

    turnover_rate = turnover_rate.where(pd.notnull(turnover_rate), SENTINEL)
    atvr_series = turnover_rate.groupby(level='instrument').apply(
        lambda x: rolling_with_func(x, window=252, half_life=63, func_name='sum')
    )

    atvr = atvr_series.reset_index(level=0, drop=True)

    return atvr
