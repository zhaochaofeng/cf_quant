"""
    波动率因子
"""

import pandas as pd
import numpy as np
from .utils import capm_regress, cal_cmra, rolling_with_func, factor_output, get_excess_ret
from utils import dt
from barra.base import BaseDataLoader


data_loader = BaseDataLoader()
time_decorator = dt.time_decorator


@time_decorator
@factor_output
def HBETA(df, num_worker=1) -> pd.Series:
    """
    Formulation: r_{i,t} = \alpha_i + \beta_i \cdot r_{m,t} + \epsilon_{i,t}
    Description：【CAPM Beta因子】通过滚动窗口(504天)加权最小二乘回归，
        以沪深300指数收益率为自变量，股票收益率为因变量，估计每只股票的
        beta系数。半衰期为252天。
    """

    df = df.sort_index()
    close = df['$close']
    ex_ret = get_excess_ret(close)

    # CAPM 回归: window=504, half_life=252
    beta, alpha, sigma = capm_regress(ex_ret, window=504, half_life=252, num_worker=num_worker)

    return beta


@time_decorator
@factor_output
def HSIGMA(df):
    """
    Formulation: 通过CAPM模型回归得到的残差标准差
    Description：【残差波动率因子】通过滚动窗口(504天)加权最小二乘回归，
        以沪深300指数收益率为自变量，股票收益率为因变量，估计每只股票的
        残差波动率。半衰期为252天。
    """

    df = df.sort_index()
    close = df['$close']
    ex_ret = get_excess_ret(close)
    
    # CAPM 回归获取残差波动率: window=504, half_life=252
    beta, alpha, sigma = capm_regress(ex_ret, window=504, half_life=252, num_worker=1)

    return sigma


@time_decorator
@factor_output
def DASTD(df):
    """
    Formulation: 带半衰期权重的日收益率标准差
    Description：【日波动率因子】过去252个交易日的日收益率标准差，使用权重递减的
        半衰期为42天的指数权重计算。
    """

    df = df.sort_index()
    close = df['$close']
    ex_ret = get_excess_ret(close)

    dastd = ex_ret.groupby(level='instrument', group_keys=False).apply(
        lambda x: rolling_with_func(x, window=252, half_life=42, func_name='std')
    )
    # 年化
    dastd = np.sqrt(252) * dastd

    return dastd


@time_decorator
@factor_output
def CMRA(df):
    """
    Formulation: CMRA = Z_max - Z_min, where Z_t = sum of returns over t months
    Description：【累计收益范围因子】计算过去12个月的累计收益率范围，
        Z值为每个月的累计收益率，CMRA为最大值与最小值之差。
        CNE6版本使用对数收益率。
    """

    df = df.sort_index()
    close = df['$close']
    ex_ret = get_excess_ret(close)

    log_returns = np.log(1 + ex_ret)

    cmra = log_returns.groupby(level='instrument', group_keys=False).rolling(
        window=252, min_periods=int(252 * 0.8)
    ).apply(
        lambda x: cal_cmra(x, months=12, days_per_month=21),
        raw=True
    )

    return cmra


@time_decorator
def VOLATILITY_20D(df):
    """
    Formulation: VOLATILITY_{20D,t}=\sqrt{252} \times \sqrt{\frac{1}{19} \sum_{i=0}^{19} (r_{t-i}-\bar{r} )^{2}}
    Description：【波动率因子】过去 20 个交易日的日收益率标准差，即 20 日历史波动率；假设每年有 252 个交易日，进行年化处理。该因子用于衡量短期价格风险以及均值回归的可能性。
    Backtest：
        {'IC': 0.01634871895541512, 'ICIR': 0.11389840663271152, 'RIC': 0.019717409105986405, 'RICIR': 0.1173889770424266,
        'The following are analysis results of benchmark return(1day).'
                               risk
        mean               0.000732
        std                0.009844
        annualized_return  0.174329
        information_ratio  1.147896
        max_drawdown      -0.108000
        'The following are analysis results of the excess return without cost(1day).'
                               risk
        mean               0.000464
        std                0.005427
        annualized_return  0.110408
        information_ratio  1.318655
        max_drawdown      -0.100014
        'The following are analysis results of the excess return with cost(1day).'
                               risk
        mean               0.000266
        std                0.005430
        annualized_return  0.063214
        information_ratio  0.754668
        max_drawdown      -0.113394
    """

    # Ensure the index is sorted
    df = df.sort_index()

    # Extract the close price
    close_series = df['$close']

    # Calculate daily returns for each instrument
    # Shift by 1 to align return from day t-1 to t at index t
    daily_returns = close_series.groupby(level='instrument').pct_change()

    # Define the window size for volatility calculation
    window = 20

    # Calculate rolling standard deviation of daily returns
    # Use ddof=1 for sample standard deviation (as per formula with 1/(N-1))
    rolling_std = daily_returns.groupby(level='instrument').rolling(window=window, min_periods=window).std(ddof=1)

    # Reset index to match original MultiIndex structure
    # rolling_std 索引为 <instrument, instrument, datetime>
    rolling_std = rolling_std.reset_index(level=0, drop=True)

    # Annualize the volatility: multiply by sqrt(252)
    annualized_volatility = rolling_std * np.sqrt(252)

    # Create result DataFrame
    result_df = pd.DataFrame({'VOLATILITY_20D': annualized_volatility})

    # Drop rows where volatility is NaN (due to insufficient data)
    result_df = result_df.dropna()

    return result_df


