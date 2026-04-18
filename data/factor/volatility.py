"""
    波动率因子
"""

import pandas as pd
import numpy as np
from .utils import capm_regress, cal_cmra, rolling_with_func, SENTINEL
from utils import dt

time_decorator = dt.time_decorator


@time_decorator
def BETA(df, num_worker=1):
    """
    Formulation: r_{i,t} = \alpha_i + \beta_i \cdot r_{m,t} + \epsilon_{i,t}
    Description：【CAPM Beta因子】通过滚动窗口(504天)加权最小二乘回归，
        以沪深300指数收益率为自变量，股票收益率为因变量，估计每只股票的
        beta系数。半衰期为252天。
    """

    df = df.sort_index()
    stock_returns = df['$change']

    # CAPM 回归: window=504, half_life=252
    beta, alpha, sigma = capm_regress(stock_returns, window=504, half_life=252, num_worker=num_worker)

    # 构造结果 DataFrame
    result_df = pd.DataFrame({'BETA': beta})
    result_df = result_df.dropna()

    return result_df


@time_decorator
def HSIGMA(df):
    """
    Formulation: 通过CAPM模型回归得到的残差标准差
    Description：【残差波动率因子】通过滚动窗口(504天)加权最小二乘回归，
        以沪深300指数收益率为自变量，股票收益率为因变量，估计每只股票的
        残差波动率。半衰期为252天。
    """
    
    df = df.sort_index()
    stock_returns = df['$change']
    
    # CAPM 回归获取残差波动率: window=504, half_life=252
    beta, alpha, sigma = capm_regress(stock_returns, window=504, half_life=252, num_worker=1)
    
    # 构造结果 DataFrame
    result_df = pd.DataFrame({'HSIGMA': sigma})
    result_df = result_df.dropna()
    
    return result_df


@time_decorator
def DASTD(df):
    """
    Formulation: 带半衰期权重的日收益率标准差
    Description：【日波动率因子】过去252个交易日的日收益率标准差，使用权重递减的
        半衰期为42天的指数权重计算。
    """
    
    # Ensure the index is sorted
    df = df.sort_index()
    
    # Extract the daily returns
    daily_returns = df['$change']
    
    # Calculate rolling standard deviation with half-life weighting
    # Window = 252 days, Half-life = 42 days (BARRA CNE6 specification)
    dastd_series = rolling_with_func(
        daily_returns, 
        window=252, 
        half_life=42,
        func_name='std'
    )
    # 年化
    dastd_series = np.sqrt(252) * dastd_series
    # Create result DataFrame
    result_df = pd.DataFrame({'DASTD': dastd_series})
    
    # Drop rows where volatility is NaN (due to insufficient data)
    result_df = result_df.dropna()
    
    return result_df


@time_decorator
def CMRA(df):
    """
    Formulation: CMRA = Z_max - Z_min, where Z_t = sum of returns over t months
    Description：【累计收益范围因子】计算过去12个月的累计收益率范围，
        Z值为每个月的累计收益率，CMRA为最大值与最小值之差。
        CNE6版本使用对数收益率。
    """
    
    # Ensure the index is sorted
    df = df.sort_index()
    
    # Extract daily returns and convert to log returns (CNE6 version)
    log_returns = np.log(1 + df['$change'])
    
    # Replace NaN with SENTINEL for consistent missing value handling
    log_returns = log_returns.where(pd.notnull(log_returns), SENTINEL)
    
    # Apply rolling CMRA calculation per instrument
    # cal_cmra internally handles SENTINEL values
    cmra_series = log_returns.groupby(level='instrument').rolling(
        window=252, min_periods=21  # At least 1 month of data
    ).apply(
        lambda x: cal_cmra(x, months=12, days_per_month=21, sentinel=SENTINEL),
        raw=True
    )
    
    # Reset the extra groupby level to restore original MultiIndex structure
    cmra_series = cmra_series.reset_index(level=0, drop=True)
    
    # Create result DataFrame
    result_df = pd.DataFrame({'CMRA': cmra_series})
    
    # Drop NaN values only
    # Note: CMRA can be 0 when max Z equals min Z (rare but valid)
    result_df = result_df.dropna()
    
    return result_df


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


