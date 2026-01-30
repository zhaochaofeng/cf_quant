"""
    动量因子函数
"""

import pandas as pd
import numpy as np


def MOM_10D(df):
    """
    Formulation: MOM_{10D,t} = \frac{Close_t - Close_{t-10}}{Close_{t-10}}
    Description：【动量因子】计算10天的收益率短期价格趋势及动量效应
    Backtest：
    {'IC': 0.017894337475029908, 'ICIR': 0.1445482899646844, 'RIC': 0.013924254228330967, 'RICIR': 0.10783298067152193}
    'The following are analysis results of benchmark return(1day).'
                           risk
    mean               0.000646
    std                0.009712
    annualized_return  0.153717
    information_ratio  1.025907
    max_drawdown      -0.109772
    'The following are analysis results of the excess return without cost(1day).'
                           risk
    mean               0.000104
    std                0.003998
    annualized_return  0.024646
    information_ratio  0.399569
    max_drawdown      -0.062664
    'The following are analysis results of the excess return with cost(1day).'
                           risk
    mean              -0.000091
    std                0.003997
    annualized_return -0.021686
    information_ratio -0.351696
    max_drawdown      -0.071350
    """

    # Ensure the index is sorted
    df = df.sort_index()

    # Extract the close price column
    close_series = df['$close']

    # Group by instrument and calculate the 10-day momentum
    # Shift by 10 to get Close_{t-10}, then compute percentage change
    mom_10d = close_series.groupby(level='instrument').pct_change(periods=10) * 100

    # Rename the series to the factor name
    mom_10d.name = 'MOM_10D'

    # Convert to DataFrame
    result_df = mom_10d.to_frame()

    return result_df


def REVERSAL_5D(df):
    """
    Formulation: REVERSAL_{5D,t} = - \frac{Close_t - Close_{t-5}}{Close_{t-5}} \times 100
    Description: 【均值回归因子】 5 日价格反转因子，计算方法为 5 日动量的负值。该因子通过识别过去 5 个交易日内价格向单一方向显著移动后的超买或超卖状态，来捕捉短期的均值回归现象。
    Indicator:
    {'IC': 0.015911179329660092, 'ICIR': 0.11922679649114731, 'RIC': 0.018980111842142367, 'RICIR': 0.13298780964608237}
    'The following are analysis results of benchmark return(1day).'
                           risk
    mean               0.000646
    std                0.009712
    annualized_return  0.153717
    information_ratio  1.025907
    max_drawdown      -0.109772
    'The following are analysis results of the excess return without cost(1day).'
                           risk
    mean               0.000417
    std                0.004317
    annualized_return  0.099215
    information_ratio  1.489651
    max_drawdown      -0.058491
    'The following are analysis results of the excess return with cost(1day).'
                           risk
    mean               0.000219
    std                0.004319
    annualized_return  0.052207
    information_ratio  0.783572
    max_drawdown      -0.067981


    """
    # Ensure the index is sorted
    df = df.sort_index()

    # Extract the close price column
    close_series = df['$close']

    # Define the look-back period (5 trading days)
    lookback = 5

    # Function to compute 5-day reversal per instrument
    def compute_reversal(group):
        # Calculate 5-day momentum: (Close_t - Close_{t-5}) / Close_{t-5} * 100
        momentum = (group - group.shift(lookback)) / group.shift(lookback) * 100
        # Reversal is negative of momentum
        reversal = -momentum
        return reversal

    # Apply the calculation per instrument
    result_series = close_series.groupby(level='instrument').apply(compute_reversal)

    # Reset index to maintain MultiIndex structure
    result_series = result_series.reset_index(level=0, drop=True)

    # Sort the index to ensure proper alignment
    result_series = result_series.sort_index()

    # Create a DataFrame with the factor name as column
    result_df = pd.DataFrame({'REVERSAL_5D': result_series})

    return result_df


def MOM_VOL_ADJ_10D(df):
    """
    Formulation: MOM\_VOL\_ADJ_{10D,t} = \frac{MOM_{10D,t}}{VOLATILITY_{10D,t}}
    Description: 【波动率调节动量因子】 10 日波动率调节动量，计算方法为 10 日动量除以 10 日历史波动率。该因子通过风险对趋势信号进行归一化处理，旨在识别波动率较低但动量较强的标的，从而提供更纯净的动量信号
    Indication:
    {'IC': 0.017017542216167286, 'ICIR': 0.1300690966361479, 'RIC': 0.018690758060998448, 'RICIR': 0.13056632305383084}
    'The following are analysis results of benchmark return(1day).'
                           risk
    mean               0.000646
    std                0.009712
    annualized_return  0.153717
    information_ratio  1.025907
    max_drawdown      -0.109772
    'The following are analysis results of the excess return without cost(1day).'
                           risk
    mean               0.000646
    std                0.004279
    annualized_return  0.153670
    information_ratio  2.327898
    max_drawdown      -0.031030
    'The following are analysis results of the excess return with cost(1day).'
                           risk
    mean               0.000453
    std                0.004281
    annualized_return  0.107749
    information_ratio  1.631522
    max_drawdown      -0.034492
    """
    # Ensure the index is sorted
    df = df.sort_index()

    # Extract the close price column
    close_series = df['$close']

    # Calculate 10-day momentum: (Close_t - Close_{t-10}) / Close_{t-10} * 100
    mom_10d = close_series.groupby(level='instrument').pct_change(periods=10) * 100

    # Calculate daily returns: r_t = (Close_t - Close_{t-1}) / Close_{t-1}
    daily_returns = close_series.groupby(level='instrument').pct_change()

    # Calculate 10-day historical volatility: rolling standard deviation of daily returns over 10 days
    # Use window=10 to include current day and previous 9 days, with min_periods=10
    # Do NOT annualize (remove * np.sqrt(252))
    volatility_10d = daily_returns.groupby(level='instrument').rolling(window=10, min_periods=10).std()
    # The rolling operation adds an extra level, reset index to match original
    volatility_10d = volatility_10d.reset_index(level=0, drop=True)

    # Align indices and combine into DataFrame
    combined = pd.DataFrame({'MOM': mom_10d, 'VOL': volatility_10d})

    # Calculate volatility-adjusted momentum: MOM / VOL
    # Avoid division by zero or very small numbers
    combined['MOM_VOL_ADJ_10D'] = combined['MOM'] / combined['VOL'].replace(0, np.nan)

    # Extract the factor series
    factor_series = combined['MOM_VOL_ADJ_10D']
    factor_series.name = 'MOM_VOL_ADJ_10D'

    # Convert to DataFrame
    result_df = factor_series.to_frame()

    return result_df


