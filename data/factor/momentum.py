"""
    动量因子函数
"""

import pandas as pd
import numpy as np


def MOM_10D(df):
    """
    Formulation: MOM_{10D,t} = \frac{Close_t - Close_{t-10}}{Close_{t-10}}
    Description：计算10天的收益率短期价格趋势及动量效应
    Backtest：
        {'IC': 0.013424545859375895, 'ICIR': 0.09727675306079529, 'RIC': 0.0169775775558526, 'RICIR': 0.10040896575551078
        'The following are analysis results of benchmark return(1day).'
                               risk
        mean               0.000732
        std                0.009844
        annualized_return  0.174329
        information_ratio  1.147896
        max_drawdown      -0.108000
        'The following are analysis results of the excess return without cost(1day).'
                               risk
        mean               0.000429
        std                0.005609
        annualized_return  0.102049
        information_ratio  1.179279
        max_drawdown      -0.100784
        'The following are analysis results of the excess return with cost(1day).'
                               risk
        mean               0.000230
        std                0.005612
        annualized_return  0.054834
        information_ratio  0.633378
        max_drawdown      -0.113143

    """

    # Ensure the index is sorted
    df = df.sort_index()

    # Extract the adjusted close price column
    close_series = df['$close']

    # Group by instrument and calculate the 10-day momentum
    # Shift by 10 days to get Close_{t-10}, then compute percentage change
    mom_10d = close_series.groupby(level='instrument', group_keys=False).apply(lambda x: (x - x.shift(10)) / x.shift(10))

    # Create a DataFrame with the result
    result_df = pd.DataFrame({'MOM_10D': mom_10d})

    return result_df


def MOM_HIGH_LOW_EFFICIENCY_10D(df):
    """
    Formulation: MOM\_HIGH\_LOW\_EFFICIENCY_{10D,t} = \frac{MOM_{10D,t}}{\frac{1}{10} \sum_{i=0}^{9} \left( \frac{High_{t-i} - Low_{t-i}}{Close_{t-i-1}} \right)}
    Description: 因子本质上是一个风险调整后的动量。该因子在捕捉价格趋势的同时，根据日内波动率进行了调整，旨在过滤掉那些伴随剧烈波动而产生的动量，从而突出那些更具“效率”的价格运动
    Indicator:
        [{'IC': 0.015031907353221411, 'ICIR': 0.10732327750639177, 'RIC': 0.02042996097315334, 'RICIR': 0.11659939725234152
        'The following are analysis results of benchmark return(1day).'
                               risk
        mean               0.000732
        std                0.009844
        annualized_return  0.174329
        information_ratio  1.147896
        max_drawdown      -0.108000
        'The following are analysis results of the excess return without cost(1day).'
                               risk
        mean               0.000484
        std                0.005673
        annualized_return  0.115287
        information_ratio  1.317320
        max_drawdown      -0.101371
        'The following are analysis results of the excess return with cost(1day).'
                               risk
        mean               0.000286
        std                0.005675
        annualized_return  0.068133
        information_ratio  0.778252
        max_drawdown      -0.113503

    """

    # Ensure the index is sorted
    df = df.sort_index()

    # Extract necessary columns
    close_series = df['$close']
    high_series = df['$high']
    low_series = df['$low']

    # Calculate 10-day momentum: (Close_t - Close_{t-10}) / Close_{t-10} * 100
    mom_10d = close_series.groupby(level='instrument', group_keys=False).apply(lambda x: (x - x.shift(10)) / x.shift(10)) * 100
    # mom_10d = close_series.groupby(level='instrument').pct_change(periods=10) * 100

    # Calculate daily high-low range scaled by previous close: (High_{t-i} - Low_{t-i}) / Close_{t-i-1}
    # First, shift close by 1 to get previous day's close
    close_shifted = close_series.groupby(level='instrument').shift(1)
    daily_range_scaled = (high_series - low_series) / close_shifted

    # Calculate 10-day average of daily scaled range: (1/10) * sum_{i=0}^{9} (High_{t-i} - Low_{t-i}) / Close_{t-i-1}
    avg_range_10d = daily_range_scaled.groupby(level='instrument').rolling(window=10, min_periods=10).mean()
    avg_range_10d = avg_range_10d.reset_index(level=0, drop=True)

    # Align indices and combine into DataFrame
    combined = pd.DataFrame({'MOM': mom_10d, 'AVG_RANGE': avg_range_10d})

    # Calculate high-low efficiency momentum: MOM / AVG_RANGE
    # Avoid division by zero or very small numbers
    combined['MOM_HIGH_LOW_EFFICIENCY_10D'] = combined['MOM'] / combined['AVG_RANGE'].replace(0, np.nan)

    # Extract the factor series
    factor_series = combined['MOM_HIGH_LOW_EFFICIENCY_10D']
    factor_series.name = 'MOM_HIGH_LOW_EFFICIENCY_10D'

    # Convert to DataFrame
    result_df = factor_series.to_frame()

    return result_df











