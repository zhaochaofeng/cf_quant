"""
    波动率因子
"""

import pandas as pd
import numpy as np


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


