"""
    动量因子函数
"""

import pandas as pd


def MOM_10D(df):
    # Load the data
    # df = pd.read_hdf("daily_pv.h5", key="data")

    # Ensure the index is sorted
    df = df.sort_index()

    # Extract the adjusted close price column
    close_series = df['$close']

    # Group by instrument and calculate the 10-day momentum
    # Shift by 10 days to get Close_{t-10}, then compute percentage change
    mom_10d = close_series.groupby(level='instrument').pct_change(periods=10)

    # Create a DataFrame with the result
    result_df = pd.DataFrame({'MOM_10D': mom_10d})

    # Save to HDF5 file
    # result_df.to_hdf('result.h5', key='data', mode='w')

    # Print summary information
    # print(result_df.info())
    # print("\nFirst few rows:")
    # print(result_df.head())
    return result_df


def VW_MOM_5D(df):
    # Load the data
    # df = pd.read_hdf("daily_pv.h5", key="data")

    # Ensure the index is sorted
    df = df.sort_index()

    # Extract necessary columns
    close = df['$close']
    volume = df['$volume']

    # Calculate 5-day price momentum: (Close_t / Close_{t-5}) - 1
    # Shift by 5 periods to get Close_{t-5}
    close_lag5 = close.groupby(level='instrument').shift(5)
    price_momentum = (close / close_lag5) - 1

    # Calculate 5-day average volume
    # Rolling sum over 5 days including current day (t-4 to t)
    volume_sum = volume.groupby(level='instrument').rolling(window=5, min_periods=5).sum()
    # Reset index to match original MultiIndex
    volume_sum = volume_sum.reset_index(level=0, drop=True)
    avg_volume = volume_sum / 5

    # Calculate volume-weighted momentum
    vw_mom = price_momentum * avg_volume

    # Create result DataFrame
    result = pd.DataFrame({'VW_MOM_5D': vw_mom})

    # Save to HDF5 file
    # result.to_hdf('result.h5', key='data', mode='w')

    # Print summary information
    # print(result.info())
    # print("\nFirst few rows:")
    # print(result.head())
    return result








