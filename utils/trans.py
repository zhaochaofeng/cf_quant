'''
    数据转换函数
'''

import pandas as pd


def calculate_excess_returns(returns_df: pd.DataFrame, benchmark_df: pd.DataFrame) -> pd.DataFrame:
    """计算超额收益：个股收益减去基准收益

    按 instrument 分组，将每只股票的收益率减去相同日期的基准收益率。
    索引结构为 MultiIndex (instrument, datetime)，收益率仅一列。

    Args:
        returns_df: 个股收益，索引为 (instrument, datetime)，单列
        benchmark_df: 基准收益，索引为 (instrument, datetime)，instrument仅有一个取值

    Returns:
        pd.DataFrame: 超额收益，索引与 returns_df 相同
    """
    col = returns_df.columns[0]
    # 提取基准收益为 Series（datetime -> float）
    benchmark_series = benchmark_df[col].droplevel('instrument')
    # 按 datetime 对齐，用 reindex 扩展到 returns_df 的每一行
    dates = returns_df.index.get_level_values('datetime')
    aligned_benchmark = benchmark_series.reindex(dates).values
    return returns_df - aligned_benchmark.reshape(-1, 1)


