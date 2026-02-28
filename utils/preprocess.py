'''
    数据预处理函数
'''
import numpy as np
import pandas as pd
from typing import Union


def winsorize(
    data: Union[np.ndarray, pd.Series, pd.DataFrame],
    method: str = 'std',
    k: float = 3,
    lower: float = 0.01,
    upper: float = 0.99,
    axis: int = 0
) -> pd.DataFrame:
    """去极值处理，将超出边界的值截断到边界值。

    Args:
        data: 输入数据，支持 np.ndarray / pd.Series / pd.DataFrame
        method: 去极值方式，'std'(标准差) 或 'quantile'(分位数)
        k: 标准差倍数，仅 method='std' 时生效，默认3倍标准差
        lower: 下分位数，仅 method='quantile' 时生效
        upper: 上分位数，仅 method='quantile' 时生效
        axis: 计算轴向，0=按列，1=按行

    Returns:
        pd.DataFrame: 去极值后的数据
    """
    if isinstance(data, np.ndarray):
        df = pd.DataFrame(data)
    elif isinstance(data, pd.Series):
        df = data.to_frame()
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise TypeError(f'不支持的数据类型: {type(data)}')

    if method == 'std':
        mean = df.mean(axis=axis)
        std = df.std(axis=axis)
        lower_bound = mean - k * std
        upper_bound = mean + k * std
    elif method == 'quantile':
        lower_bound = df.quantile(lower, axis=axis)
        upper_bound = df.quantile(upper, axis=axis)
    else:
        raise ValueError(f"method 必须为 'std' 或 'quantile'，当前值: {method}")

    return df.clip(lower=lower_bound, upper=upper_bound, axis=1 - axis)


def standardize(
    data: Union[np.ndarray, pd.Series, pd.DataFrame],
    method: str = 'zscore',
    axis: int = 0
) -> pd.DataFrame:
    """标准化处理。

    Args:
        data: 输入数据，支持 np.ndarray / pd.Series / pd.DataFrame
        method: 标准化方式
            'zscore': Z-Score标准化，(x - mean) / std
            'minmax': Min-Max归一化，(x - min) / (max - min)，映射到[0, 1]
        axis: 计算轴向，0=按列，1=按行

    Returns:
        pd.DataFrame: 标准化后的数据
    """
    if isinstance(data, np.ndarray):
        df = pd.DataFrame(data)
    elif isinstance(data, pd.Series):
        df = data.to_frame()
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise TypeError(f'不支持的数据类型: {type(data)}')

    if method == 'zscore':
        mean = df.mean(axis=axis)
        std = df.std(axis=axis)
        return df.subtract(mean, axis=1 - axis).div(std, axis=1 - axis)
    elif method == 'minmax':
        min_val = df.min(axis=axis)
        max_val = df.max(axis=axis)
        return df.subtract(min_val, axis=1 - axis).div(max_val - min_val, axis=1 - axis)
    else:
        raise ValueError(f"method 必须为 'zscore' 或 'minmax'，当前值: {method}")


