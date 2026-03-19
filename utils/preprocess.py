'''
    数据预处理函数
'''
import numpy as np
import pandas as pd
from typing import Optional, Union
from utils import WLS


def _compute_bounds(df: pd.DataFrame, method: str, k: float,
                    lower: float, upper: float, axis: int) -> tuple:
    """计算去极值的上下界

    Args:
        df: 输入数据
        method: 去极值方式
        k: 标准差/MAD倍数
        lower: 下分位数
        upper: 上分位数
        axis: 计算轴向

    Returns:
        tuple: (lower_bound, upper_bound)
    """
    if method == 'std':
        mean = df.mean(axis=axis)
        std = df.std(axis=axis)
        lower_bound = mean - k * std
        upper_bound = mean + k * std
    elif method == 'quantile':
        lower_bound = df.quantile(lower, axis=axis)
        upper_bound = df.quantile(upper, axis=axis)
    elif method == 'median':
        median = df.median(axis=axis)
        # 衡量数据与中位数的平均偏离程度
        mad = (df - median).abs().median(axis=axis)
        lower_bound = median - k * 1.4826 * mad
        upper_bound = median + k * 1.4826 * mad
    else:
        raise ValueError(f"method 必须为 'std'/'quantile'/'median'，当前值: {method}")
    return lower_bound, upper_bound


def _apply_bounds(df: pd.DataFrame, lower_bound, upper_bound,
                  axis: int, inclusive: bool) -> pd.DataFrame:
    """根据 inclusive 参数对超出边界的值进行截断或置 NaN

    Args:
        df: 输入数据
        lower_bound: 下界
        upper_bound: 上界
        axis: 计算轴向（clip 时使用 1 - axis）
        inclusive: True 截断到边界值，False 置为 NaN

    Returns:
        pd.DataFrame: 处理后的数据
    """
    clipped = df.clip(lower=lower_bound, upper=upper_bound, axis=1 - axis)
    if inclusive:
        return clipped
    # inclusive=False: 被截断的位置置为 NaN
    return df.where(df == clipped, np.nan)


def _to_dataframe(data: Union[np.ndarray, pd.Series, pd.DataFrame]) -> pd.DataFrame:
    """将输入数据统一转换为 DataFrame

    Args:
        data: 输入数据，支持 np.ndarray / pd.Series / pd.DataFrame

    Returns:
        pd.DataFrame: 转换后的数据（DataFrame 输入会被复制以避免副作用）
    """
    if isinstance(data, np.ndarray):
        return pd.DataFrame(data)
    if isinstance(data, pd.Series):
        return data.to_frame()
    if isinstance(data, pd.DataFrame):
        return data.copy()
    raise TypeError(f'不支持的数据类型: {type(data)}')


def _validate_groupby_index(df: pd.DataFrame, level: Union[int, str]) -> None:
    """校验 DataFrame 索引是否满足 groupby(level=...) 的执行条件

    Args:
        df: 输入数据
        level: 索引层级（int 或 str）

    Raises:
        ValueError: 索引不满足 groupby 条件
    """
    if not isinstance(df.index, pd.MultiIndex):
        raise ValueError(
            f'使用 level 参数需要 MultiIndex，当前索引类型: {type(df.index).__name__}'
        )
    if isinstance(level, int):
        if level < 0 or level >= df.index.nlevels:
            raise ValueError(
                f'level={level} 超出索引层级范围 [0, {df.index.nlevels - 1}]'
            )
    elif isinstance(level, str):
        if level not in df.index.names:
            raise ValueError(
                f"索引中不存在名称 '{level}'，可用名称: {df.index.names}"
            )
    else:
        raise TypeError(f'level 参数类型必须为 int 或 str，当前: {type(level).__name__}')


def winsorize(
    data: Union[np.ndarray, pd.Series, pd.DataFrame],
    method: str = 'std',
    k: float = 3,
    lower: float = 0.01,
    upper: float = 0.99,
    axis: int = 0,
    level: Optional[Union[int, str]] = None,
    inclusive: bool = True
) -> pd.DataFrame:
    """去极值处理。

    Args:
        data: 输入数据，支持 np.ndarray / pd.Series / pd.DataFrame
        method: 去极值方式，'std'(标准差) ,'quantile'(分位数), 'median'(中位数)
        k: 标准差/MAD倍数，method='std'或'median'时生效，默认3
        lower: 下分位数，仅 method='quantile' 时生效
        upper: 上分位数，仅 method='quantile' 时生效
        axis: 计算轴向，0=按列，1=按行（level 模式下忽略此参数）
        level: 分组索引层级（int 或 str），按该层级 groupby 后分组去极值。
               None 表示不分组，对全量数据去极值
        inclusive: True 将超出边界的值截断到边界值，False 将超出边界的值置为 NaN

    Returns:
        pd.DataFrame: 去极值后的数据
    """
    df = _to_dataframe(data)

    # 无分组：全局去极值
    if level is None:
        lower_bound, upper_bound = _compute_bounds(df, method, k, lower, upper, axis)
        return _apply_bounds(df, lower_bound, upper_bound, axis, inclusive)

    # 有分组：校验索引 → groupby 分组去极值
    _validate_groupby_index(df, level)

    def _group_winsorize(group: pd.DataFrame) -> pd.DataFrame:
        lb, ub = _compute_bounds(group, method, k, lower, upper, axis=0)
        return _apply_bounds(group, lb, ub, axis=0, inclusive=inclusive)

    return df.groupby(level=level, group_keys=False).apply(_group_winsorize)


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
    df = _to_dataframe(data)

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


def neutralize(
    y: Union[np.ndarray, pd.Series, pd.DataFrame],
    x: Union[np.ndarray, pd.Series, pd.DataFrame],
    weight: Union[np.ndarray, float] = 1,
    intercept: bool = True
) -> Union[np.ndarray, pd.Series, pd.DataFrame]:
    """中性化处理：用x对y进行线性回归，取残差

    通过WLS回归拟合y，返回残差（实际值 - 预测值），即剔除x影响后的y。

    Args:
        y: 因变量，支持 array/Series/DataFrame
        x: 自变量，支持 array/Series/DataFrame，可以是1列或多列
        weight: 权重，默认等权
        intercept: 是否包含截距项，默认True

    Returns:
        中性化后的残差，维度和类型与y一致
    """
    is_series = isinstance(y, pd.Series)
    is_1d_array = isinstance(y, np.ndarray) and y.ndim == 1
    is_nd_array = isinstance(y, np.ndarray) and y.ndim > 1

    y_df = _to_dataframe(y)
    x_df = _to_dataframe(x)

    # 调用WLS获取残差
    _, _, resid = WLS(y_df, x_df, intercept=intercept, weight=weight, verbose=True)

    # 恢复原始类型
    if is_series:
        return resid.iloc[:, 0]
    if is_1d_array:
        return resid.values.flatten()
    if is_nd_array:
        return resid.values
    return resid

