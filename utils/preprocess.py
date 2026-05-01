'''
    数据预处理函数
'''
import numpy as np
import pandas as pd
from typing import Optional, Union
from .stats import WLS
from .logger import LoggerFactory

logger = LoggerFactory.get_logger(__name__)


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
    logger.info('winsorize ...')
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


def _standardize_df(df: pd.DataFrame, method: str, axis: int) -> pd.DataFrame:
    """对 DataFrame 执行标准化计算

    Args:
        df: 输入数据
        method: 'zscore' 或 'minmax'
        axis: 计算轴向

    Returns:
        pd.DataFrame: 标准化后的数据
    """
    if method == 'zscore':
        mean = df.mean(axis=axis)
        std = df.std(axis=axis)
        return df.subtract(mean, axis=1 - axis).div(std, axis=1 - axis)
    if method == 'minmax':
        min_val = df.min(axis=axis)
        max_val = df.max(axis=axis)
        return df.subtract(min_val, axis=1 - axis).div(max_val - min_val, axis=1 - axis)
    raise ValueError(f"method 必须为 'zscore' 或 'minmax'，当前值: {method}")


def standardize(
    data: Union[np.ndarray, pd.Series, pd.DataFrame],
    method: str = 'zscore',
    axis: int = 0,
    level: Optional[Union[int, str]] = None
) -> pd.DataFrame:
    """标准化处理。

    Args:
        data: 输入数据，支持 np.ndarray / pd.Series / pd.DataFrame
        method: 标准化方式
            'zscore': Z-Score标准化，(x - mean) / std
            'minmax': Min-Max归一化，(x - min) / (max - min)，映射到[0, 1]
        axis: 计算轴向，0=按列，1=按行（level 模式下忽略此参数）
        level: 分组索引层级（int 或 str），按该层级 groupby 后分组标准化。
               None 表示不分组，对全量数据标准化

    Returns:
        pd.DataFrame: 标准化后的数据
    """
    logger.info('standardize ...')
    df = _to_dataframe(data)

    # 无分组：全局标准化
    if level is None:
        return _standardize_df(df, method, axis)

    # 有分组：校验索引 → groupby 分组标准化
    _validate_groupby_index(df, level)

    def _group_standardize(group: pd.DataFrame) -> pd.DataFrame:
        return _standardize_df(group, method, axis=0)

    return df.groupby(level=level, group_keys=False).apply(_group_standardize)


def _fillna_df(df: pd.DataFrame, method: str, axis: int) -> pd.DataFrame:
    """对 DataFrame 执行填充操作

    Args:
        df: 输入数据
        method: 填充方式
        axis: 轴向，0=按列，1=按行

    Returns:
        pd.DataFrame: 填充后的数据
    """
    if method == 'zero':
        return df.fillna(0)
    elif method == 'ffill':
        return df.ffill(axis=axis)
    elif method == 'bfill':
        return df.bfill(axis=axis)
    elif method == 'ffill_bfill':
        return df.ffill(axis=axis).bfill(axis=axis)
    elif method == 'mean':
        return df.fillna(df.mean(axis=axis))
    raise ValueError(f"method 必须为 'zero'/'ffill'/'bfill'/'ffill_bfill'/'mean'，当前值: {method}")


def fillna(
    data: Union[pd.Series, pd.DataFrame],
    method: str = 'zero',
    axis: int = 0,
    level: Optional[Union[int, str]] = None,
    method_dict: Optional[dict] = None
) -> Union[pd.Series, pd.DataFrame]:
    """缺失值填充处理。

    Args:
        data: 待填充数据，支持 pd.Series / pd.DataFrame
        method: 填充方式
            'zero': 用 0 填充
            'ffill': 向前填充
            'bfill': 向后填充
            'ffill_bfill': 先向前再向后填充
            'mean': 均值填充
        axis: 填充轴向，0=按列，1=按行（ffill/bfill 时有效）
        level: 分组索引层级（int 或 str），按该层级 groupby 后分组填充均值，仅对 axis=0 生效
               None 表示不分组，对全量数据填充
        method_dict: 对 DataFrame 不同列指定不同填充方式，格式 {col_name: method}。
                     仅 data 为 DataFrame 时有效。未指定的列不做处理（保留原值）。

    Returns:
        Union[pd.Series, pd.DataFrame]: 填充后的数据，类型与输入一致
    """
    logger.info('fillna ...')
    is_series = isinstance(data, pd.Series)

    if is_series:
        if method_dict is not None:
            logger.warning('method_dict 仅在 DataFrame 模式下生效，当前输入为 Series，忽略 method_dict')
        df = data.to_frame()
        result = _fillna_df(df, method, axis) if level is None else df.groupby(level=level, group_keys=False).apply(lambda g: _fillna_df(g, method, axis=0))
        return result.iloc[:, 0]

    # data is DataFrame
    df = data.copy()

    if method_dict:
        # 仅对 method_dict 中指定的列填充，未指定的列不做处理
        for col, col_method in method_dict.items():
            if col in df.columns:
                if level is None:
                    df[col] = _fillna_df(df[[col]], col_method, axis).iloc[:, 0]
                else:
                    _validate_groupby_index(df, level)
                    df[col] = df.groupby(level=level, group_keys=False).apply(
                        lambda g, c=col, cm=col_method: _fillna_df(g[[c]], cm, axis=0).iloc[:, 0]
                    )
            else:
                logger.warning(f"method_dict 中指定的列 '{col}' 在 data 中不存在，已忽略")
        return df

    # 无 method_dict：统一填充
    if level is None:
        return _fillna_df(df, method, axis)

    _validate_groupby_index(df, level)

    def _group_fillna(group: pd.DataFrame) -> pd.DataFrame:
        return _fillna_df(group, method, axis=0)

    return df.groupby(level=level, group_keys=False).apply(_group_fillna)


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
    logger.info('neutralize ...')
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

