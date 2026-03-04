"""因子计算工具函数"""
import pandas as pd
import numpy as np
import qlib
from qlib.data import D
from utils import WLS, multiprocessing_wrapper


provider_uri = '~/.qlib/qlib_data/custom_data_hfq'

BENCHMARK = 'SH000300'

# 缺失值标记，用于 rolling 计算中处理 NaN
SENTINEL = 1e10


def get_qlib_data(instruments: [str, list], fields: list, start_date: str, end_date: str) -> pd.DataFrame:
    """ 获取qlib 字段数据 """
    qlib.init(provider_uri=provider_uri)
    if isinstance(instruments, str):
        instruments = D.instruments(market=instruments)
    df = D.features(instruments=instruments, fields=fields, start_time=start_date, end_time=end_date)
    return df


def get_benchmark_ret(start_date: str, end_date: str) -> pd.Series:
    """ 获取 Benchmark 收益率"""
    try:
        benchmark_df = get_qlib_data(
            instruments=[BENCHMARK], fields=['$change'],
            start_date=start_date, end_date=end_date,
        )
        benchmark_ret = benchmark_df['$change']
        if benchmark_ret.index.nlevels > 1:
            benchmark_ret = benchmark_ret.droplevel('instrument')
    except Exception as e:
        raise Exception(f"获取 Benchmark ret 数据失败: {e}")
    return benchmark_ret


def get_exp_weight(window, half_life):
    """ 半衰期权重 . 如：
    [0.04956612 0.05693652 0.06540289 0.07512819 0.08629962 0.09913224
        0.11387304 0.13080577 0.15025637 0.17259925]
    """
    exp_wt = np.asarray([0.5 ** (1 / half_life)] * window) ** np.arange(window)
    return exp_wt[::-1] / np.sum(exp_wt)


def get_stock_list_info(instruments: list, start_date: str, end_date: str) -> tuple:
    """获取股票上市/退市日期信息

    Args:
        instruments: list, 股票代码列表
        start_date: str, 开始日期
        end_date: str, 结束日期

    Returns:
        tuple: (list_date_map, delist_date_map)
            - list_date_map: dict, instrument -> pd.Timestamp
            - delist_date_map: dict, instrument -> pd.Timestamp or pd.NaT
    """
    # 待修复：list_date/delist_date 传入qlib后数据改变。如 SH600188 的 list_date从1998-07-01变为19980700
    return {}, {}
    '''
    df = get_qlib_data(
        instruments=instruments,
        fields=['$list_date', '$delist_date'],
        start_date=start_date,
        end_date=end_date,
    )

    # df['$list_date'] = pd.to_datetime(df['$list_date'], format='%Y%m%d').dt.strftime('%Y-%m-%d')
    # df['$delist_date'] = pd.to_datetime(df['$delist_date'], format='%Y%m%d').dt.strftime('%Y-%m-%d')

    list_date_map = {}
    delist_date_map = {}
    for inst in df.index.get_level_values('instrument').unique():
        inst_data = df.loc[inst]
        list_val = inst_data['$list_date'].dropna()
        delist_val = inst_data['$delist_date'].dropna()
        list_date_map[inst] = (
            pd.to_datetime(list_val.iloc[0]) if len(list_val) > 0 else pd.NaT
        )
        delist_date_map[inst] = (
            pd.to_datetime(delist_val.iloc[0]) if len(delist_val) > 0 else pd.NaT
        )

    return list_date_map, delist_date_map
    '''


def capm_regress(stock_returns, window=504, half_life=252, benchmark=BENCHMARK, num_worker=1):
    """CAPM滚动回归

    使用滚动窗口WLS回归估计每只股票相对于基准指数的beta、alpha和残差波动率。

    Args:
        stock_returns: pd.Series, MultiIndex (instrument, datetime) - 股票日收益率
        window: int, 滚动窗口大小（默认504个交易日，约2年）
        half_life: int, 半衰期（默认252个交易日，约1年）
        benchmark: str, 基准指数代码
        num_worker: int, 并行计算的进程数，通常设置为1，设置>=2可能执行速度更慢

    Returns:
        tuple: (beta, alpha, sigma)
            - beta: pd.Series, MultiIndex (instrument, datetime)
            - alpha: pd.Series, MultiIndex (instrument, datetime)
            - sigma: pd.Series, MultiIndex (instrument, datetime)
    """

    instruments = stock_returns.index.get_level_values('instrument').unique().tolist()
    # [:10] 用于日期截断。'2025-01-02 00:00:00' -> '2025-01-02'
    start_date = str(stock_returns.index.get_level_values('datetime').min())[:10]
    end_date = str(stock_returns.index.get_level_values('datetime').max())[:10]

    # 获取基准收益率
    try:
        benchmark_df = get_qlib_data(
            instruments=[benchmark], fields=['$change'],
            start_date=start_date, end_date=end_date,
        )
        benchmark_ret = benchmark_df['$change']
        if benchmark_ret.index.nlevels > 1:
            benchmark_ret = benchmark_ret.droplevel('instrument')
        
        # 检查基准数据是否为空
        if len(benchmark_ret.dropna()) == 0:
            print(f"警告: 基准指数 {benchmark} 在指定时间范围内无有效数据")
            # 尝试扩展时间范围
            extended_start = str(pd.to_datetime(start_date) - pd.Timedelta(days=30))[:10]
            extended_end = str(pd.to_datetime(end_date) + pd.Timedelta(days=30))[:10]
            print(f"尝试扩展时间范围: {extended_start} 到 {extended_end}")
            benchmark_df = get_qlib_data(
                instruments=[benchmark], fields=['$change'],
                start_date=extended_start, end_date=extended_end,
            )
            benchmark_ret = benchmark_df['$change']
            if benchmark_ret.index.nlevels > 1:
                benchmark_ret = benchmark_ret.droplevel('instrument')
            
            if len(benchmark_ret.dropna()) == 0:
                raise ValueError(f"基准指数 {benchmark} 无有效数据")
    except Exception as e:
        print(f"获取基准数据失败: {e}")
        raise

    # dict. 获取上市/退市日期
    list_date_map, delist_date_map = get_stock_list_info(
        instruments=instruments, start_date=start_date, end_date=end_date,
    )

    # 调用滚动回归
    beta, alpha, sigma = rolling_regress(
        y=stock_returns, x=benchmark_ret,
        window=window, half_life=half_life,
        list_date_map=list_date_map, delist_date_map=delist_date_map,
        num_worker=num_worker,
    )

    return beta, alpha, sigma


def rolling_regress(y, x, window=504, half_life=252, intercept=True, fill_na=0,
                     list_date_map=None, delist_date_map=None, num_worker=1):
    """滚动加权最小二乘回归（CAPM回归）

    对股票收益率与基准收益率进行滚动窗口的加权最小二乘回归，
    用于估计 CAPM 模型中的 beta、alpha 和残差波动率(sigma)。

    Args:
        y: pd.Series, MultiIndex (instrument, datetime) - 股票日收益率
        x: pd.Series, index=datetime - 基准收益率（单索引）
        window: int, 滚动窗口大小
        half_life: int or None, 半衰期权重。None 表示等权
        intercept: bool, 是否包含截距项
        fill_na: int/float or str, 填充NaN的值，支持数值或填充方法(如'ffill')
        list_date_map: dict, instrument -> pd.Timestamp, 上市日期映射
        delist_date_map: dict, instrument -> pd.Timestamp, 退市日期映射
        num_worker: int, 并行计算的进程数，通常设置为1，设置>=2可能执行速度更慢

    Returns:
        tuple: (beta, alpha, sigma)
            - beta: pd.Series, MultiIndex (instrument, datetime)
            - alpha: pd.Series, MultiIndex (instrument, datetime)
            - sigma: pd.Series, MultiIndex (instrument, datetime)
    """
    # 转为宽表格式: index=datetime, columns=instruments
    y_wide = y.unstack(level='instrument')
    stocks = y_wide.columns.tolist()

    # 处理 x 的索引（如有 MultiIndex 则去掉 instrument 层）
    if isinstance(x.index, pd.MultiIndex):
        x = x.droplevel('instrument')

    # 对齐时间，并排序
    common_dates = y_wide.index.intersection(x.index).sort_values()
    y_wide = y_wide.loc[common_dates]
    x = x.loc[common_dates]

    # 计算半衰期权重
    if half_life:
        weight = get_exp_weight(window, half_life)
    else:
        weight = 1

    # 填充参数
    fill_args = {'method': fill_na} if isinstance(fill_na, str) else {'value': fill_na}

    # 找到 x 第一个非空值的索引，截断之前的数据
    x_valid = x.dropna()
    if len(x_valid) == 0:
        raise ValueError("基准收益率全部为空")
    start_date = x_valid.index[0]
    y_wide = y_wide.loc[start_date:]
    x = x.loc[start_date:]

    n = len(y_wide)
    
    # 准备并行计算的参数列表
    func_calls = []
    for i in range(n - window + 1):
        window_y = y_wide.iloc[i:i + window]
        window_x_vals = x.iloc[i:i + window].values
        
        window_sdate = y_wide.index[i]
        window_edate = y_wide.index[i + window - 1]
        
        # 根据上市/退市日期过滤股票
        if list_date_map is not None and delist_date_map is not None\
                and len(list_date_map) > 0 and len(delist_date_map) > 0:
            stks_to_regress = sorted([
                s for s in stocks
                if s in list_date_map
                and pd.notna(list_date_map.get(s))
                and list_date_map[s] <= window_sdate
                and (pd.isna(delist_date_map.get(s))
                     or delist_date_map[s] >= window_edate)
            ])
        else:
            stks_to_regress = stocks
        
        if not stks_to_regress:
            continue
        
        # 将单次回归计算封装为函数调用
        func_calls.append((
            _single_regression,
            (window_y, window_x_vals, stks_to_regress, fill_args, weight, intercept, window_edate)
        ))
    
    # 并行执行所有回归计算
    results = multiprocessing_wrapper(func_calls, n=num_worker)
    
    # 解包结果
    beta_list, alpha_list, sigma_list = [], [], []
    for beta_series, alpha_series, sigma_series in results:
        beta_list.append(beta_series)
        alpha_list.append(alpha_series)
        sigma_list.append(sigma_series)

    if not beta_list:
        empty = pd.Series(dtype=float)
        empty.index = pd.MultiIndex.from_tuples([], names=['instrument', 'datetime'])
        return empty, empty, empty

    # 合并为 DataFrame (index=datetime, columns=instruments)
    beta_wide = pd.DataFrame(beta_list).reindex(columns=stocks)
    alpha_wide = pd.DataFrame(alpha_list).reindex(columns=stocks)
    sigma_wide = pd.DataFrame(sigma_list).reindex(columns=stocks)

    # 转回 MultiIndex (instrument, datetime) 格式
    def _to_multiindex_series(df):
        s = df.stack()
        s.index.names = ['datetime', 'instrument']
        return s.reorder_levels(['instrument', 'datetime']).sort_index()

    return (
        _to_multiindex_series(beta_wide),
        _to_multiindex_series(alpha_wide),
        _to_multiindex_series(sigma_wide),
    )



def _single_regression(window_y, window_x_vals, stks_to_regress, fill_args, weight, intercept, window_edate):
    """单次 WLS 回归计算（用于并行化）
    
    Args:
        window_y: pd.DataFrame, 窗口期内的股票收益率
        window_x_vals: np.ndarray, 窗口期内的基准收益率
        stks_to_regress: list, 待回归的股票列表
        fill_args: dict, NaN 填充参数
        weight: np.ndarray or int, 权重
        intercept: bool, 是否包含截距项
        window_edate: pd.Timestamp, 窗口结束日期
    
    Returns:
        tuple: (beta_series, alpha_series, sigma_series)
    """
    rolling_y = window_y[stks_to_regress].fillna(**fill_args)
    
    # WLS 回归
    b, a, resid = WLS(
        rolling_y.values, pd.DataFrame(window_x_vals),
        intercept=intercept, weight=weight, verbose=True
    )
    
    # 残差标准差
    vol = np.std(resid.values, axis=0)
    
    beta_series = pd.Series(b.values.flatten(), index=stks_to_regress, name=window_edate)
    alpha_series = pd.Series(a.values, index=stks_to_regress, name=window_edate)
    sigma_series = pd.Series(vol, index=stks_to_regress, name=window_edate)
    
    return beta_series, alpha_series, sigma_series


def cal_cmra(series, months=12, days_per_month=21, sentinel=SENTINEL):
    """计算 Cumulative Return Range over Months (CMRA) 因子
    
    支持过滤 sentinel 标记的缺失值。
    
    Args:
        series: array-like, 股票收益率序列（可能包含 sentinel 标记的缺失值）
        months: int, 计算月份数，默认12个月
        days_per_month: int, 每月交易日数，默认21天
        sentinel: scalar, 缺失值标记，默认 SENTINEL
    
    Returns:
        float: CMRA值
    """
    # 过滤 sentinel 缺失值
    series = np.array(series)
    valid_mask = series != sentinel
    valid_series = series[valid_mask]
    
    # 数据不足时返回 NaN
    if len(valid_series) < days_per_month:
        return np.nan
    
    # 计算每个月的累计收益 Z_t
    z = []
    for i in range(1, months + 1):
        end_idx = len(valid_series)
        start_idx = max(0, end_idx - i * days_per_month)
        month_sum = valid_series[start_idx:end_idx].sum()
        z.append(month_sum)
    
    z = sorted(z)
    return z[-1] - z[0]


def weighted_std(series, weights):
    """加权标准差
    
    Args:
        series: array-like, 数据序列
        weights: array-like, 权重（已归一化）
    
    Returns:
        float: 加权标准差
    """
    return np.sqrt(np.sum((series - np.mean(series)) ** 2 * weights))


def weighted_func(func, series, weights):
    """加权函数分发：std走加权标准差，其余走 func(series * weights)
    
    Args:
        func: numpy函数，如 np.std, np.sum 等
        series: array-like, 有效数据序列
        weights: array-like, 权重
    
    Returns:
        加权计算结果
    """
    weights = weights / np.sum(weights)
    if func.__name__ == 'std':
        return weighted_std(series, weights)
    else:
        return func(series * weights)


def nanfunc(series, func, sentinel=SENTINEL, weights=None):
    """过滤 sentinel 缺失值后执行函数计算
    
    Args:
        series: array-like, 输入序列
        func: callable, numpy函数
        sentinel: scalar, 缺失值标记
        weights: array-like, 权重（可选）
    
    Returns:
        函数计算结果
    """
    valid_idx = np.argwhere(series != sentinel)
    if weights is not None:
        return weighted_func(func, series[valid_idx], weights=weights[valid_idx])
    else:
        return func(series[valid_idx])


def rolling_with_func(series, window, half_life=None, func_name='std', weights=None):
    """通用滚动函数，通过 func_name 指定 numpy 聚合函数

    Args:
        series: pd.Series, 输入序列
        window: int, 滚动窗口大小
        half_life: int, 半衰期（可选）
        func_name: str, numpy函数名（'std', 'sum', 'mean', 'max', 'min'等）
        weights: array-like, 自定义权重（可选）
    
    Returns:
        pd.Series: 计算结果序列
    """
    func = getattr(np, func_name, None)
    if func is None:
        raise AttributeError(
            f"Search func:{func_name} from numpy failed, "
            f"only numpy ufunc is supported currently, please retry."
        )
    
    # 将 NaN 替换为 SENTINEL
    series = series.where(pd.notnull(series), SENTINEL)
    
    if half_life or (weights is not None):
        exp_wt = get_exp_weight(window, half_life) if half_life else weights
        args = (func, SENTINEL, exp_wt)
    else:
        args = (func, SENTINEL)
    # raw 表示传入 nanfunc 函数的为 array 数据，用于加速计算
    return series.rolling(window=window).apply(nanfunc, args=args, raw=True)


def cal_liquidity(series, days_per_month=21, sentinel=SENTINEL):
    """计算流动性因子的对数换手率均值
    
    用于 STOM、STOQ、STOA 等流动性因子计算。
    
    Args:
        series: array-like, 换手率序列（可能包含 sentinel 标记的缺失值）
        days_per_month: int, 每月交易日数，用于计算月份数
        sentinel: scalar, 缺失值标记，默认 SENTINEL
    
    Returns:
        float: ln(换手率均值)，如果数据不足返回 np.nan
    """
    # 过滤 sentinel 缺失值
    series = np.array(series)
    valid_mask = series != sentinel
    valid_series = series[valid_mask]
    
    # 数据不足时返回 NaN（至少需要一个完整月的数据）
    if len(valid_series) < days_per_month:
        return np.nan
    
    # 计算月份数
    freq = len(valid_series) // days_per_month
    if freq == 0:
        return np.nan
    
    # 计算对数换手率均值
    res = np.log(np.nansum(valid_series) / freq)
    
    # 检查结果是否有效
    if np.isinf(res) or np.isnan(res):
        return np.nan
    
    return res


def calc_seasonality(group, nyears=5, value_col='$change'):
    """计算单个股票的季节性因子
    
    用于 SEASON 因子计算，计算过去 nyears 年同月份的超额收益均值。
    
    Args:
        group: pd.DataFrame, 单个股票的数据，包含 'datetime', 'month' 和 value_col 列
        nyears: int, 计算历史均值的年数，默认5年
        value_col: str, 收益率列名，默认 '$change'
    
    Returns:
        pd.DataFrame: 包含 'instrument', 'datetime', 'SEASON' 列的 DataFrame
    """
    group = group.sort_values('datetime')
    result = []
    
    for idx, row in group.iterrows():
        current_month = row['month']
        current_date = row['datetime']
        
        # 获取过去同月份的数据（不包括当前月，因为当前月是被预测的对象）
        historical = group[
            (group['month'] == current_month) & 
            (group['datetime'] < current_date)
        ]
        
        # 取最近 nyears 年的均值
        if len(historical) > 0:
            season_val = historical.tail(nyears)[value_col].mean()
        else:
            season_val = np.nan
        
        result.append({
            'instrument': row['instrument'],
            'datetime': current_date,
            'SEASON': season_val
        })
    
    return pd.DataFrame(result)


if __name__ == '__main__':
    print(get_exp_weight(window=10, half_life=5))




