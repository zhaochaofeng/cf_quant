"""因子计算工具函数"""
import pandas as pd
import numpy as np
# from .config import BENCHMARK
from config import BENCHMARK_CONFIG, PROVIDER_URI
from utils import WLS, multiprocessing_wrapper
from utils import LoggerFactory

logger = LoggerFactory.get_logger(__name__)


# capm_regress 结果缓存：避免 BETA/HSIGMA/HALPHA 重复计算同一回归
_capm_cache: dict = {}
# 基准收益率 & 上市信息缓存
_benchmark_cache: dict = {}
_stock_info_cache: dict = {}

# 缺失值标记，用于 rolling 计算中处理 NaN
SENTINEL = 1e10


def get_qlib_data(instruments: [str, list], fields: list, start_date: str, end_date: str) -> pd.DataFrame:
    """获取 qlib 字段数据"""
    from qlib.data import D  # 延迟导入
    if isinstance(instruments, str):
        instruments = D.instruments(market=instruments)
    df = D.features(instruments=instruments, fields=fields, start_time=start_date, end_time=end_date)
    return df


def get_benchmark_ret(start_date: str, end_date: str) -> pd.Series:
    """获取 Benchmark 收益率（带缓存）"""
    bm_key = (BENCHMARK_CONFIG['BENCHMARK'], start_date, end_date)
    if bm_key in _benchmark_cache:
        return _benchmark_cache[bm_key]
    try:
        benchmark_df = get_qlib_data(
            instruments=[BENCHMARK_CONFIG['BENCHMARK']], fields=['$change'],
            start_date=start_date, end_date=end_date,
        )
        benchmark_ret = benchmark_df['$change']
        if benchmark_ret.index.nlevels > 1:
            benchmark_ret = benchmark_ret.droplevel('instrument')
    except Exception as e:
        raise Exception(f"获取 Benchmark ret 数据失败: {e}")
    _benchmark_cache[bm_key] = benchmark_ret
    return benchmark_ret


def get_exp_weight(window, half_life):
    """ 半衰期权重 . 如：
    [0.04956612 0.05693652 0.06540289 0.07512819 0.08629962 0.09913224
        0.11387304 0.13080577 0.15025637 0.17259925]
    """
    exp_wt = np.asarray([0.5 ** (1 / half_life)] * window) ** np.arange(window)
    return exp_wt[::-1] / np.sum(exp_wt)


def get_mysql_data(instrument: [], table: str, fields: [], start_date: str, end_date: str) -> pd.DataFrame:
    from utils import sql_engine
    engine = sql_engine()
    fields += ['qlib_code']
    sql = """ SELECT {} FROM {} WHERE day >= '{}' AND day <= '{}' AND qlib_code in ({}) """.\
        format(','.join(fields), table, start_date, end_date, ','.join([f"'{i}'" for i in instrument]))
    df = pd.read_sql(sql, engine)
    df.set_index('qlib_code', inplace=True)
    df.index.name = 'instrument'
    return df


def get_stock_list_info(instruments: list, date: str) -> tuple:
    """获取股票上市/退市日期信息

    Args:
        instruments: list, 股票代码列表
        date: str, 查询日期

    Returns:
        tuple: (list_date_map, delist_date_map)
            - list_date_map: dict, instrument -> pd.Timestamp
            - delist_date_map: dict, instrument -> pd.Timestamp or pd.NaT
    """
    df = get_mysql_data(instruments, 'stock_info_ts', ['list_date', 'delist_date'], date, date)
    if df.empty:
        raise Exception('无法获取股票上市/退市日期信息!!!')

    # 统一转换日期格式
    df['list_date'] = pd.to_datetime(df['list_date'])
    df['delist_date'] = pd.to_datetime(df['delist_date'])

    # 按股票分组取第一条（去重），并直接转换为字典
    df_unique = df.groupby(level='instrument').first()
    list_date_map = df_unique['list_date'].to_dict()
    delist_date_map = df_unique['delist_date'].to_dict()

    if not list_date_map or not delist_date_map:
        raise Exception('get_stock_list_info: 无法获取股票上市/退市日期信息')

    return list_date_map, delist_date_map


def capm_regress(stock_returns, window=504, half_life=252, benchmark=BENCHMARK_CONFIG['BENCHMARK'], num_worker=1):
    """CAPM滚动回归

    使用滚动窗口WLS回归估计每只股票相对于基准指数的 beta、alpha和残差波动率。
    内置缓存：相同 (window, half_life, benchmark, date_range) 参数不会重复计算。

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
    # 在多进程环境中确保 qlib 已初始化
    try:
        from qlib.data import D
        # 尝试使用 D 来检查 qlib 是否已初始化
        _ = D.instruments(market=BENCHMARK_CONFIG['market'])
    except Exception:
        # qlib 未初始化，尝试初始化
        try:
            import qlib
            from utils.qlib_ops import PTTM
            qlib.init(
                provider_uri=PROVIDER_URI,
                custom_ops=[PTTM],
            )
            logger.info('子进程中 qlib 初始化完成')
        except Exception as e:
            err_msg = f'子进程中 qlib 初始化失败: {e}'
            logger.error(err_msg)
            raise Exception(err_msg)

    instruments = stock_returns.index.get_level_values('instrument').unique().tolist()
    # [:10] 用于日期截断。'2025-01-02 00:00:00' -> '2025-01-02'
    start_date = str(stock_returns.index.get_level_values('datetime').min())[:10]
    end_date = str(stock_returns.index.get_level_values('datetime').max())[:10]

    # 检查缓存（BETA/HSIGMA/HALPHA 使用相同参数，避免重复计算）
    cache_key = (window, half_life, benchmark, start_date, end_date, len(instruments))
    if cache_key in _capm_cache:
        logger.info(f'capm_regress 命中缓存: window={window}, half_life={half_life}, '
                    f'benchmark={benchmark}, start_date={start_date}, '
                    f'end_date={end_date}, num_instruments={len(instruments)}')
        return _capm_cache[cache_key]

    # 获取基准收益率（缓存，避免长计算后 qlib 连接超时）
    bm_key = (benchmark, start_date, end_date)
    if bm_key in _benchmark_cache:
        benchmark_ret = _benchmark_cache[bm_key]
        logger.info(f'benchmark_ret 命中缓存: benchmark={benchmark}, start_date={start_date}, end_date={end_date}')
    else:
        try:
            benchmark_df = get_qlib_data(
                instruments=[benchmark], fields=['$change'],
                start_date=start_date, end_date=end_date,
            )
            benchmark_ret = benchmark_df['$change']
            if benchmark_ret.index.nlevels > 1:
                benchmark_ret = benchmark_ret.droplevel('instrument')
            if len(benchmark_ret.dropna()) == 0:
                err_msg = f"基准指数 {benchmark} 在指定时间范围内无有效数据"
                logger.error(err_msg)
                raise Exception(err_msg)
            _benchmark_cache[bm_key] = benchmark_ret
            logger.info(f'benchmark_ret 计算完成并缓存: benchmark={benchmark}, start_date={start_date}, end_date={end_date}')
        except Exception as e:
            logger.error(f"获取基准数据失败: {e}")
            raise

    # 获取上市/退市日期（缓存）
    info_key = (end_date, len(instruments))
    if info_key in _stock_info_cache:
        list_date_map, delist_date_map = _stock_info_cache[info_key]
        logger.info(f'stock_info 命中缓存: end_date={end_date}, num_instruments={len(instruments)}')
    else:
        list_date_map, delist_date_map = get_stock_list_info(
            instruments=instruments, date=end_date,
        )
        _stock_info_cache[info_key] = (list_date_map, delist_date_map)
        logger.info(f'stock_info 计算完成并缓存: end_date={end_date}, num_instruments={len(instruments)}, '
                    f'list_date_map len={len(list_date_map)}, delist_date_map len={len(delist_date_map)}')
    # 调用滚动回归
    beta, alpha, sigma = rolling_regress(
        y=stock_returns, x=benchmark_ret,
        window=window, half_life=half_life,
        list_date_map=list_date_map, delist_date_map=delist_date_map,
        num_worker=num_worker,
    )

    result = (beta, alpha, sigma)
    _capm_cache[cache_key] = result
    logger.info(f'capm_regress 计算完成并缓存: window={window}, half_life={half_life}, benchmark={benchmark}, '
                f'start_date={start_date}, end_date={end_date}, num_instruments={len(instruments)}')

    return result


def rolling_regress(y, x, window=504, half_life=252, intercept=True,
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
        list_date_map: dict, instrument -> pd.Timestamp, 上市日期映射
        delist_date_map: dict, instrument -> pd.Timestamp, 退市日期映射
        num_worker: int, 并行计算的进程数，通常设置为1，设置>=2可能执行速度更慢

    Returns:
        tuple: (beta, alpha, sigma)
            - beta: pd.Series, MultiIndex (instrument, datetime)
            - alpha: pd.Series, MultiIndex (instrument, datetime)
            - sigma: pd.Series, MultiIndex (instrument, datetime)
    """
    logger.info('rolling_regress ...')
    # 转为宽表格式: index=datetime, columns=instruments
    y_wide = y.unstack(level='instrument')
    logger.info('y_wide shape: {}, x shape: {}, window: {}, half_life: {}'.format(y_wide.shape, x.shape, window, half_life))

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

    n = len(y_wide)
    if n < window:
        raise ValueError(f"数据长度({n})小于窗口大小({window})")

    # 准备并行计算的参数列表
    func_calls = []
    # 从下标 window 开始遍历到 n-1
    for i in range(n - window + 1):
        window_y = y_wide.iloc[i:i + window]  # DataFrame
        window_x_vals = x.iloc[i:i + window].values  # ndarray

        window_sdate = y_wide.index[i]   # 窗口开始日期
        window_edate = y_wide.index[i + window - 1]  # 窗口结束日期

        # 根据上市/退市日期过滤股票
        if list_date_map and delist_date_map:
            # 排除在 [window_sdate, window_edate] 之间上市或退市的股票
            stks_to_regress = sorted([
                s for s in stocks
                if s in list_date_map
                and pd.notna(list_date_map.get(s))
                and list_date_map[s] <= window_sdate
                and (pd.isna(delist_date_map.get(s)) or delist_date_map[s] >= window_edate)
            ])
        else:
            stks_to_regress = stocks

        if not stks_to_regress:
            logger.warning('No stocks to regress in this window: [{}, {}]'.format(window_sdate, window_edate))
            continue

        # 将单次回归计算封装为函数调用
        func_calls.append((
            _single_regression,
            (window_y, window_x_vals, stks_to_regress, weight, intercept, window_edate)
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
    '''
                SZ000001  SZ000002
    2020-02-03  1.227315  1.145945
    '''
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


def _single_regression(window_y, window_x_vals, stks_to_regress, weight, intercept, window_edate):
    """单次 WLS 回归计算（用于并行化）

    Args:
        window_y: pd.DataFrame, 窗口期内的股票收益率. 列: instrument
        window_x_vals: np.ndarray, 窗口期内的基准收益率
        stks_to_regress: list, 待回归的股票列表
        weight: np.ndarray or int, 权重
        intercept: bool, 是否包含截距项
        window_edate: pd.Timestamp, 窗口结束日期

    Returns:
        tuple: (beta_series, alpha_series, sigma_series)
    """
    rolling_y = window_y[stks_to_regress]
    min_valid_ratio = 0.8

    # 初始化结果字典
    beta_dict, alpha_dict, sigma_dict = {}, {}, {}

    # 逐只股票处理 NaN
    for stock in stks_to_regress:
        y_stock = rolling_y[stock]

        # 找到该股票有效数据的掩码
        valid_mask = y_stock.notna().values
        valid_count = valid_mask.sum()
        total_count = len(y_stock)

        # 检查有效数据是否足够
        if valid_count < total_count * min_valid_ratio:
            # 有效数据不足，该股票返回 NaN
            beta_dict[stock] = np.nan
            alpha_dict[stock] = np.nan
            sigma_dict[stock] = np.nan
            continue

        # 提取有效数据
        y_valid = y_stock[valid_mask].values
        x_valid = window_x_vals[valid_mask]

        # 调整权重（如果是数组）
        if isinstance(weight, np.ndarray):
            w_valid = weight[valid_mask]
        else:
            w_valid = weight

        # WLS 回归（单只股票）
        try:
            b, a, resid = WLS(
                y_valid.reshape(-1, 1), pd.DataFrame(x_valid),
                intercept=intercept, weight=w_valid, verbose=True, backend='numpy'
            )

            # 残差标准差（使用样本标准差 ddof=1）
            vol = np.std(resid.values, ddof=1)

            beta_dict[stock] = b.values.flatten()[0]
            alpha_dict[stock] = a if a is None or np.isscalar(a) else a.values[0]
            sigma_dict[stock] = vol
        except Exception:
            # 回归失败，返回 NaN
            beta_dict[stock] = np.nan
            alpha_dict[stock] = np.nan
            sigma_dict[stock] = np.nan

    # 按照 stks_to_regress 的顺序创建 Series
    '''
    SZ000001    1.227315
    SZ000002    1.145945
    Name: 2020-02-03 00:00:00, dtype: float64
    '''
    beta_series = pd.Series([beta_dict[s] for s in stks_to_regress], index=stks_to_regress, name=window_edate)
    alpha_series = pd.Series([alpha_dict[s] for s in stks_to_regress], index=stks_to_regress, name=window_edate)
    sigma_series = pd.Series([sigma_dict[s] for s in stks_to_regress], index=stks_to_regress, name=window_edate)

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


def cal_liquidity(series, days_per_month=21, sentinel=SENTINEL, min_valid_ratio=0.8):
    """计算流动性因子的对数换手率均值（支持缺失值和比例月份计算）
    
    用于 STOM、STOQ、STOA 等流动性因子计算。
    
    Args:
        series: array-like, 换手率序列（可能包含 sentinel 标记的缺失值）
        days_per_month: int, 每月交易日数，用于计算月份数
        sentinel: scalar, 缺失值标记，默认 SENTINEL
        min_valid_ratio: float, 最小有效值比例（默认0.8，允许20%缺失）
    
    Returns:
        float: ln(换手率均值)，如果数据不足返回 np.nan
    """
    series = np.array(series)
    valid_mask = series != sentinel
    valid_series = series[valid_mask]
    
    # 检查有效值比例是否足够
    if len(valid_series) < len(series) * min_valid_ratio:
        return np.nan
    
    # 按实际有效天数计算等效月份数（浮点除法）
    n_months = len(valid_series) / days_per_month
    if n_months == 0:
        return np.nan
    
    # 计算对数换手率均值
    res = np.log(np.nansum(valid_series) / n_months)
    
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


def get_annual_data(series, field_name: str):
    """使用 PRef 算子获取年度年报数据

    根据 series 的 datetime 索引映射财年，使用 PRef 批量查询年报数据，
    返回 (instrument, year) 粒度的年度数据。

    财年规则：
        1-4月 → 使用 year-2 年报（如 2024-03 用 202204 年报）
        5-12月 → 使用 year-1 年报（如 2024-06 用 202304 年报）

    Args:
        series: pd.Series, MultiIndex (instrument, datetime)
                仅用于获取 instrument 和 datetime 索引
        field_name: str, 字段名（如 'revenue_q'）

    Returns:
        pd.Series: MultiIndex (instrument, year)，年度年报数据
                   year 为财报年份（fiscal_year // 100）
    """
    # ========== 输入验证 ==========
    if not isinstance(series, pd.Series):
        raise TypeError(f'series must be pd.Series, got {type(series)}')

    if series.empty:
        logger.warning(f'get_annual_data: empty series for {field_name}')
        return series

    if not isinstance(series.index, pd.MultiIndex):
        raise ValueError(f'series must have MultiIndex, got {type(series.index)}')

    required_levels = ['instrument', 'datetime']
    if not all(level in series.index.names for level in required_levels):
        raise ValueError(f'series index must have levels {required_levels}, got {series.index.names}')

    if not isinstance(field_name, str) or not field_name:
        raise ValueError(f'field_name must be non-empty string, got {field_name}')

    # 提取索引信息
    instruments = series.index.get_level_values('instrument').unique().tolist()
    datetimes = series.index.get_level_values('datetime')

    if len(instruments) == 0 or len(datetimes) == 0:
        logger.warning(f'get_annual_data: no instruments or datetimes for {field_name}')
        return pd.Series(dtype=float, index=pd.MultiIndex.from_tuples([], names=['instrument', 'year']))

    # 计算每个日期对应的 fiscal_year (YYYY04格式)
    fiscal_years = _extract_fiscal_years(datetimes)
    unique_fiscal_years = sorted(set(fiscal_years))

    # ========== 批量查询年报数据 ==========
    pref_fields = _build_pref_fields(field_name, unique_fiscal_years)

    try:
        start_time = datetimes.min()
        end_time = datetimes.max()

        logger.debug(f'get_annual_data: querying {len(pref_fields)} fields for {len(instruments)} instruments')

        # 一次性查询所有数据（通过 get_qlib_data 自动处理连接保活）
        df_all = get_qlib_data(
            instruments=instruments,
            fields=pref_fields,
            start_date=str(start_time)[:10],
            end_date=str(end_time)[:10],
        )

        if df_all.empty:
            logger.warning(f'get_annual_data: empty result from D.features for {field_name}')
            return pd.Series(dtype=float, index=pd.MultiIndex.from_tuples([], names=['instrument', 'year']))

    except Exception as e:
        logger.error(f'get_annual_data: failed to query data for {field_name}: {e}')
        raise RuntimeError(f'Failed to query annual data for {field_name}: {e}')

    # ========== 提取年度数据 ==========
    # 创建财年到字段名的映射
    fy_to_field = {fy: f'PRef($${field_name}, {fy // 100}04)' for fy in unique_fiscal_years}

    # 构建映射表：每个 <instrument, datetime> 对应的 fiscal_year
    mapping_df = pd.DataFrame({
        'instrument': series.index.get_level_values('instrument'),
        'datetime': datetimes,
        'fiscal_year': fiscal_years
    })

    # 将宽表转为长表
    df_long = df_all.reset_index().melt(
        id_vars=['instrument', 'datetime'],
        var_name='field',
        value_name='value'
    )

    # 根据 fiscal_year 映射到对应的 field 名称
    mapping_df['field'] = mapping_df['fiscal_year'].map(fy_to_field)

    # 合并获取对应值
    merged_df = mapping_df.merge(
        df_long,
        on=['instrument', 'datetime', 'field'],
        how='left'
    )

    # ========== 聚合为年度数据 ==========
    # 对每个 (instrument, fiscal_year) 取最后一个有效值（年报数据恒定）
    merged_df['year'] = merged_df['fiscal_year'] // 100

    # 按 instrument 和 year 分组，取最后一个非空值
    annual_data = merged_df.groupby(['instrument', 'year'])['value'].last()
    annual_data.name = series.name

    logger.debug(f'get_annual_data: successfully extracted {len(annual_data.dropna())}/{len(annual_data)} values for {field_name}')

    return annual_data


def get_annual_data_year_end(series):
    """从日频数据中提取年度数据（每年最后一个交易日的值）
    
    与 get_annual_data 不同：本函数取每年最后一个交易日的值，
    适用于日频交易数据（如市值、价格等），而非财务报告数据。
    
    Args:
        series: pd.Series, MultiIndex (instrument, datetime)
                日频数据（如 $circ_mv, $close 等）
    
    Returns:
        pd.Series: MultiIndex (instrument, year), 年度数据
                   year 为数据所在年份
    """
    data = series.reset_index()
    data.columns = ['instrument', 'datetime', 'value']
    data['year'] = data['datetime'].dt.year
    
    # 取每年最后一个交易日的值
    annual = data.groupby(['instrument', 'year']).last().reset_index()
    annual = annual.set_index(['instrument', 'year'])['value']
    annual.index.names = ['instrument', 'year']
    annual.name = series.name
    return annual


def calc_variation(series, window=5, min_periods=3):
    """计算过去N年的变异系数（标准差/均值）
    
    Args:
        series: pd.Series, MultiIndex (instrument, year) 年度数据
        window: int, 窗口年数，默认5年
        min_periods: int, 最小有效年数，默认3年
    
    Returns:
        pd.Series: 变异系数
    """
    def _calc_cv(x):
        valid = x.dropna()
        if len(valid) < min_periods:
            return np.nan
        std = valid.std()
        mean = valid.mean()
        if mean == 0 or np.isnan(mean):
            return np.nan
        return std / mean
    
    return series.groupby(level='instrument').rolling(window=window, min_periods=min_periods).apply(_calc_cv, raw=False)


def calc_growth_rate_slope(series, window=5, min_periods=3):
    """计算过去N年增长率（基于线性回归斜率/均值）
    
    Args:
        series: pd.Series, MultiIndex (instrument, year) 年度数据
        window: int, 窗口年数，默认5年
        min_periods: int, 最小有效年数，默认3年
    
    Returns:
        pd.Series: 增长率（斜率/均值）
    """
    def _calc_slope(y):
        valid_mask = y.notna()
        valid = y[valid_mask]
        if len(valid) < min_periods:
            return np.nan
        # 保留原始时间位置（与 barra_cne6_factor._cal_growth_rate 一致）
        x = np.arange(1, len(y) + 1)[valid_mask.values]
        # 线性回归: y = a + b*x
        coef = np.polyfit(x, valid.values, 1)
        slope = coef[0]
        mean_val = valid.mean()
        if np.abs(mean_val) < 1e-6 or np.isnan(mean_val):
            return np.nan
        return slope / mean_val
    
    return series.groupby(level='instrument').rolling(window=window, min_periods=min_periods).apply(_calc_slope, raw=False)


def calc_cv(series, window=5, min_periods=3):
    """计算变异系数（Coefficient of Variation）
    
    变异系数 = 标准差 / 均值，用于衡量数据的相对波动程度。
    常用于 VSAL、VERN、VFLO 等盈利波动率因子计算。
    
    Args:
        series: pd.Series, MultiIndex (instrument, year) 年度数据
        window: int, 窗口年数，默认5年
        min_periods: int, 最小有效年数，默认3年
    
    Returns:
        pd.Series: 变异系数
    """
    def calc_cv_inner(x):
        valid = x.dropna()
        if len(valid) < min_periods:
            return np.nan
        std = valid.std()
        mean = valid.mean()
        if abs(mean) < 1e-6 or np.isnan(mean):
            return np.nan
        return std / mean
    
    return series.groupby(level='instrument').rolling(window=window, min_periods=min_periods).apply(calc_cv_inner, raw=False)


def map_annual_to_daily(annual_series, daily_index, is_map: bool = True):
    """将年度指标映射回日频
    
    将 (instrument, year) 索引的年度计算结果，按财年规则映射到每个交易日。
    
    映射规则：
        get_annual_data 的 year 已是财报年份（report_year），
        交易日 D 的 fiscal_year 即其应引用的财报年份，
        两者直接匹配。
    
    Args:
        annual_series: pd.Series, MultiIndex (instrument, year)
            由 get_annual_data + calc_cv/calc_growth_rate_slope 产出的年度指标
            year 为财报年份
        daily_index: pd.MultiIndex (instrument, datetime)
            目标日频索引
    
    Returns:
        pd.Series: MultiIndex (instrument, datetime)，日频数据
    """
    # 构建交易日→fiscal_year 映射
    instruments = daily_index.get_level_values('instrument')
    datetimes = daily_index.get_level_values('datetime')
    if is_map:
        fiscal_years = datetimes.map(get_fiscal_year_for_date)
    else:
        fiscal_years = datetimes.map(lambda x: x.year)

    day_df = pd.DataFrame({
        'instrument': instruments,
        'datetime': datetimes,
        'annual_year': fiscal_years,
    })

    annual_df = annual_series.reset_index()
    annual_df.columns = ['instrument', 'annual_year', 'value']

    merged = day_df.merge(annual_df, on=['instrument', 'annual_year'], how='left')
    result = merged.set_index(['instrument', 'datetime'])['value']
    result.name = annual_series.name
    return result


def get_fiscal_year_for_date(date):
    """根据日期判断应使用的财政年度
    
    规则：
    - 1月1日-4月30日：使用上上财年数据（year - 2）
    - 5月1日及之后：使用上财年数据（year - 1）
    
    例：
    - 2024-03-01 -> 2022年（上上财年）
    - 2024-05-01 -> 2023年（上财年）
    
    Args:
        date: datetime-like, 日期
    
    Returns:
        int: 财政年度年份
    """
    date = pd.to_datetime(date)
    year = date.year
    month = date.month
    
    if month <= 4:
        # 1-4月使用上上财年
        return year - 2
    else:
        # 5-12月使用上财年
        return year - 1


def get_fiscal_year_04(date):
    """根据日期计算财年（YYYY04格式，年报用Q4表示）
    
    规则：
    - 1月1日-4月30日：使用上上财年数据（year - 2）
    - 5月1日及之后：使用上财年数据（year - 1）
    
    Args:
        date: datetime-like, 日期
    
    Returns:
        int: 财年，格式为 YYYY04（年 * 100 + 4），如 202204 表示2022年年报
    """
    date = pd.to_datetime(date)
    year = date.year
    month = date.month
    
    if month <= 4:
        return (year - 2) * 100 + 4
    else:
        return (year - 1) * 100 + 4


def _extract_fiscal_years(datetimes):
    """从时间序列中提取每个日期对应的财年 (YYYY04格式)
    
    Args:
        datetimes: pd.DatetimeIndex 或 iterable, 日期序列
        
    Returns:
        list: 每个日期对应的财年 (YYYY04格式)
    """
    return [get_fiscal_year_04(dt) for dt in datetimes]


def _build_pref_fields(field_name, unique_years):
    """构建 PRef 字段列表
    
    Args:
        field_name: str, 字段名
        unique_years: list, 财年列表 (YYYY04格式)
        
    Returns:
        list: PRef 字段列表
    """
    return [f'PRef($${field_name}, {fy // 100}04)' for fy in unique_years]


def remap_lyr(series, field_name):
    """使用 PRef 算子获取年报数据，按财年规则重映射（优化版）
    
    优化内容：
    1. 批量查询：一次性查询所有 instruments 和财年的数据
    2. 向量化映射：使用 merge 替代逐行循环匹配
    3. 增强错误处理：详细记录失败的查询
    4. 输入验证：增加参数校验
    
    财年规则：
        1-4月 → 使用 year-2 年报（如 2024-03 用 202204 年报）
        5-12月 → 使用 year-1 年报（如 2024-06 用 202304 年报）
    
    注意：如果年报尚未发布（通常在次年4月才发布），PRef 返回 NaN

    Args:
        series: pd.Series, MultiIndex (instrument, datetime)
                仅用于获取 instrument 和 datetime 索引，不使用其值
        field_name: str, 字段名（不含 P()/$$ 前缀），如 'revenue_q', 'total_ncl_q'

    Returns:
        pd.Series: MultiIndex (instrument, datetime)，按财年规则映射后的年报数据
    """
    # ========== 输入验证 ==========
    if not isinstance(series, pd.Series):
        raise TypeError(f'series must be pd.Series, got {type(series)}')
    
    if series.empty:
        logger.warning(f'remap_lyr: empty series for {field_name}')
        return series
    
    if not isinstance(series.index, pd.MultiIndex):
        raise ValueError(f'series must have MultiIndex, got {type(series.index)}')
    
    required_levels = ['instrument', 'datetime']
    if not all(level in series.index.names for level in required_levels):
        raise ValueError(f'series index must have levels {required_levels}, got {series.index.names}')
    
    if not isinstance(field_name, str) or not field_name:
        raise ValueError(f'field_name must be non-empty string, got {field_name}')
    
    # 提取索引信息
    instruments = series.index.get_level_values('instrument').unique().tolist()
    datetimes = series.index.get_level_values('datetime')
    
    if len(instruments) == 0 or len(datetimes) == 0:
        logger.warning(f'remap_lyr: no instruments or datetimes for {field_name}')
        return series
    
    # 计算每个日期对应的 fiscal_year
    fiscal_years = _extract_fiscal_years(datetimes)
    unique_fiscal_years = sorted(set(fiscal_years))

    # ========== 批量查询年报数据 ==========
    pref_fields = _build_pref_fields(field_name, unique_fiscal_years)
    '''
    ['PRef($$revenue_q, 202104)',
     'PRef($$revenue_q, 202204)',
     'PRef($$revenue_q, 202304)',
     'PRef($$revenue_q, 202404)']
    '''
    try:
        # 一次性查询所有数据（通过 get_qlib_data 自动处理连接保活）
        start_time = datetimes.min()
        end_time = datetimes.max()
        
        logger.debug(f'remap_lyr: querying {len(pref_fields)} fields for {len(instruments)} instruments')
        # index: <instrument, datetime>; columns: <PRef($$field, YYYYMM)>
        df_all = get_qlib_data(
            instruments=instruments,
            fields=pref_fields,
            start_date=str(start_time)[:10],
            end_date=str(end_time)[:10],
        )
        
        if df_all.empty:
            logger.warning(f'remap_lyr: empty result from D.features for {field_name}')
            return pd.Series(np.nan, index=series.index, name=series.name)
        
    except Exception as e:
        logger.error(f'remap_lyr: failed to query data for {field_name}: {e}')
        raise RuntimeError(f'Failed to query annual data for {field_name}: {e}')
    
    # ========== 构建映射表 ==========
    # 创建财年到字段名的映射
    fy_to_field = {fy: f'PRef($${field_name}, {fy // 100}04)' for fy in unique_fiscal_years}
    '''
    {202104: 'PRef($$revenue_q, 202104)',
     202204: 'PRef($$revenue_q, 202204)',
     202304: 'PRef($$revenue_q, 202304)',
     202404: 'PRef($$revenue_q, 202404)'}
    '''

    # ========== 方案1: Melt + Merge ==========
    # 1. 将宽表转为长表 (id_vars保留索引列)
    df_long = df_all.reset_index().melt(
        id_vars=['instrument', 'datetime'],
        var_name='field',
        value_name='value'
    )
    '''
    melt后格式:
         instrument   datetime                  field      value
    0       SH600000 2024-01-02  PRef($$revenue_q, 202104)  100.0
    1       SH600000 2024-01-02  PRef($$revenue_q, 202204)  110.0
    ...
    '''
    
    # 2. 构建目标映射表 (每个<instrument,datetime>应该取哪个field)
    target_df = pd.DataFrame({
        'instrument': series.index.get_level_values('instrument'),
        'datetime': series.index.get_level_values('datetime'),
        'fiscal_year': fiscal_years
    })
    # 根据fiscal_year映射到对应的field名称
    target_df['field'] = target_df['fiscal_year'].map(fy_to_field)
    
    # 3. 合并获取对应值 (根据instrument+datetime+field三键匹配)
    result_df = target_df.merge(
        df_long, 
        on=['instrument', 'datetime', 'field'], 
        how='left'
    )
    
    # 4. 重建结果 Series (保持原索引)
    result = pd.Series(
        result_df['value'].values,
        index=series.index,
        name=series.name
    )
    
    logger.debug(f'remap_lyr: successfully mapped {len(result.dropna())}/{len(result)} values for {field_name}')
    
    return result
