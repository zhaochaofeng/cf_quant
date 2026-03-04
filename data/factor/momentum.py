"""
    动量因子 (Momentum Factors)
    
    基于 BARRA CNE6 模型实现：
    - STREV: 短期反转因子
    - SEASON: 季节因子（月度日历效应）
    - INDMOM: 行业动量因子
    - RSTR: 相对强度因子
"""

import numpy as np
import pandas as pd

from .utils import (
    rolling_with_func, calc_seasonality,
    capm_regress,
    get_benchmark_ret
)


def STREV(df):
    """
    Formulation: STREV = sum(return * weight)
    Description：【短期反转因子】过去1个月（21个交易日）的日收益率加权之和，
        使用半衰期为5天的指数权重。该因子用于捕捉短期价格反转效应。
    """
    # 确保索引排序
    df = df.sort_index()
    
    # 使用日收益率
    daily_returns = df['$change']
    
    # 使用 rolling_with_func 计算加权收益率之和
    # window=21, half_life=5, func_name='sum'
    strev_series = daily_returns.groupby(level='instrument').apply(
        lambda x: rolling_with_func(x, window=21, half_life=5, func_name='sum')
    )
    # 重置索引，确保格式正确
    strev_series = strev_series.reset_index(level=0, drop=True)
    
    # 构造结果 DataFrame
    result_df = pd.DataFrame({'STREV': strev_series})
    result_df = result_df.dropna()
    
    return result_df


def SEASON(df, nyears=5):
    """
    Formulation: SEASON = mean(R_stock - R_benchmark) over past Y years
    Description：【季节因子】捕捉股票月度日历效应，衡量股票在特定月份
        是否存在系统性的超额收益规律（如"一月效应"、"春节效应"等）。
        计算过去Y年（默认5年）同月份相对于基准的超额收益均值。
        
    计算方法：
        1. 计算对数收益率 ln(1+r)
        2. 月度累计对数收益率（每月最后一个交易日的对数收益率累加）
        3. 使用沪深300指数作为市场基准计算月度基准收益率
        4. 计算超额收益 = 股票月度收益率 - 基准月度收益率
        5. 对每个股票，按月份分组，计算过去nyears年同月份超额收益的均值
        6. 将月度因子值扩展回日度
    """
    # 确保索引排序
    df = df.sort_index()
    
    # 获取股票日收益率并计算对数收益率
    stock_ret = df['$change']
    log_ret = np.log(1 + stock_ret)
    
    # 计算月度累计对数收益率（每月最后一个交易日的对数收益率累加）
    monthly_log_ret = log_ret.groupby(level='instrument').resample('ME', level='datetime').sum()

    # 获取基准指数数据（沪深300）
    start_date = str(df.index.get_level_values('datetime').min())[:10]
    end_date = str(df.index.get_level_values('datetime').max())[:10]
    benchmark_ret = get_benchmark_ret(start_date, end_date)
    
    # 计算基准的对数收益率并月度累计
    benchmark_log_ret = np.log(1 + benchmark_ret)
    benchmark_monthly_log_ret = benchmark_log_ret.resample('ME').sum()
    
    # 计算超额收益（股票月度对数收益 - 基准月度对数收益）
    # 将基准数据对齐到股票数据并计算超额收益
    benchmark_aligned = benchmark_monthly_log_ret.reindex(
        monthly_log_ret.index.get_level_values('datetime')
    ).values
    excess_ret = monthly_log_ret - benchmark_aligned

    # 准备超额收益数据
    excess_ret.name = 'excess_ret'
    excess_ret_df = excess_ret.reset_index()
    excess_ret_df['month'] = excess_ret_df['datetime'].dt.month

    # 按股票分组计算季节性。每只股票仅包含月末日期数据
    # 列：[instrument, datetime, SEASON]
    seasonality_list = [
        calc_seasonality(group, nyears=nyears, value_col='excess_ret')
        for _, group in excess_ret_df.groupby('instrument')
    ]

    if not seasonality_list:
        return pd.DataFrame({'SEASON': []})
    seasonality_monthly = pd.concat(seasonality_list, ignore_index=True)

    # 月度数据扩展回日度：通过月末日期 merge
    daily_index = stock_ret.index.to_frame().reset_index(drop=True)
    daily_index['month_end'] = daily_index['datetime'] + pd.offsets.MonthEnd(0)
    seasonality_monthly['month_end'] = seasonality_monthly['datetime'] + pd.offsets.MonthEnd(0)
    # 先用daily_index 与 seasonality_monthly 进行merge，然后对缺失值进行前向填充
    merged = daily_index.merge(
        seasonality_monthly[['instrument', 'month_end', 'SEASON']],
        on=['instrument', 'month_end'], how='left'
    ).set_index(['instrument', 'datetime'])
    merged['SEASON'] = merged.groupby(level='instrument')['SEASON'].ffill()

    return merged[['SEASON']].dropna()


def INDMOM(df):
    """
    Formulation: INDMOM = RS_industry - RS_stock * weight / sum(weight)
    Description：【行业动量因子】6个月（126个交易日）行业动量减去个股动量的加权贡献，
        用于捕捉行业层面的动量效应。使用半衰期为21天的指数权重。
        权重为流通市值的平方根（cap_sqrt）。
    """
    # 确保索引排序
    df = df.sort_index()
    
    # 获取日收益率、流通市值平方根（权重）、行业分类
    daily_returns = df['$change']
    cap_sqrt = np.sqrt(df['$circ_mv'])
    industry = df['$ind_one'].fillna(-1).astype(int)

    # 计算对数收益率
    log_ret = np.log(1 + daily_returns)
    
    # 计算个股动量 rs（6个月，半衰期21天的加权对数收益率累计和）
    rs = log_ret.groupby(level='instrument').apply(
        lambda x: rolling_with_func(x, window=126, half_life=21, func_name='sum')
    )
    rs = rs.reset_index(level=0, drop=True)
    
    # 准备数据：合并 rs, cap_sqrt, industry
    data = rs.reset_index()
    data = data.merge(cap_sqrt.reset_index(), on=['instrument', 'datetime'])
    data = data.merge(industry.reset_index(), on=['instrument', 'datetime'])
    data.columns = ['instrument', 'datetime', 'rs', 'weight', 'ind']
    
    # 排除无效行业数据
    data = data[data['ind'] >= 0]
    
    # 按日期和行业分组计算行业动量（个股 rolling momentum 的 cap_sqrt 加权平均）
    def calc_ind_momentum(group):
        w = group['weight']
        return (group['rs'] * w).sum() / w.sum() if w.sum() > 0 else np.nan
    
    ind_mom = data.groupby(['datetime', 'ind']).apply(
        calc_ind_momentum, include_groups=False
    ).reset_index()
    ind_mom.columns = ['datetime', 'ind', 'rs_ind']
    
    # 将行业动量合并回数据
    data = data.merge(ind_mom, on=['datetime', 'ind'], how='left')
    
    # INDMOM = rs_ind - rs * weight / sum(weight)
    data['INDMOM'] = data['rs_ind'] - data['rs'] * data['weight'] / data['weight'].sum()
    
    # 构造结果 DataFrame
    result_df = data.set_index(['instrument', 'datetime'])[['INDMOM']]
    result_df = result_df.dropna()
    
    return result_df


def RSTR(df):
    """
    Formulation: RSTR = sum(excess_return * weight), then smooth with 11-day MA
    Description：【相对强度因子】过去1年（252个交易日）的日超额收益率
        （股票收益率 - 基准收益率）加权之和，使用半衰期为126天的指数权重，
        最后进行11天的移动平均平滑。
    """
    # 确保索引排序
    df = df.sort_index()
    
    # 获取股票日收益率
    stock_ret = df['$change']
    log_ret = np.log(1 + stock_ret)
    
    # 市场基准收益率
    start_date = str(df.index.get_level_values('datetime').min())[:10]
    end_date = str(df.index.get_level_values('datetime').max())[:10]
    benchmark_ret = get_benchmark_ret(start_date, end_date)
    benchmark_log_ret = np.log(1 + benchmark_ret)

    # 计算超额收益
    excess_ret = log_ret - benchmark_log_ret
    
    # 计算加权超额收益之和（252天窗口，半衰期126天）
    rstr_raw = excess_ret.groupby(level='instrument').apply(
        lambda x: rolling_with_func(x, window=252, half_life=126, func_name='sum')
    )
    # 重置索引，确保格式正确
    rstr_raw = rstr_raw.reset_index(level=0, drop=True)
    
    # 11天移动平均平滑
    rstr = rstr_raw.groupby(level='instrument').rolling(window=11, min_periods=1).mean()
    rstr = rstr.reset_index(level=0, drop=True)
    
    # 构造结果 DataFrame
    result_df = pd.DataFrame({'RSTR': rstr})
    result_df = result_df.dropna()
    
    return result_df


def HALPHA(df):
    """
    Formulation: 通过CAPM模型回归得到的截距项Alpha
    """

    df = df.sort_index()
    stock_returns = df['$change']

    # CAPM 回归
    beta, alpha, sigma = capm_regress(stock_returns, window=504, half_life=252, num_worker=1)

    # 构造结果 DataFrame
    result_df = pd.DataFrame({'HALPHA': alpha})
    result_df = result_df.dropna()

    return result_df




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


