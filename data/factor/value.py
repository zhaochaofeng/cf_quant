"""
    价值因子 (Value Factors)
    包含：BTOP, ETOP, CETOP, EM, LTRSTR, LTHALPHA
"""

import pandas as pd
import numpy as np
from .utils import capm_regress, remap_lyr
from utils.dt import time_decorator


@time_decorator
def BTOP(df):
    """
    Book to Price (账面市值比)
    Formulation: BTOP = 普通股账面价值 / 总市值
        普通股账面价值 = 股东权益合计(不含少数股东权益) - 其他权益工具(优先股)
    Description：衡量资产价值，股价相对于公司净资产的折溢价程度。
        BTOP越高，股价相对净资产越便宜，股票越偏向"价值型"。
    数据字段：股东权益合计(不含少数股东权益)、其他权益工具(优先股)、总市值
    """
    df = df.sort_index()
    
    # 获取数据
    total_hldr_eqy_raw = df['P($$total_hldr_eqy_exc_min_int_q)']    # 股东权益合计(不含少数股东权益)
    oth_eqt_tools_raw = df['P($$oth_eqt_tools_p_shr_q)'].fillna(0)  # 优先股

    total_hldr_eqy = remap_lyr(total_hldr_eqy_raw, 'total_hldr_eqy_exc_min_int_q')
    oth_eqt_tools = remap_lyr(oth_eqt_tools_raw, 'oth_eqt_tools_p_shr_q')

    total_mv = df['$total_mv']

    # 计算普通股账面价值
    bv = total_hldr_eqy - oth_eqt_tools

    # BTOP = 账面价值 / 总市值
    total_mv[total_mv == 0] = np.nan
    btop = bv / total_mv
    
    result_df = pd.DataFrame({'BTOP': btop})
    result_df = result_df.dropna()
    return result_df


@time_decorator
def ETOP(df):
    """
    Trailing Earnings-to-price Ratio (EP比)
    Formulation: ETOP = 最近12个月净利润（TTM） / 总市值
    Description：衡量盈利价值（历史），股价相对于近期已实现盈利的便宜程度。
        ETOP值高，股价相对盈利便宜，偏价值；ETOP值低，股价相对盈利偏贵，偏成长。
    数据字段：净利润(不含少数股东损益) TTM、总市值
    """
    df = df.sort_index()
    
    # 获取数据（使用 PTTM 计算的 TTM 净利润）
    earnings_ttm = df['PTTM($$n_income_attr_p_q)']
    total_mv = df['$total_mv']

    # ETOP = TTM净利润 / 总市值
    # 避免除以0
    total_mv = total_mv.replace(0, np.nan)
    etop = earnings_ttm / total_mv
    
    result_df = pd.DataFrame({'ETOP': etop})
    result_df = result_df.dropna()
    return result_df


@time_decorator
def CETOP(df):
    """
    Cash Earnings To Price (现金盈利价比)
    Formulation: CETOP = 过去12个月现金盈利（TTM） / 总市值
    Description：使用经营性现金流除以市值，剥离非现金会计项干扰，
        更真实反映公司现金产生能力与股价的关系。
    数据字段：经营活动产生的现金流量净额 TTM、总市值
    """
    df = df.sort_index()
    
    # 获取数据（使用 PTTM 计算的 TTM 经营现金流）
    cash_earnings_ttm = df['PTTM($$n_cashflow_act_q)']
    total_mv = df['$total_mv']

    # CETOP = TTM经营现金流 / 总市值
    # 避免除以0
    total_mv = total_mv.replace(0, np.nan)
    cetop = cash_earnings_ttm / total_mv
    
    result_df = pd.DataFrame({'CETOP': cetop})
    result_df = result_df.dropna()
    return result_df


@time_decorator
def EM(df):
    """
    Enterprise Multiple (企业价值倍数的倒数)
    Formulation: EM = EBIT / EV
        EV = 总市值 + 总带息债务 - 货币资金
        总带息债务 = 短期借款 + 长期借款 + 一年内到期的非流动负债 + 应付债券
    Description：衡量企业整体价值，剔除资本结构影响，从全体投资者视角看公司核心业务回报率。
        EM高，企业价值相对核心盈利便宜，投资回报率高。
    数据字段：息税前利润、短期借款、长期借款、一年内到期的非流动负债、应付债券、货币资金、总市值
    """
    df = df.sort_index()

    # ebit 字段缺失值很多
    ebit = df['P($$ebit_q)']
    # 带息债务（缺失视为无该类借款，fillna(0)）
    st_borr = df['P($$st_borr_q)'].fillna(0)
    lt_borr = df['P($$lt_borr_q)'].fillna(0)
    non_cur_liab = df['P($$non_cur_liab_due_1y_q)'].fillna(0)
    bond_payable = df['P($$bond_payable_q)'].fillna(0)
    
    # 货币资金和总市值（缺失保留NaN）
    cash = df['P($$money_cap_q)']
    total_mv = df['$total_mv']

    # 计算总带息债务
    total_interest_bearing_debt = st_borr + lt_borr + non_cur_liab + bond_payable

    # 计算企业价值 EV
    ev = total_mv + total_interest_bearing_debt - cash
    ev = ev.replace(0, np.nan)
    # EM = EBIT / EV
    em = ebit / ev
    
    result_df = pd.DataFrame({'EM': em})
    result_df = result_df.dropna()
    return result_df


@time_decorator
def LTRSTR(df):
    """
    Long Term Relative Strength (长期相对强度)
    Formulation: 
        (1) 计算非滞后的长期相对强度：对股票对数超额收益率进行加权求和
            时间窗口1040个交易日，半衰期260个交易日
        (2) 滞后273个交易日，在11个交易日的时间窗口内取非滞后值等权平均值，然后取相反数
    Description：衡量股票在超长期（3-5年）维度上，其价格趋势的疲弱或超跌程度。
    数据字段：股票收盘价涨跌幅、沪深300指数涨跌幅
    """
    df = df.sort_index()
    
    # 获取股票收益率
    stock_ret = df['$change']
    
    # 计算时间范围
    start_date = str(stock_ret.index.get_level_values('datetime').min())[:10]
    end_date = str(stock_ret.index.get_level_values('datetime').max())[:10]
    
    # 使用 utils 中的工具函数获取基准收益率
    from .utils import get_benchmark_ret
    benchmark_ret = get_benchmark_ret(start_date, end_date)
    
    # 计算对数超额收益率（按股票分组处理，避免stack/unstack）
    def calc_excess_ret(group):
        # 对齐基准收益率
        common_dates = group.index.get_level_values('datetime').intersection(benchmark_ret.index)
        group_ret = group[group.index.get_level_values('datetime').isin(common_dates)]
        bm_ret = benchmark_ret.loc[common_dates]
        return pd.Series(np.log((1 + group_ret.values) / (1 + bm_ret.values)), index=group_ret.index)
    
    excess_ret = stock_ret.groupby(level='instrument', group_keys=False).apply(calc_excess_ret)
    
    # 使用 utils 中的 rolling_with_func 进行半衰期加权滚动求和
    from .utils import rolling_with_func
    ltrstr_raw = excess_ret.groupby(level='instrument', group_keys=False).apply(
        lambda x: rolling_with_func(x, window=1040, half_life=260, func_name='sum')
    )
    
    # 滞后273个交易日，并在11个交易日窗口内取平均，然后取相反数
    ltrstr = (-1) * ltrstr_raw.groupby(level='instrument', group_keys=False).apply(
        lambda x: x.shift(273).rolling(window=11, min_periods=1).mean()
    )
    
    result_df = pd.DataFrame({'LTRSTR': ltrstr})
    result_df = result_df.dropna()
    return result_df


@time_decorator
def LTHALPHA(df):
    """
    Long Term Historical Alpha (长期历史alpha)
    Formulation:
        (1) 计算非滞后的长期历史Alpha：取CAPM回归的截距项
            时间窗口1040个交易日，半衰期260个交易日
        (2) 滞后273个交易日，在11个交易日的时间窗口内取非滞后值等权平均值，然后取相反数
    Description：衡量股票在超长期（3-5年）维度上，其经风险调整后的超额收益的缺失或落后程度。
    数据字段：股票收盘价涨跌幅、沪深300指数收盘价涨跌幅
    """
    df = df.sort_index()
    
    # 获取股票收益率
    stock_ret = df['$change']
    
    # 使用 capm_regress 计算 alpha
    # window=1040, half_life=260
    beta, alpha, sigma = capm_regress(stock_ret, window=1040, half_life=260, num_worker=1)
    
    # 将 alpha 转为宽表格式进行处理
    alpha_wide = alpha.unstack(level='instrument')
    
    # 滞后273个交易日，并在11个交易日窗口内取平均，然后取相反数
    lthalpha = (-1) * alpha_wide.shift(273).rolling(window=11, min_periods=1).mean()
    
    # 转回 MultiIndex 格式
    lthalpha_series = lthalpha.stack()
    lthalpha_series.index.names = ['datetime', 'instrument']
    lthalpha_series = lthalpha_series.reorder_levels(['instrument', 'datetime']).sort_index()
    
    result_df = pd.DataFrame({'LTHALPHA': lthalpha_series})
    result_df = result_df.dropna()
    return result_df
