"""
    价值因子 (Value Factors)
    包含：BTOP, ETOP, CETOP, EM, LTRSTR, LTHALPHA
"""

import numpy as np
import pandas as pd

from barra.base import BaseDataLoader
from utils import get_ret
from utils.dt import time_decorator
from .utils import capm_regress, factor_output, rolling_with_func, get_excess_ret

data_loader = BaseDataLoader()


@time_decorator
@factor_output
def BTOP(df) -> pd.Series:
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
    total_hldr_eqy = df['P($$total_hldr_eqy_exc_min_int_q)'].fillna(0)    # 股东权益合计(不含少数股东权益)[资产负债表]
    oth_eqt_tools = df['P($$oth_eqt_tools_p_shr_q)'].fillna(0)            # 优先股[资产负债表]

    total_mv = df['$total_mv']

    # 计算普通股账面价值
    bv = total_hldr_eqy - oth_eqt_tools

    # BTOP = 账面价值 / 总市值
    total_mv[total_mv == 0] = np.nan
    btop = bv / total_mv

    return btop


@time_decorator
@factor_output
def ETOP(df) :
    """
    Trailing Earnings-to-price Ratio (EP比)
    Formulation: ETOP = 最近12个月净利润（TTM） / 总市值
    Description：衡量盈利价值（历史），股价相对于近期已实现盈利的便宜程度。
        ETOP值高，股价相对盈利便宜，偏价值；ETOP值低，股价相对盈利偏贵，偏成长。
    数据字段：净利润(不含少数股东损益) TTM、总市值
    """
    df = df.sort_index()
    
    # 净利润TTM
    earnings_ttm = df['PTTM($$n_income_attr_p_q)']
    total_mv = df['$total_mv']

    # ETOP = TTM净利润 / 总市值
    # 避免除以0
    total_mv = total_mv.replace(0, np.nan)
    etop = earnings_ttm / total_mv

    return etop


@time_decorator
@factor_output
def CETOP(df) -> pd.Series:
    """
    Cash Earnings To Price (现金盈利价比)
    Formulation: CETOP = 过去12个月现金盈利（TTM） / 总市值
    Description：使用经营性现金流除以市值，剥离非现金会计项干扰，
        更真实反映公司现金产生能力与股价的关系。
    数据字段：经营活动产生的现金流量净额 TTM、总市值
    """
    df = df.sort_index()
    
    # 经营现金流 TTM
    cash_earnings_ttm = df['PTTM($$n_cashflow_act_q)']
    total_mv = df['$total_mv']

    # CETOP = 经营现金流TTM / 总市值
    # 避免除以0
    total_mv = total_mv.replace(0, np.nan)
    cetop = cash_earnings_ttm / total_mv

    return cetop


@time_decorator
@factor_output
def EM(df) -> pd.Series:
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

    # ebit 字段缺失值很多. 不能设置为 0，因为 ebit 可以为负
    ebit = df['P($$ebit_a)']
    # 带息债务（缺失视为无该类借款，fillna(0)）
    st_borr = df['P($$st_borr_q)'].fillna(0)       # 短期借款[资产负债表]
    lt_borr = df['P($$lt_borr_q)'].fillna(0)       # 长期借款[资产负债表]
    non_cur_liab = df['P($$non_cur_liab_due_1y_q)'].fillna(0)       # 一年内到期的非流动负债[资产负债表]
    bond_payable = df['P($$bond_payable_q)'].fillna(0)   # 应付债券[资产负债表]
    
    # 货币资金和总市值（缺失保留NaN）
    cash = df['P($$money_cap_q)'].fillna(0)   # 货币资金[资产负债表]
    total_mv = df['$total_mv']

    # 计算总带息债务
    total_interest_bearing_debt = st_borr + lt_borr + non_cur_liab + bond_payable

    # 计算企业价值 EV
    ev = total_mv + total_interest_bearing_debt - cash
    ev = ev.replace(0, np.nan)
    # EM = EBIT / EV
    em = ebit / ev

    return em


@time_decorator
@factor_output
def LTRSTR(df) -> pd.Series:
    """
    Long Term Relative Strength (长期相对强度)
    Formulation: 
        (1) 计算非滞后的长期相对强度：对股票对数超额收益率进行加权求和
            时间窗口1040个交易日，半衰期260个交易日
        (2) 滞后273个交易日，在11个交易日的时间窗口内取非滞后值等权平均值，然后取相反数
    Description：衡量股票在超长期（3-5年）维度上，其价格趋势的疲弱或超跌程度。
    数据字段：股票收盘价、沪深300指数
    """
    df = df.sort_index()
    
    # 股票对数收益率
    close = df['$close']

    stock_ret = get_ret(close)
    log_stock_ret = np.log(1 + stock_ret)
    
    # 市场对数收益率
    start_date = str(stock_ret.index.get_level_values('datetime').min())[:10]
    end_date = str(stock_ret.index.get_level_values('datetime').max())[:10]
    bench_ret = data_loader.load_benchmark_ret(start_date, end_date)
    log_bm_ret = np.log(1 + bench_ret)

    # 相对于市场的对数超额收益率
    excess_ret = log_stock_ret - log_bm_ret

    ltrstr_raw = excess_ret.groupby(level='instrument', group_keys=False).apply(
        lambda x: rolling_with_func(x, window=1040, half_life=260, func_name='sum')
    )

    # 滞后273个交易日，并在11个交易日窗口内取平均，然后取相反数
    ltrstr = (-1) * ltrstr_raw.groupby(level='instrument', group_keys=False).transform(
        lambda x: x.rolling(window=11, min_periods=1).mean().shift(273)
    )

    return ltrstr


@time_decorator
@factor_output
def LTHALPHA(df) -> pd.Series:
    """
    Long Term Historical Alpha (长期历史alpha)
    Formulation:
        (1) 计算非滞后的长期历史Alpha：取CAPM回归的截距项
            时间窗口1040个交易日，半衰期260个交易日
        (2) 滞后273个交易日，在11个交易日的时间窗口内取非滞后值等权平均值，然后取相反数
    Description：衡量股票在超长期（3-5年）维度上，其经风险调整后的超额收益的缺失或落后程度。
    """
    df = df.sort_index()

    close = df['$close']
    ex_ret = get_excess_ret(close)
    
    # 使用 capm_regress 计算 alpha
    # window=1040, half_life=260
    beta, alpha, sigma = capm_regress(ex_ret, window=1040, half_life=260, num_worker=1)

    lthalpha = -alpha.groupby(level='instrument', group_keys=False).transform(
        lambda x: x.rolling(window=11, min_periods=1).mean().shift(273)
    )

    return lthalpha
