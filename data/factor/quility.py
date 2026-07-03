"""质量因子 - 杠杆因子、盈利质量、盈利能力、投资质量"""

import pandas as pd
import numpy as np
from .utils import (
    remap_lyr, calc_cv, get_annual_data, get_annual_data2,
    map_annual_to_daily, calc_growth_rate_slope,
    get_annual_data_year_end,
    factor_output
)
from utils.dt import time_decorator

# ==================== Leverage (杠杆) ====================

@time_decorator
@factor_output
def MLEV(df) -> pd.Series:
    """
    Formulation: MLEV = (ME + PE + LD) / ME
    Description：【市场杠杆因子】衡量企业整体杠杆水平。
        ME 为总市值，PE 为优先股，LD 长期负债
    """
    df = df.sort_index()

    me = df['$total_mv']                                    # 总市值
    me = me.groupby(level='instrument').shift(1)            # 滞后1日，避免前视偏差
    pe = df['P($$oth_eqt_tools_p_shr_a)'].fillna(0)         # 优先股
    # 长期借款 + 应付债券 + 长期应付款
    ld = df['P($$lt_borr_a)'].fillna(0) + df['P($$bond_payable_a)'].fillna(0) + df['P($$lt_payable_a)'].fillna(0)

    # 防止分母为 0
    me[me == 0] = np.nan
    mlev = (me + pe + ld) / me

    return mlev


@time_decorator
@factor_output
def BLEV(df) -> pd.Series:
    """
    Formulation: BLEV = (BE + PE + LD) / BE
                  其中 BE = 股东权益合计(不含少数股东权益) - 其他权益工具(优先股)
    Description：【账面杠杆因子】衡量企业账面杠杆水平。
        BE 为账面权益，PE 为优先股，LD 为长期负债
    """
    df = df.sort_index()

    pe = df['P($$oth_eqt_tools_p_shr_a)'].fillna(0)            # 优先股
    # 长期借款 + 应付债券 + 长期应付款
    ld = df['P($$lt_borr_a)'].fillna(0) + df['P($$bond_payable_a)'].fillna(0) + df['P($$lt_payable_a)'].fillna(0)
    # BE = 股东权益合计(不含少数股东权益) - 优先股
    be = df['P($$total_hldr_eqy_exc_min_int_a)'] - pe

    be[be == 0] = np.nan
    blev = (be + pe + ld) / be

    return blev


@time_decorator
@factor_output
def DTOA(df):
    """
    Formulation: DTOA = TL / TA
    Description：【债务资产比因子】衡量企业负债水平。
        TL 为负债合计，TA 为资产总计。
    """
    df = df.sort_index()

    tl = df['P($$total_liab_a)'].fillna(0)    # 负债合计
    ta = df['P($$total_assets_q)']            # 资产总计

    ta[ta == 0] = np.nan
    dtoa = tl / ta

    return dtoa


# ==================== Earnings Variability (盈利波动率) ====================

@time_decorator
@factor_output
def VSAL(df):
    """
    Variation in Sales (营业收入波动率)
    Formulation: std(revenue, 5Y) / mean(revenue, 5Y)
    Description：过去五个财年的年营业收入标准差除以平均年营业收入。
        反映公司营业收入的波动情况，消除公司规模影响。
    数据字段：营业收入 P($$revenue_q)
    """
    df = df.sort_index()
    revenue = df['P($$revenue_a)']

    # 提取年度数据 (instrument, year)，每年仅包含1条数据
    '''
    instrument  year
    SZ000001    2019    1.029580e+11
                2020    1.165640e+11
    '''
    # Fix: 滚动计算时，前边界小于window的区间值为NaN，需要按照window拉长计算区间
    annual_rev = get_annual_data2(revenue)

    # 对年度数据计算5年滚动变异系数
    cv = calc_cv(annual_rev, window=5, min_periods=3)
    cv = cv.reset_index(level=0, drop=True)

    # 将年度结果映射回日频
    vsal = map_annual_to_daily(cv, df.index)

    return vsal


@time_decorator
@factor_output
def VERN(df):
    """
    Variation in Earnings (盈利波动率)
    Formulation: std(n_income_attr_p, 5Y) / mean(n_income_attr_p, 5Y)
    Description：过去五个财年的年净利润标准差除以平均年净利润。
        捕捉公司财务报表底层盈利的稳定程度。
    数据字段：净利润(不含少数股东损益)
    """
    df = df.sort_index()
    income = df['P($$n_income_attr_p_a)']

    annual_income = get_annual_data2(income)
    cv = calc_cv(annual_income, window=5, min_periods=3, is_abs=True)
    cv = cv.reset_index(level=0, drop=True)
    vern = map_annual_to_daily(cv, df.index)

    return vern


@time_decorator
@factor_output
def VFLO(df):
    """
    Variation in Cash-Flows (现金流波动率)
    Formulation: std(n_cashflow_act, 5Y) / mean(n_cashflow_act, 5Y)
    Description：过去五个财年的年经营性活动现金流标准差除以平均经营性活动现金流。
        反映公司经营活动产生现金的稳定性和可预测性。
    数据字段：经营活动产生的现金流量净额
    """
    df = df.sort_index()
    cf = df['P($$n_cashflow_act_a)']

    annual_cf = get_annual_data2(cf)
    cv = calc_cv(annual_cf, window=5, min_periods=3, is_abs=True)
    cv = cv.reset_index(level=0, drop=True)
    vflo = map_annual_to_daily(cv, df.index)

    return vflo


# ==================== Earnings Quality (盈利质量) ====================

@time_decorator
def ABS(df):
    """
    Accruals Balance Sheet Version（资产负债表应计项目）
    Formulation: ABS = -ACCR_BS / TA
        ACCR_BS = NOA(t) - NOA(t-1) - DA(t)
        NOA = (TA - Cash) - (TL - TD)
        TD = 短期借款 + 长期借款 + 一年内到期的非流动负债 + 应付债券
        DA = 固定资产折旧 + 无形资产摊销 + 长期待摊费用摊销
    Description：揭示公司利润中有多少变成了真实的现金，有多少只是账面上的数字增长。
        正应计：利润>现金流，盈利质量存疑；负应计：利润<现金流，盈利质量极高。
    数据字段：资产总计、货币资金、负债合计、短期借款、长期借款、
        一年内到期的非流动负债、应付债券、固定资产折旧、无形资产摊销、长期待摊费用摊销
    """
    df = df.sort_index()
    
    # 原始数据获取（日频 P() 数据）
    ta_raw = df['P($$total_assets_q)'].fillna(0)      # 总资产
    cash_raw = df['P($$money_cap_q)'].fillna(0)       # 货币资金
    tl_raw = df['P($$total_liab_q)'].fillna(0)        # 总负债
    st_borr_raw = df['P($$st_borr_q)'].fillna(0)      # 短期借款
    lt_borr_raw = df['P($$lt_borr_q)'].fillna(0)      # 长期借款
    non_cur_raw = df['P($$non_cur_liab_due_1y_q)'].fillna(0)  # 一年内到期的非流动负债
    bond_raw = df['P($$bond_payable_q)'].fillna(0)    # 应付债券
    depr_raw = df['P($$depr_fa_coga_dpba_q)'].fillna(0)  # 固定资产折旧
    amort_raw = df['P($$amort_intang_assets_q)'].fillna(0)  # 无形资产摊销
    lt_amort_raw = df['P($$lt_amort_deferred_exp_q)'].fillna(0)  # 长期待摊费用摊销

    # 提取年度数据（使用 PRef 查询原始字段）
    # NOA = (TA - Cash) - (TL - TD)，需要先获取各组成部分的年度数据
    ta_annual = get_annual_data(ta_raw, 'total_assets_q').fillna(0)
    cash_annual = get_annual_data(cash_raw, 'money_cap_q').fillna(0)
    tl_annual = get_annual_data(tl_raw, 'total_liab_q').fillna(0)
    st_borr_annual = get_annual_data(st_borr_raw, 'st_borr_q').fillna(0)
    lt_borr_annual = get_annual_data(lt_borr_raw, 'lt_borr_q').fillna(0)
    non_cur_annual = get_annual_data(non_cur_raw, 'non_cur_liab_due_1y_q').fillna(0)
    bond_annual = get_annual_data(bond_raw, 'bond_payable_q').fillna(0)
    depr_annual = get_annual_data(depr_raw, 'depr_fa_coga_dpba_q').fillna(0)
    amort_annual = get_annual_data(amort_raw, 'amort_intang_assets_q').fillna(0)
    lt_amort_annual = get_annual_data(lt_amort_raw, 'lt_amort_deferred_exp_q').fillna(0)

    # 在年度粒度上计算中间变量
    td_annual = st_borr_annual + lt_borr_annual + non_cur_annual + bond_annual
    da_annual = depr_annual + amort_annual + lt_amort_annual
    noa_annual = (ta_annual - cash_annual) - (tl_annual - td_annual)
    
    # 在年度粒度上计算 NOA(t) - NOA(t-1)
    noa_lag_annual = noa_annual.groupby(level='instrument').shift(1)
    noa_diff_annual = noa_annual - noa_lag_annual
    
    # 在年度粒度计算 ACCR_BS = NOA(t) - NOA(t-1) - DA(t)
    accr_bs_annual = noa_diff_annual - da_annual
    
    # 在年度粒度计算 ABS = -ACCR_BS / TA
    ta_annual[ta_annual == 0] = np.nan
    abs_annual = -accr_bs_annual / ta_annual
    
    # 映射回日频
    abs_val = map_annual_to_daily(abs_annual, df.index)
    
    result_df = pd.DataFrame({'ABS': abs_val})
    result_df = result_df.dropna()
    return result_df


@time_decorator
def ACF(df):
    """
    Accruals CashFlow version (现金流量表应计项目)
    Formulation: ACF = -ACCR_CF / TA
        ACCR_CF = NI(t) - CFO(t) + DA(t)
        DA = 固定资产折旧 + 无形资产摊销 + 长期待摊费用摊销
    Description：直观反应利润和现金流之间的缺口，缺口越大，盈利质量越值得怀疑。
    数据字段：净利润(不含少数股东损益)、经营活动产生的现金流量净额、
        固定资产折旧、无形资产摊销、长期待摊费用摊销、资产总计
    """
    df = df.sort_index()
    
    # 原始数据获取
    ni_raw = df['P($$n_income_attr_p_q)'].fillna(0)   # 净利润
    cfo_raw = df['P($$n_cashflow_act_q)'].fillna(0)   # 经营现金流
    depr_raw = df['P($$depr_fa_coga_dpba_q)'].fillna(0)  # 固定资产折旧
    amort_raw = df['P($$amort_intang_assets_q)'].fillna(0)  # 无形资产摊销
    lt_amort_raw = df['P($$lt_amort_deferred_exp_q)'].fillna(0)  # 长期待摊费用摊销
    ta_raw = df['P($$total_assets_q)'].fillna(0)      # 总资产
    
    # 按财年规则重映射
    ni = remap_lyr(ni_raw, 'n_income_attr_p_q')
    cfo = remap_lyr(cfo_raw, 'n_cashflow_act_q')
    depr = remap_lyr(depr_raw, 'depr_fa_coga_dpba_q')
    amort = remap_lyr(amort_raw, 'amort_intang_assets_q')
    lt_amort = remap_lyr(lt_amort_raw, 'lt_amort_deferred_exp_q')
    ta = remap_lyr(ta_raw, 'total_assets_q')
    
    # 计算折旧摊销 DA
    da = depr + amort + lt_amort
    
    # 计算应计项目 ACCR_CF = NI - CFO + DA
    accr_cf = ni - cfo + da
    
    # 计算 ACF = -ACCR_CF / TA
    ta[ta == 0] = np.nan
    acf_val = -accr_cf / ta
    
    result_df = pd.DataFrame({'ACF': acf_val})
    result_df = result_df.dropna()
    return result_df


# ==================== Profitability (盈利能力) ====================

@time_decorator
@factor_output
def ATO(df):
    """
    Asset Turnover (资产周转率)
    Formulation: ATO = Sales(TTM) / TA
    Description：衡量运营效率，公司利用其总资产产生收入的能力。
        比率越高，说明资产运营效率越高。
    数据字段：营业收入(TTM)、资产总计
    """
    df = df.sort_index()
    
    # TTM 营业收入和总资产
    sales_ttm = df['PTTM($$revenue_q)']
    ta = df['P($$total_assets_q)']   # 总资总计[资产负债表]

    # 计算 ATO
    ta[ta == 0] = np.nan
    ato = sales_ttm / ta

    return ato


@time_decorator
@factor_output
def GP(df):
    """
    Gross Profitability (资产毛收益率)
    Formulation: GP = (Sales - COGS) / TA
    Description：衡量核心盈利效率，公司运用每单位资产能创造多少毛利。
        剔除了销售、管理、研发等费用的影响。
    数据字段：营业收入、营业成本、资产总计
    注意：营业成本字段为 revenue_q
    """
    df = df.sort_index()

    sales = df['P($$revenue_a)'].fillna(0)   # 营业收入
    cogs = df['P($$oper_cost_a)'].fillna(0)  # 营业成本
    ta = df['P($$total_assets_q)']           # 资产总计[资产负债表]

    # 计算 GP = (Sales - COGS) / TA
    ta[ta == 0] = np.nan
    gp = (sales - cogs) / ta

    return gp


@time_decorator
@factor_output
def GPM(df):
    """
    Gross Profit Margin (销售毛利率)
    Formulation: GPM = (Sales - COGS) / Sales
    Description：衡量定价权与成本控制，每单位收入中利润的占比。
        高毛利率通常意味着强大的品牌、定价权或成本优势。
    数据字段：营业收入、营业成本
    """
    df = df.sort_index()

    sales = df['P($$revenue_a)']  # 营业收入
    # 营业成本
    cogs = df['P($$oper_cost_a)'].fillna(0)

    # 避免除以0
    sales[sales == 0] = np.nan
    
    # 计算 GPM = (Sales - COGS) / Sales
    gpm = (sales - cogs) / sales

    return gpm


@time_decorator
@factor_output
def ROA(df):
    """
    Return On Assets (总资产收益率)
    Formulation: ROA = Earnings(TTM) / TA
    Description：衡量综合盈利能力，公司利用全部资产创造净利润的整体效率。
    数据字段：净利润(TTM)、资产总计
    """
    df = df.sort_index()
    
    # TTM 净利润和总资产
    earnings_ttm = df['PTTM($$n_income_attr_p_q)'].fillna(0)
    ta = df['P($$total_assets_q)']    # 资产总计[资产负债表]

    # 计算 ROA
    ta[ta == 0] = np.nan
    roa = earnings_ttm / ta

    return roa


# ==================== Investment Quality (投资质量) ====================

@time_decorator
@factor_output
def AGRO(df):
    """
    Total Assets Growth Rate (总资产增长率)
    Formulation: AGRO = -(过去5年总资产对时间回归的斜率 / 平均总资产)
    Description：衡量资产扩张程度，增长过快的公司可能依赖并购或重资产扩张。
    数据字段：资产总计
    """
    df = df.sort_index()
    ta = df['P($$total_assets_a)']

    # 提取年度数据
    annual_ta = get_annual_data2(ta)

    # 对年度数据计算5年滚动增长率
    growth = calc_growth_rate_slope(annual_ta, window=5, min_periods=3)
    growth = growth.reset_index(level=0, drop=True)

    # 将年度结果映射回日频
    agro = -map_annual_to_daily(growth, df.index)

    return agro


@time_decorator
@factor_output
def IGRO(df):
    """
    Issuance Growth (股票发行量增长率)
    Formulation: IGRO = -(过去5年流通股本对时间回归的斜率 / 平均流通股本)
    Description：衡量股权稀释，频繁增发的公司对外部股权融资依赖度高。
    数据字段：流通股本 $float_share
    """
    df = df.sort_index()
    float_share = df['$float_share']
    
    # 提取年度数据（每年最后一个交易日）
    annual_circ_mv = get_annual_data2(float_share)
    '''
    instrument  year
    SH600000    2018    2.810377e+10
                2019    2.810377e+10
    '''
    # 对年度数据计算5年滚动增长率（斜率/均值）
    growth = calc_growth_rate_slope(annual_circ_mv, window=5, min_periods=3)
    growth = growth.reset_index(level=0, drop=True)
    '''
    instrument  year
    SH600000    2018         NaN
                2019         NaN
                2020    0.021886
                2021    0.017382
    '''
    # 将年度结果映射回日频，并取负号
    igro = -map_annual_to_daily(growth, df.index)

    return igro


@time_decorator
@factor_output
def CXGRO(df):
    """
    Capital Expenditure Growth (资本支出增长率)
    Formulation: CXGRO = -(过去5年资本支出对时间回归的斜率 / 平均资本支出)
    Description：衡量资本开支增速，增速过高的公司可能存在过度投资风险。
    数据字段：购建固定资产、无形资产和其他长期资产支付的现金
    """
    df = df.sort_index()
    capex = df['P($$c_pay_acq_const_fiolta_a)']

    # 提取年度数据
    annual_capex = get_annual_data2(capex)

    # 对年度数据计算5年滚动增长率
    growth = calc_growth_rate_slope(annual_capex, window=5, min_periods=3)
    growth = growth.reset_index(level=0, drop=True)

    # 将年度结果映射回日频
    cxgro = -map_annual_to_daily(growth, df.index)

    return cxgro
