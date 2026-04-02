"""质量因子 - 杠杆因子、盈利质量、盈利能力、投资质量"""

import pandas as pd
import numpy as np
from .utils import (
    remap_lyr, calc_cv, get_annual_data,
    map_annual_to_daily, calc_growth_rate_slope,
    get_annual_data_year_end
)

# ==================== Leverage (盈利波动率) ====================

def MLEV(df):
    """
    Formulation: MLEV = (ME + PE + LD) / ME
    Description：【市场杠杆因子】衡量企业整体杠杆水平。
        ME 为总市值，PE 为优先股，LD 为非流动负债。
        财年规则：1-4月使用上上财年数据，5月及之后使用上财年数据。
    """
    df = df.sort_index()

    me = df['$total_mv'] * 10000                  # 总市值（万元）
    me = me.groupby(level='instrument').shift(1)  # 滞后1日，避免前视偏差
    pe_raw = df['P($$oth_eqt_tools_p_shr_q)']     # 优先股
    ld_raw = df['P($$total_ncl_q)']               # 非流动负债合计
    pe_raw.fillna(0, inplace=True)
    ld_raw.fillna(0, inplace=True)

    # 按财年规则重映射财务数据
    pe = remap_lyr(pe_raw, 'oth_eqt_tools_p_shr_q')
    ld = remap_lyr(ld_raw, 'total_ncl_q')

    mlev = (me + pe + ld) / me

    result_df = pd.DataFrame({'MLEV': mlev})
    result_df = result_df.dropna()
    return result_df


def BLEV(df):
    """
    Formulation: BLEV = (BE + PE + LD) / BE
                  其中 BE = 股东权益合计(不含少数股东权益) - 其他权益工具(优先股)
    Description：【账面杠杆因子】衡量企业账面杠杆水平。
        BE 为账面权益，PE 为优先股，LD 为非流动负债。
        财年规则：1-4月使用上上财年数据，5月及之后使用上财年数据。
    """
    df = df.sort_index()

    pe_raw = df['P($$oth_eqt_tools_p_shr_q)']             # 优先股
    ld_raw = df['P($$total_ncl_q)']                       # 非流动负债合计
    be_raw_raw = df['P($$total_hldr_eqy_exc_min_int_q)']  # 股东权益合计(不含少数股东权益)
    be_raw_raw.fillna(0, inplace=True)
    pe_raw.fillna(0, inplace=True)
    ld_raw.fillna(0, inplace=True)

    # 按财年规则重映射财务数据
    pe = remap_lyr(pe_raw, 'oth_eqt_tools_p_shr_q')
    ld = remap_lyr(ld_raw, 'total_ncl_q')
    be_raw = remap_lyr(be_raw_raw, 'total_hldr_eqy_exc_min_int_q')

    # BE = 股东权益合计(不含少数股东权益) - 优先股
    be = be_raw - pe
    # 将be 元素为0的设置NaN，防止除以0错误
    be[be == 0] = np.nan

    blev = (be + pe + ld) / be

    result_df = pd.DataFrame({'BLEV': blev})
    result_df = result_df.dropna()
    return result_df


def DTOA(df):
    """
    Formulation: DTOA = TL / TA
    Description：【债务资产比因子】衡量企业负债水平。
        TL 为负债合计，TA 为资产总计。
        财年规则：1-4月使用上上财年数据，5月及之后使用上财年数据。
    """
    df = df.sort_index()

    tl_raw = df['P($$total_liab_q)']    # 负债合计
    ta_raw = df['P($$total_assets_q)']  # 资产总计
    tl_raw.fillna(0, inplace=True)

    # 按财年规则重映射财务数据
    tl = remap_lyr(tl_raw, 'total_liab_q')
    ta = remap_lyr(ta_raw, 'total_assets_q')

    dtoa = tl / ta

    result_df = pd.DataFrame({'DTOA': dtoa})
    result_df = result_df.dropna()
    return result_df


# ==================== Earnings Variability (盈利波动率) ====================

def VSAL(df):
    """
    Variation in Sales (营业收入波动率)
    Formulation: std(revenue, 5Y) / mean(revenue, 5Y)
    Description：过去五个财年的年营业收入标准差除以平均年营业收入。
        反映公司营业收入的波动情况，消除公司规模影响。
    数据字段：营业收入 P($$revenue_q)
    """
    df = df.sort_index()
    revenue_raw = df['P($$revenue_q)'].fillna(0)

    # 提取年度数据 (instrument, year)，每年仅包含1条数据
    '''
    instrument  year
    SZ000001    2019    1.029580e+11
                2020    1.165640e+11
    '''
    annual_rev = get_annual_data(revenue_raw)

    # 对年度数据计算5年滚动变异系数
    cv = calc_cv(annual_rev, window=5, min_periods=3)
    cv = cv.reset_index(level=0, drop=True)

    # 将年度结果映射回日频
    vsal = map_annual_to_daily(cv, df.index)

    result_df = pd.DataFrame({'VSAL': vsal})
    result_df = result_df.dropna()
    return result_df


def VERN(df):
    """
    Variation in Earnings (盈利波动率)
    Formulation: std(n_income_attr_p, 5Y) / mean(n_income_attr_p, 5Y)
    Description：过去五个财年的年净利润标准差除以平均年净利润。
        捕捉公司财务报表底层盈利的稳定程度。
    数据字段：净利润(不含少数股东损益) P($$n_income_attr_p_q)
    """
    df = df.sort_index()
    income_raw = df['P($$n_income_attr_p_q)'].fillna(0)

    annual_income = get_annual_data(income_raw)
    cv = calc_cv(annual_income, window=5, min_periods=3)
    cv = cv.reset_index(level=0, drop=True)
    vern = map_annual_to_daily(cv, df.index)

    result_df = pd.DataFrame({'VERN': vern})
    result_df = result_df.dropna()
    return result_df


def VFLO(df):
    """
    Variation in Cash-Flows (现金流波动率)
    Formulation: std(n_cashflow_act, 5Y) / mean(n_cashflow_act, 5Y)
    Description：过去五个财年的年经营性活动现金流标准差除以平均经营性活动现金流。
        反映公司经营活动产生现金的稳定性和可预测性。
    数据字段：经营活动产生的现金流量净额 P($$n_cashflow_act_q)
    """
    df = df.sort_index()
    cf_raw = df['P($$n_cashflow_act_q)'].fillna(0)

    annual_cf = get_annual_data(cf_raw)
    cv = calc_cv(annual_cf, window=5, min_periods=3)
    cv = cv.reset_index(level=0, drop=True)
    vflo = map_annual_to_daily(cv, df.index)

    result_df = pd.DataFrame({'VFLO': vflo})
    result_df = result_df.dropna()
    return result_df


# ==================== Earnings Quality (盈利质量) ====================

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
    
    # 在原始日频数据上计算中间变量（用于提取年度数据）
    td_raw = st_borr_raw + lt_borr_raw + non_cur_raw + bond_raw  # 带息债务
    da_raw = depr_raw + amort_raw + lt_amort_raw                  # 折旧摊销
    noa_raw = (ta_raw - cash_raw) - (tl_raw - td_raw)             # 净经营资产
    
    # 提取年度数据（每年1月第一个值 → 对应上一年年报）
    noa_annual = get_annual_data(noa_raw)
    da_annual = get_annual_data(da_raw)
    ta_annual = get_annual_data(ta_raw)
    
    # 在年度粒度上计算 NOA(t) - NOA(t-1)
    noa_lag_annual = noa_annual.groupby(level='instrument').shift(1)
    noa_diff_annual = noa_annual - noa_lag_annual
    
    # 在年度粒度计算 ACCR_BS = NOA(t) - NOA(t-1) - DA(t)
    accr_bs_annual = noa_diff_annual - da_annual
    
    # 在年度粒度计算 ABS = -ACCR_BS / TA
    abs_annual = -accr_bs_annual / ta_annual
    
    # 映射回日频
    abs_val = map_annual_to_daily(abs_annual, df.index)
    
    result_df = pd.DataFrame({'ABS': abs_val})
    result_df = result_df.dropna()
    return result_df


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
    acf_val = -accr_cf / ta
    
    result_df = pd.DataFrame({'ACF': acf_val})
    result_df = result_df.dropna()
    return result_df


# ==================== Profitability (盈利能力) ====================

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
    sales_ttm = df['PTTM($$revenue_q)'].fillna(0)
    ta_raw = df['P($$total_assets_q)'].fillna(0)
    
    # 总资产使用最新报告期数据（非TTM）
    ta = remap_lyr(ta_raw, 'total_assets_q')
    
    # 计算 ATO
    ato = sales_ttm / ta
    
    result_df = pd.DataFrame({'ATO': ato})
    result_df = result_df.dropna()
    return result_df


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
    
    # 使用上财年数据
    sales_raw = df['P($$revenue_q)'].fillna(0)
    # 营业成本
    cogs_raw = df['P($$oper_cost_q)'].fillna(0)
    ta_raw = df['P($$total_assets_q)']
    
    sales = remap_lyr(sales_raw, 'revenue_q')
    cogs = remap_lyr(cogs_raw, 'oper_cost_q')
    ta = remap_lyr(ta_raw, 'total_assets_q')
    
    # 计算 GP = (Sales - COGS) / TA
    gp = (sales - cogs) / ta
    
    result_df = pd.DataFrame({'GP': gp})
    result_df = result_df.dropna()
    return result_df


def GPM(df):
    """
    Gross Profit Margin (销售毛利率)
    Formulation: GPM = (Sales - COGS) / Sales
    Description：衡量定价权与成本控制，每单位收入中利润的占比。
        高毛利率通常意味着强大的品牌、定价权或成本优势。
    数据字段：营业收入、营业成本
    注意：营业成本字段为 revenue_q
    """
    df = df.sort_index()
    
    # 使用上财年数据
    sales_raw = df['P($$revenue_q)'].fillna(0)
    # 营业成本
    cogs_raw = df['P($$oper_cost_q)'].fillna(0)
    
    sales = remap_lyr(sales_raw, 'revenue_q')
    cogs = remap_lyr(cogs_raw, 'oper_cost_q')
    
    # 避免除以0
    sales[sales == 0] = np.nan
    
    # 计算 GPM = (Sales - COGS) / Sales
    gpm = (sales - cogs) / sales
    
    result_df = pd.DataFrame({'GPM': gpm})
    result_df = result_df.dropna()
    return result_df


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
    ta_raw = df['P($$total_assets_q)'].fillna(0)
    
    # 总资产使用最新报告期数据
    ta = remap_lyr(ta_raw, 'total_assets_q')
    
    # 计算 ROA
    roa = earnings_ttm / ta
    
    result_df = pd.DataFrame({'ROA': roa})
    result_df = result_df.dropna()
    return result_df


# ==================== Investment Quality (投资质量) ====================

def AGRO(df):
    """
    Total Assets Growth Rate (总资产增长率)
    Formulation: AGRO = -(过去5年总资产对时间回归的斜率 / 平均总资产)
    Description：衡量资产扩张程度，增长过快的公司可能依赖并购或重资产扩张。
    数据字段：资产总计
    """
    df = df.sort_index()
    ta_raw = df['P($$total_assets_q)'].fillna(0)

    # 提取年度数据
    annual_ta = get_annual_data(ta_raw)

    # 对年度数据计算5年滚动增长率
    growth = calc_growth_rate_slope(annual_ta, window=5, min_periods=3)
    growth = growth.reset_index(level=0, drop=True)

    # 将年度结果映射回日频
    agro = -map_annual_to_daily(growth, df.index)

    result_df = pd.DataFrame({'AGRO': agro})
    result_df = result_df.dropna()
    return result_df


def IGRO(df):
    """
    Issuance Growth (股票发行量增长率)
    Formulation: IGRO = -(过去5年流通股本对时间回归的斜率 / 平均流通股本)
    Description：衡量股权稀释，频繁增发的公司对外部股权融资依赖度高。
    数据字段：流通股本 $circ_mv
    """
    df = df.sort_index()
    circ_mv = df['$circ_mv'].fillna(0)
    
    # 提取年度数据（每年最后一个交易日）
    annual_circ_mv = get_annual_data_year_end(circ_mv)
    
    # 对年度数据计算5年滚动增长率（斜率/均值）
    growth = calc_growth_rate_slope(annual_circ_mv, window=5, min_periods=3)
    growth = growth.reset_index(level=0, drop=True)
    
    # 将年度结果映射回日频，并取负号
    igro = -map_annual_to_daily(growth, df.index)
    
    result_df = pd.DataFrame({'IGRO': igro})
    result_df = result_df.dropna()
    return result_df


def CXGRO(df):
    """
    Capital Expenditure Growth (资本支出增长率)
    Formulation: CXGRO = -(过去5年资本支出对时间回归的斜率 / 平均资本支出)
    Description：衡量资本开支增速，增速过高的公司可能存在过度投资风险。
    数据字段：购建固定资产、无形资产和其他长期资产支付的现金
    """
    df = df.sort_index()
    capex_raw = df['P($$c_pay_acq_const_fiolta_q)'].fillna(0)

    # 提取年度数据
    annual_capex = get_annual_data(capex_raw)

    # 对年度数据计算5年滚动增长率
    growth = calc_growth_rate_slope(annual_capex, window=5, min_periods=3)
    growth = growth.reset_index(level=0, drop=True)

    # 将年度结果映射回日频
    cxgro = -map_annual_to_daily(growth, df.index)

    result_df = pd.DataFrame({'CXGRO': cxgro})
    result_df = result_df.dropna()
    return result_df

