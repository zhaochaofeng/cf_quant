"""
成长因子
"""
import pandas as pd

from .utils import get_annual_data, calc_growth_rate_slope, map_annual_to_daily


def EGRO(df):
    """
    Historical Earnings Per Share Growth Rate（每股收益增长率）
    
    Formulation: 过去5个财年每股收益（EPS）对时间回归的斜率 / 平均年EPS
    Description：反应已实现的盈利增长趋势，衡量公司过去的增长执行力。
    数据字段：基本每股收益 P($$basic_eps_q)
    """
    df = df.sort_index()
    
    # 获取基本每股收益（年度数据）
    eps = df['P($$basic_eps_q)']
    
    # 提取年度数据（每年1月第一个非空值，对应上一年年报）
    annual_eps = get_annual_data(eps)
    
    # 计算5年增长率（斜率/均值），至少需要3年数据
    growth = calc_growth_rate_slope(annual_eps, window=5, min_periods=3)
    
    # 重置索引，准备映射回日频
    growth = growth.reset_index(level=0, drop=True)
    
    # 将年度增长率映射回日频
    egro = map_annual_to_daily(growth, df.index)
    
    result_df = pd.DataFrame({'EGRO': egro})
    return result_df.dropna()


def SGRO(df):
    """
    Historical Sales Per Share Growth Rate（每股营业收入增长率）
    
    Formulation: 过去5个财年每股营收对时间回归的斜率 / 平均每股营收
    Description：描述已实现的营收增长趋势，衡量公司业务规模的扩张历史。
    数据字段：总股本 $total_share，营业收入 P($$revenue_q)
    """
    df = df.sort_index()
    
    # 获取总股本和营业收入
    total_share = df['$total_share']
    revenue = df['P($$revenue_q)']
    
    # 计算每股营收 = 营业收入 / 总股本
    # 注意：revenue 单位为元，total_share 单位为万股
    revenue_per_share = revenue / (total_share * 10000)
    
    # 提取年度数据
    annual_rps = get_annual_data(revenue_per_share)
    
    # 计算5年增长率（斜率/均值），至少需要3年数据
    growth = calc_growth_rate_slope(annual_rps, window=5, min_periods=3)
    
    # 重置索引，准备映射回日频
    growth = growth.reset_index(level=0, drop=True)
    
    # 将年度增长率映射回日频
    sgro = map_annual_to_daily(growth, df.index)
    
    result_df = pd.DataFrame({'SGRO': sgro})
    return result_df.dropna()
