"""
成长因子
"""

from utils.dt import time_decorator
from .utils import (
    get_annual_data2,
    calc_growth_rate_slope,
    map_annual_to_daily,
    factor_output
)


@time_decorator
@factor_output
def EGRO(df):
    """
    Historical Earnings Per Share Growth Rate（每股收益增长率）
    
    Formulation: 过去5个财年每股收益（EPS）对时间回归的斜率 / 平均年EPS
    Description：反应已实现的盈利增长趋势，衡量公司过去的增长执行力。
    数据字段：基本每股收益 P($$basic_eps_q)
    """
    df = df.sort_index()
    
    # 基本每股收益
    eps = df['P($$basic_eps_a)']
    
    # 提取年度数据
    annual_eps = get_annual_data2(eps)
    
    # 计算5年增长率（斜率/均值），至少需要3年数据
    growth = calc_growth_rate_slope(annual_eps, window=5, min_periods=3)
    
    # 重置索引，准备映射回日频
    growth = growth.reset_index(level=0, drop=True)
    
    # 将年度增长率映射回日频
    egro = map_annual_to_daily(growth, df.index)

    return egro


@time_decorator
@factor_output
def SGRO(df):
    """
    Historical Sales Per Share Growth Rate（每股营业收入增长率）
    
    Formulation: 过去5个财年每股营收对时间回归的斜率 / 平均每股营收
    Description：描述已实现的营收增长趋势，衡量公司业务规模的扩张历史。
    数据字段：总股本 $total_share，营业收入 P($$revenue_q)
    """
    df = df.sort_index()

    total_share = df['$total_share']    # 总股本
    revenue = df['P($$revenue_a)']      # 营业收入

    rps = revenue / total_share
    annual_rps = get_annual_data2(rps)

    # 计算5年增长率（斜率/均值），至少需要3年数据
    growth = calc_growth_rate_slope(annual_rps, window=5, min_periods=3)

    # 重置索引，准备映射回日频
    growth = growth.reset_index(level=0, drop=True)
    
    # 将年度增长率映射回日频
    sgro = map_annual_to_daily(growth, df.index)

    return sgro
