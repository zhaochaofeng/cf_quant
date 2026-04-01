""" 字段质量分析脚本 - 分析2025年全年CSI300因子数据质量 """

import qlib
from qlib.data import D
import pandas as pd
import numpy as np
from utils import PTTM


def analyze_field_quality(df: pd.DataFrame) -> pd.DataFrame:
    """分析字段质量
    
    Args:
        df: 输入数据，MultiIndex (instrument, datetime)
        
    Returns:
        DataFrame: 质量分析结果
    """
    results = []
    total_count = df.shape[0]
    
    for col in df.columns:
        series = df[col]
        
        # 基础统计
        valid_count = series.notna().sum()
        missing_count = series.isna().sum()
        missing_pct = missing_count / total_count * 100
        
        # 零值统计 (仅针对有效非零值)
        valid_series = series.dropna()
        zero_count = (valid_series == 0).sum()
        zero_pct = zero_count / valid_count * 100 if valid_count > 0 else 0
        
        # 异常值检测 (IQR方法，3倍IQR)
        if valid_count > 0 and valid_series.dtype in [np.float64, np.float32]:
            q1 = valid_series.quantile(0.25)
            q3 = valid_series.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 3 * iqr
            upper_bound = q3 + 3 * iqr
            outlier_count = ((valid_series < lower_bound) | (valid_series > upper_bound)).sum()
            outlier_pct = outlier_count / valid_count * 100
        else:
            outlier_count = 0
            outlier_pct = 0
        
        # 基本统计量
        if valid_count > 0:
            mean_val = valid_series.mean()
            median_val = valid_series.median()
            std_val = valid_series.std()
            min_val = valid_series.min()
            max_val = valid_series.max()
        else:
            mean_val = median_val = std_val = min_val = max_val = np.nan
        
        results.append({
            '字段名': col,
            '总样本数': total_count,
            '有效样本': valid_count,
            '缺失数': missing_count,
            '缺失率%': round(missing_pct, 2),
            '零值数': zero_count,
            '零值率%': round(zero_pct, 2),
            '异常值数': outlier_count,
            '异常值率%': round(outlier_pct, 2),
            '均值': round(mean_val, 4) if not np.isnan(mean_val) else np.nan,
            '中位数': round(median_val, 4) if not np.isnan(median_val) else np.nan,
            '标准差': round(std_val, 4) if not np.isnan(std_val) else np.nan,
            '最小值': round(min_val, 4) if not np.isnan(min_val) else np.nan,
            '最大值': round(max_val, 4) if not np.isnan(max_val) else np.nan,
        })
    
    return pd.DataFrame(results)


if __name__ == '__main__':
    # 初始化qlib
    qlib.init(provider_uri='~/.qlib/qlib_data/custom_data_hfq', custom_ops=[PTTM])
    
    # 获取CSI300所有股票
    instruments = D.instruments(market='csi300')
    instruments = D.list_instruments(
        instruments, start_time='2025-01-01', end_time='2025-12-31', as_list=True
    )
    print(f"股票数量: {len(instruments)}")
    
    # 定义需要分析的字段
    fields = [
        # 基础交易数据
        '$ind_one', '$change', '$close', '$circ_mv', '$total_mv', '$total_share', '$amount',
        
        # 资产负债表
        'P($$oth_eqt_tools_p_shr_q)', 'P($$total_ncl_q)', 'P($$total_hldr_eqy_exc_min_int_q)',
        'P($$total_assets_q)', 'P($$total_liab_q)', 'P($$money_cap_q)',
        
        # 利润表
        'P($$revenue_q)', 'P($$n_income_attr_p_q)', 'P($$oper_cost_q)', 'P($$basic_eps_q)', 'P($$ebit_q)',
        
        # 现金流量表
        'P($$n_cashflow_act_q)', 'P($$depr_fa_coga_dpba_q)', 'P($$amort_intang_assets_q)',
        'P($$lt_amort_deferred_exp_q)', 'P($$c_pay_acq_const_fiolta_q)',
        
        # 借款相关
        'P($$st_borr_q)', 'P($$lt_borr_q)', 'P($$non_cur_liab_due_1y_q)', 'P($$bond_payable_q)',
        
        # TTM数据
        'PTTM($$revenue_q)', 'PTTM($$n_income_attr_p_q)', 'PTTM($$n_cashflow_act_q)',
    ]
    
    print(f"字段数量: {len(fields)}")
    print("正在加载数据...")
    
    # 加载2025年全年数据
    df = D.features(
        instruments, fields=fields,
        start_time='2025-01-01', end_time='2025-12-31'
    )
    print(f"数据加载完成: {df.shape[0]} 行, {df.shape[1]} 列")
    
    # 分析字段质量
    print("\n正在分析字段质量...")
    results = analyze_field_quality(df)
    
    # 打印结果
    print("\n" + "=" * 100)
    print("字段质量分析报告 (2025年 CSI300)")
    print("=" * 100)
    
    # 按缺失率排序显示
    results_sorted = results.sort_values('缺失率%', ascending=False)
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    
    print("\n【按缺失率排序】")
    print(results_sorted[['字段名', '总样本数', '有效样本', '缺失率%', '零值率%', '异常值率%']].to_string(index=False))
    
    print("\n\n【完整统计信息】")
    print(results[['字段名', '均值', '中位数', '标准差', '最小值', '最大值']].to_string(index=False))
    
    # 保存结果
    results.to_csv('/Users/chaofeng/code/cf_quant/field_quality_2025.csv', index=False, encoding='utf-8-sig')
    print("\n\n结果已保存至: field_quality_2025.csv")