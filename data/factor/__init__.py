
""" 表达式因子 """
from .factor_func import (
    MACD, BOLL, RSI_Multi, KDJ, DMI, WR, BIAS_Multi, CCI, ROC
)

""" 函数因子-CNE6 (当前38个) """
# 函数因子输入的df 索引为 <instrument, datetime>
# 1.规模因子
from .size import (
    LNCAP, MIDCAP
)

# 2.波动率因子
from .volatility import (
    BETA, HSIGMA, DASTD, CMRA
)

# 3.流动性因子
from .liquidity import (
    STOM, STOQ, STOA, ATVR
)

# 4.动量因子
from .momentum import (
    STREV, SEASON, INDMOM, RSTR, HALPHA
)

# 5.质量因子
from .quility import (
    MLEV, BLEV, DTOA,  # 杠杆
    VSAL, VERN, VFLO,  # 盈利波动性
    ABS, ACF,          # 盈利质量
    ATO, GP, GPM, ROA,  # 盈利能力,
    AGRO, IGRO, CXGRO,  # 投资能力
)

# 6.价值因子
from .value import (
    BTOP, ETOP, CETOP, EM,  LTRSTR, LTHALPHA
)

# 7.成长因子
from .growth import (
    EGRO, SGRO
)




""" 函数因子-自动构建 """
from .volatility import (
    VOLATILITY_20D
)

from .momentum import (
    MOM_10D, REVERSAL_5D, MOM_VOL_ADJ_10D
)



#  CNE6因子计算实例

'''
import qlib
from qlib.data import D
import pandas as pd
import numpy as np

# 导入所有 CNE6 因子
from data.factor.size import LNCAP, MIDCAP
from data.factor.volatility import BETA, HSIGMA, DASTD, CMRA
from data.factor.liquidity import STOM, STOQ, STOA, ATVR
from data.factor.momentum import STREV, SEASON, INDMOM, RSTR, HALPHA
from data.factor.quility import (
    MLEV, BLEV, DTOA,
    VSAL, VERN, VFLO,
    ABS, ACF,
    ATO, GP, GPM, ROA,
    AGRO, IGRO, CXGRO
)
from data.factor.value import BTOP, ETOP, CETOP, EM, LTRSTR, LTHALPHA
from data.factor.growth import EGRO, SGRO
from utils import PTTM

kwargs = {'custom_ops': [PTTM]}

# 定义所有待测试因子
FACTORS = {
    # 规模因子
    'LNCAP': LNCAP, 'MIDCAP': MIDCAP,
    # 波动率因子
    'BETA': BETA, 'HSIGMA': HSIGMA, 'DASTD': DASTD, 'CMRA': CMRA,
    # 流动性因子
    'STOM': STOM, 'STOQ': STOQ, 'STOA': STOA, 'ATVR': ATVR,
    # 动量因子
    'STREV': STREV, 'SEASON': SEASON, 'INDMOM': INDMOM, 'RSTR': RSTR, 'HALPHA': HALPHA,
    # 质量-杠杆
    'MLEV': MLEV, 'BLEV': BLEV, 'DTOA': DTOA,
    # 质量-盈利波动
    'VSAL': VSAL, 'VERN': VERN, 'VFLO': VFLO,
    # 质量-盈利质量
    'ABS': ABS, 'ACF': ACF,
    # 质量-盈利能力
    'ATO': ATO, 'GP': GP, 'GPM': GPM, 'ROA': ROA,
    # 质量-投资质量
    'AGRO': AGRO, 'IGRO': IGRO, 'CXGRO': CXGRO,
    # 价值因子
    'BTOP': BTOP, 'ETOP': ETOP, 'CETOP': CETOP, 'EM': EM, 'LTRSTR': LTRSTR, 'LTHALPHA': LTHALPHA,
    # 成长因子
    'EGRO': EGRO, 'SGRO': SGRO,
}


def _get_category(name):
    """获取因子所属类别"""
    categories = {
        'LNCAP': '规模', 'MIDCAP': '规模',
        'BETA': '波动率', 'HSIGMA': '波动率', 'DASTD': '波动率', 'CMRA': '波动率',
        'STOM': '流动性', 'STOQ': '流动性', 'STOA': '流动性', 'ATVR': '流动性',
        'STREV': '动量', 'SEASON': '动量', 'INDMOM': '动量', 'RSTR': '动量', 'HALPHA': '动量',
        'MLEV': '质量-杠杆', 'BLEV': '质量-杠杆', 'DTOA': '质量-杠杆',
        'VSAL': '质量-盈利波动', 'VERN': '质量-盈利波动', 'VFLO': '质量-盈利波动',
        'ABS': '质量-盈利质量', 'ACF': '质量-盈利质量',
        'ATO': '质量-盈利能力', 'GP': '质量-盈利能力', 'GPM': '质量-盈利能力', 'ROA': '质量-盈利能力',
        'AGRO': '质量-投资质量', 'IGRO': '质量-投资质量', 'CXGRO': '质量-投资质量',
        'BTOP': '价值', 'ETOP': '价值', 'CETOP': '价值', 'EM': '价值', 'LTRSTR': '价值', 'LTHALPHA': '价值',
        'EGRO': '成长', 'SGRO': '成长',
    }
    return categories.get(name, '其他')


if __name__ == '__main__':
    qlib.init(provider_uri='~/.qlib/qlib_data/custom_data_hfq', **kwargs)
    instruments = D.instruments(market='csi300')
    instruments = D.list_instruments(
        instruments, start_time='2020-01-01', end_time='2025-12-31', as_list=True
    )

    # 获取所需字段
    fields = [
        # 基础交易数据
        '$ind_one',                    # 一级行业分类code，类型为float，可能需要进行转换
        '$change',                     # 股票收盘价涨跌幅
        '$close',                      # 股票收盘价
        '$circ_mv',                    # 股票流通市值(万元)
        '$total_mv',                   # 股票总市值(万元)
        '$total_share',                # 总股本(万股)
        '$amount',                     # 成交额(元)

        # 资产负债表
        'P($$oth_eqt_tools_p_shr_q)',  # 其他权益工具(优先股)
        'P($$total_ncl_q)',            # 非流动负债合计
        'P($$total_hldr_eqy_exc_min_int_q)',  # 股东权益合计(不含少数股东权益)
        'P($$total_assets_q)',         # 资产总计
        'P($$total_liab_q)',           # 负债合计
        'P($$money_cap_q)',            # 货币资金

        # 利润表
        'P($$revenue_q)',              # 营业收入
        'P($$n_income_attr_p_q)',      # 净利润(不含少数股东损益)
        'P($$oper_cost_q)',            # 营业成本
        'P($$basic_eps_q)',            # 基本每股收益
        'P($$ebit_q)',                 # 息税前利润

        # 现金流量表
        'P($$n_cashflow_act_q)',       # 经营活动产生的现金流量净额
        'P($$depr_fa_coga_dpba_q)',    # 固定资产折旧、油气资产折耗、生产性生物资产折旧
        'P($$amort_intang_assets_q)',  # 无形资产摊销
        'P($$lt_amort_deferred_exp_q)',# 长期待摊费用摊销
        'P($$c_pay_acq_const_fiolta_q)',# 购建固定资产、无形资产和其他长期资产支付的现金

        # 借款相关
        'P($$st_borr_q)',              # 短期借款
        'P($$lt_borr_q)',              # 长期借款
        'P($$non_cur_liab_due_1y_q)',  # 一年内到期的非流动负债
        'P($$bond_payable_q)',         # 应付债券

        # TTM数据
        'PTTM($$revenue_q)',           # 营业收入(TTM)
        'PTTM($$n_income_attr_p_q)',   # 净利润(TTM)
        'PTTM($$n_cashflow_act_q)',    # 经营现金流(TTM)
    ]

    print("=" * 80)
    print("正在加载数据...")
    df = D.features(
        instruments[0:10], fields=fields,  # 10只股票
        start_time='2018-01-01', end_time='2026-03-02'  # 修改end_time为2026-03-02
    )
    print(f"数据加载完成: {df.shape[0]} 行, {df.shape[1]} 列")
    print("=" * 80)

    results = []
    print("\n开始计算因子...")
    print("-" * 80)

    for name, func in FACTORS.items():
        try:
            result = func(df)
            if result is not None and not result.empty:
                col_name = result.columns[0]
                values = result[col_name].dropna()
                mean_val = values.mean()
                std_val = values.std()
                count = len(values)
                results.append({
                    '因子名': name,
                    '类别': _get_category(name),
                    '样本数': count,
                    '均值': round(mean_val, 6),
                    '标准差': round(std_val, 6),
                })
                print(f"✓ {name:12s} | 样本数: {count:8d} | 均值: {mean_val:12.6f} | 标准差: {std_val:12.6f}")
            else:
                results.append({'因子名': name, '类别': _get_category(name), '样本数': 0, '均值': None, '标准差': None})
                print(f"✗ {name:12s} | 计算结果为空")
        except Exception as e:
            results.append({'因子名': name, '类别': _get_category(name), '样本数': 0, '均值': None, '标准差': None, '错误': str(e)})
            print(f"✗ {name:12s} | 错误: {str(e)[:50]}")

    print("-" * 80)
    print("\n" + "=" * 80)
    print("因子统计汇总表")
    print("=" * 80)

    df_result = pd.DataFrame(results)
    # 按类别排序
    category_order = ['规模', '波动率', '流动性', '动量', '质量-杠杆', '质量-盈利波动', '质量-盈利质量', '质量-盈利能力', '质量-投资质量', '价值', '成长']
    df_result['排序'] = df_result['类别'].map({c: i for i, c in enumerate(category_order)})
    df_result = df_result.sort_values('排序').drop('排序', axis=1).reset_index(drop=True)

    print(df_result.to_string(index=False))
    print("=" * 80)
'''

