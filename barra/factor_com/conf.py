"""
因子配置 — CNE6 风格因子、行业映射
"""

from data.factor import (
    LNCAP, MIDCAP,
    HBETA, HSIGMA, DASTD, CMRA,
    STOM, STOQ, STOA, ATVR,
    STREV, SEASON, INDMOM, RSTR, HALPHA,
    MLEV, BLEV, DTOA,
    VSAL, VERN, VFLO,
    ABS, ACF,
    ATO, GP, GPM, ROA,
    AGRO, IGRO, CXGRO,
    BTOP, ETOP, CETOP, EM, LTRSTR, LTHALPHA,
    EGRO, SGRO,
)

# Qlib 字段。分组存放
FIELD_GROUPS = [
    # 第1组: 基础交易数据（7个字段）
    {
        'name': 'base',
        'fields': [
            '$ind_one',        # 一级行业分类code
            '$close',          # 收盘价
            '$circ_mv',        # 流通市值（万元）
            '$total_mv',       # 总市值（万元）
            '$total_share',    # 总股本（万股）
            '$float_share',    # 流通股本（万股）
            '$amount',         # 成交额
        ]
    },
    # 第2组: 利润表（5个字段）
    {
        'name': 'income',
        'fields': [
            'P($$revenue_a)',              # 营业收入(年度)
            'P($$n_income_attr_p_a)',      # 净利润(不含少数股东损益)(年度)
            'P($$oper_cost_a)',            # 营业成本(年度)
            'P($$basic_eps_a)',            # 基本每股收益(年度)
            'P($$ebit_a)',                 # 息税前利润(年度)
        ]
    },
    # 第3组: 资产负债表（10个字段）
    {
        'name': 'balance',
        'fields': [
            'P($$oth_eqt_tools_p_shr_q)',          # 其他权益工具(优先股)
            'P($$total_hldr_eqy_exc_min_int_q)',   # 股东权益合计(不含少数股东权益)
            'P($$total_assets_q)',                 # 资产总计
            'P($$total_liab_q)',                   # 负债合计
            'P($$money_cap_q)',                    # 货币资金
            'P($$st_borr_q)',                      # 短期借款
            'P($$lt_borr_q)',                      # 长期借款
            'P($$non_cur_liab_due_1y_q)',          # 一年内到期的非流动负债
            'P($$bond_payable_q)',                 # 应付债券
            'P($$lt_payable_q)',                   # 长期应付款
        ]
    },
    # 第4组: 现金流量表（5个字段）
    {
        'name': 'cashflow',
        'fields': [
            'P($$n_cashflow_act_a)',               # 经营活动产生的现金流量净额(年度)
            'P($$depr_fa_coga_dpba_a)',            # 固定资产折旧、油气资产折耗、生产性生物资产折旧(年度)
            'P($$amort_intang_assets_a)',          # 无形资产摊销(年度)
            'P($$lt_amort_deferred_exp_a)',        # 长期待摊费用摊销(年度)
            'P($$c_pay_acq_const_fiolta_a)',       # 购建固定资产、无形资产和其他长期资产支付的现金(年度)
        ]
    },
    # 第5组: TTM（3个字段）
    {
        'name': 'TTM',
        'fields': [
            'PTTM($$revenue_q)',           # 营业收入(TTM)
            'PTTM($$n_income_attr_p_q)',   # 净利润(不含少数股东损益)(TTM)
            'PTTM($$n_cashflow_act_q)',    # 经营活动产生的现金流量净额(TTM)
        ]
    },
]


# 单位转换
UNIT_CONVERSION = {
    '$circ_mv': 10000,  # 流通市值（万元）
    '$total_mv': 10000,  # 总市值 （万元）
    '$total_share': 10000,  # 总股本（万股）
    '$float_share': 10000,  # 流通股本（万股）
}


# CNE6 风格因子定义（共38个）
CNE6_STYLE_FACTORS = {
    # 规模因子
    'size': ['LNCAP', 'MIDCAP'],
    # 波动率因子
    'volatility': ['HBETA', 'HSIGMA', 'DASTD', 'CMRA'],
    # 流动性因子
    'liquidity': ['STOM', 'STOQ', 'STOA', 'ATVR'],
    # 动量因子
    'momentum': ['STREV', 'SEASON', 'INDMOM', 'RSTR', 'HALPHA'],
    # 质量-杠杆因子
    'quality_leverage': ['MLEV', 'BLEV', 'DTOA'],
    # 质量-盈利波动因子
    'quality_earn_vol': ['VSAL', 'VERN', 'VFLO'],
    # 质量-盈利质量因子
    'quality_earn_quality': ['ABS', 'ACF'],
    # 质量-盈利能力因子
    'quality_profit': ['ATO', 'GP', 'GPM', 'ROA'],
    # 质量-投资质量因子
    'quality_invest': ['AGRO', 'IGRO', 'CXGRO'],
    # 价值因子
    'value': ['BTOP', 'ETOP', 'CETOP', 'EM', 'LTRSTR', 'LTHALPHA'],
    # 成长因子
    'growth': ['EGRO', 'SGRO'],
}

# 分类名称
CATEGORIES_MAP = {
    'size': '规模', 'volatility': '波动率', 'liquidity': '流动性',
    'momentum': '动量', 'quality_leverage': '质量-杠杆',
    'quality_earn_vol': '质量-盈利波动',
    'quality_earn_quality': '质量-盈利质量',
    'quality_profit': '质量-盈利能力',
    'quality_invest': '质量-投资质量',
    'value': '价值', 'growth': '成长',
}

# 因子名称列表
STYLE_FACTOR_LIST = []
for category, factors in CNE6_STYLE_FACTORS.items():
    STYLE_FACTOR_LIST.extend(factors)

# 暂时排除的因子
# exclude_factors = [
#     'VFLO', 'ROA', 'AGRO', 'VSAL', 'VERN', 'CXGRO',
#     'EGRO', 'SGRO', 'DTOA', 'BTOP', 'GPM', 'MLEV',
#     'BLEV', 'GP', 'ACF', 'ABS', 'EM', 'LTHALPHA'
# ]
# # 多重共线性排除的因子
# exclude_vif = ['LTRSTR', 'STOQ']
#
# STYLE_FACTOR_LIST = [f for f in STYLE_FACTOR_LIST if f not in exclude_factors]
# STYLE_FACTOR_LIST = [f for f in STYLE_FACTOR_LIST if f not in exclude_vif]

# 因子计算函数字典
FACTOR_FUNCTIONS = {
    'LNCAP': LNCAP,
    'MIDCAP': MIDCAP,
    'HBETA': HBETA,
    'HSIGMA': HSIGMA,
    'DASTD': DASTD,
    'CMRA': CMRA,
    'STOM': STOM,
    'STOQ': STOQ,
    'STOA': STOA,
    'ATVR': ATVR,
    'STREV': STREV,
    'SEASON': SEASON,
    'INDMOM': INDMOM,
    'RSTR': RSTR,
    'HALPHA': HALPHA,
    'MLEV': MLEV,
    'BLEV': BLEV,
    'DTOA': DTOA,
    'VSAL': VSAL,
    'VERN': VERN,
    'VFLO': VFLO,
    'ABS': ABS,
    'ACF': ACF,
    'ATO': ATO,
    'GP': GP,
    'GPM': GPM,
    'ROA': ROA,
    'AGRO': AGRO,
    'IGRO': IGRO,
    'CXGRO': CXGRO,
    'BTOP': BTOP,
    'ETOP': ETOP,
    'CETOP': CETOP,
    'EM': EM,
    'LTRSTR': LTRSTR,
    'LTHALPHA': LTHALPHA,
    'EGRO': EGRO,
    'SGRO': SGRO,
}


# 行业代码映射（申万一级行业）
INDUSTRY_MAPPING = {
    '801780': '银行',
    '801180': '房地产',
    '801230': '综合',
    '801750': '计算机',
    '801970': '环保',
    '801200': '商贸零售',
    '801890': '机械设备',
    '801730': '电力设备',
    '801720': '建筑装饰',
    '801710': '建筑材料',
    '801030': '基础化工',
    '801110': '家用电器',
    '801130': '纺织服饰',
    '801010': '农林牧渔',
    '801080': '电子',
    '801160': '公用事业',
    '801150': '医药生物',
    '801880': '汽车',
    '801210': '社会服务',
    '801960': '石油石化',
    '801050': '有色金属',
    '801770': '通信',
    '801170': '交通运输',
    '801760': '传媒',
    '801790': '非银金融',
    '801140': '轻工制造',
    '801740': '国防军工',
    '801120': '食品饮料',
    '801950': '煤炭',
    '801040': '钢铁',
    '801980': '美容护理',
}

INDUSTRY_CODES = list(INDUSTRY_MAPPING.keys())
INDUSTRY_NAMES = list(INDUSTRY_MAPPING.values())
