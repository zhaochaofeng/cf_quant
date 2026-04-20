"""
Barra CNE6 风险模型配置
"""


from data.factor import (
    LNCAP, MIDCAP,
    BETA, HSIGMA, DASTD, CMRA,
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
        'name': '基础交易数据',
        'fields': [
            '$ind_one', '$change', '$close', '$circ_mv',
            '$total_mv', '$total_share', '$float_share', '$amount',
        ]
    },
    # 第2组: 资产负债表（6个字段）
    {
        'name': '资产负债表',
        'fields': [
            'P($$oth_eqt_tools_p_shr_q)', 'P($$total_ncl_q)',
            'P($$total_hldr_eqy_exc_min_int_q)', 'P($$total_assets_q)',
            'P($$total_liab_q)', 'P($$money_cap_q)'
        ]
    },
    # 第3组: 利润表（5个字段）
    {
        'name': '利润表',
        'fields': [
            'P($$revenue_q)', 'P($$n_income_attr_p_q)',
            'P($$oper_cost_q)', 'P($$basic_eps_q)', 'P($$ebit_q)'
        ]
    },
    # 第4组: 现金流量表（5个字段）
    {
        'name': '现金流量表',
        'fields': [
            'P($$n_cashflow_act_q)', 'P($$depr_fa_coga_dpba_q)',
            'P($$amort_intang_assets_q)', 'P($$lt_amort_deferred_exp_q)',
            'P($$c_pay_acq_const_fiolta_q)'
        ]
    },
    # 第5组: 借款相关（4个字段）
    {
        'name': '借款相关',
        'fields': [
            'P($$st_borr_q)', 'P($$lt_borr_q)',
            'P($$non_cur_liab_due_1y_q)', 'P($$bond_payable_q)'
        ]
    },
    # 第6组: TTM数据（3个字段）
    {
        'name': 'TTM数据',
        'fields': [
            'PTTM($$revenue_q)', 'PTTM($$n_income_attr_p_q)',
            'PTTM($$n_cashflow_act_q)'
        ]
    },
]


# CNE6 风格因子定义（共38个）
CNE6_STYLE_FACTORS = {
    # 规模因子
    'size': ['LNCAP', 'MIDCAP'],
    # 波动率因子
    'volatility': ['BETA', 'HSIGMA', 'DASTD', 'CMRA'],
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


# 因子名称列表
STYLE_FACTOR_LIST = []
for category, factors in CNE6_STYLE_FACTORS.items():
    STYLE_FACTOR_LIST.extend(factors)

# 暂时排除的因子
exclude_factors = [
    'VFLO', 'ROA', 'AGRO', 'VSAL', 'VERN', 'CXGRO',
    'EGRO', 'SGRO', 'DTOA', 'BTOP', 'GPM', 'MLEV',
    'BLEV', 'GP', 'ACF', 'ABS'
]

STYLE_FACTOR_LIST = [f for f in STYLE_FACTOR_LIST if f not in exclude_factors]


# 因子计算函数字典
FACTOR_FUNCTIONS = {
    'LNCAP': LNCAP,
    'MIDCAP': MIDCAP,
    'BETA': BETA,
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

# 模型参数
MODEL_PARAMS = {
    # 日期窗口（交易日）
    'window': 252,
    # ARMA模型阶数
    'arma_p': 1,
    'arma_q': 1,
    # Barra 半衰期协方差参数
    'half_life_corr': 252,     # 相关系数半衰期 H_C（交易日）
    'half_life_var': 42,       # 方差半衰期 H_D（交易日）
    'ewma_init_periods': 180,   # 初始化等权样本协方差窗口 m
    # 特异风险面板回归窗口
    'panel_regression_window': 120,  # 混合回归窗口（交易日）
    # 数据时间延长（因子计算需要历史回溯，如SEASON需5年）
    'data_extend_years': 6,    # 原始字段加载时 start_time 前移年数
}

# 输出配置
OUTPUT_CONFIG = {
    'stock_risk_filename': 'stock_risk_{date}.csv',
    'factor_risk_filename': 'factor_risk_{date}.csv',
    'float_precision': 6,
    'encoding': 'utf-8',
}

# qlib数据字段
QLIB_FIELDS = {
    'industry': '$ind_one',
    'change': '$change',
    'close': '$close',
    'circ_mv': '$circ_mv',
    'total_mv': '$total_mv',
}

# 基准配置
BENCHMARK_CONFIG = {
    'csi300': {
        'market': 'csi300',
        'name': '沪深300',
    },
    'BENCHMARK': 'SH000300'
}


