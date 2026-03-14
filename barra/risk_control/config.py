"""
Barra CNE6 风险模型配置
"""

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

# 展平因子列表
STYLE_FACTOR_LIST = []
for category, factors in CNE6_STYLE_FACTORS.items():
    STYLE_FACTOR_LIST.extend(factors)

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
    # 历史窗口（月）
    'history_window': 120,  # 10年
    # ARMA模型阶数
    'arma_p': 1,
    'arma_q': 1,
    # 加权回归权重
    'weight_type': 'sqrt_market_cap',  # 市值平方根
    # 去极值参数
    'winsorize_method': 'median',  # 中位数去极值
    'winsorize_lower': 0.01,
    'winsorize_upper': 0.99,
    # 标准化参数
    'standardize_mean': 0,
    'standardize_std': 1,
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
    }
}
