"""
投资组合优化配置模块
"""
from pathlib import Path

# 基础路径
BASE_DIR = Path(__file__).parent.parent.parent
OUTPUT_DIR = BASE_DIR / 'barra' / 'portfolio' / 'output'
RISK_OUTPUT_DIR = BASE_DIR / 'barra' / 'risk_control' / 'output'
ALPHA_OUTPUT_DIR = BASE_DIR / 'barra' / 'alpha' / 'output'

# 优化参数
OPTIMIZATION_PARAMS = {
    'risk_aversion': 0.05,           # λ，风险厌恶系数（百分数单位）
    'buy_cost_rate': 0.0003,         # c_b，买入成本率 0.03%
    'sell_cost_rate': 0.0013,        # c_s，卖出成本率 0.13%
    'max_turnover': 0.10,            # T_max，换手率上限 10%
    'max_active_position': 0.05,     # U_n，个股主动头寸上限 5%
    'min_trade_threshold': 1e-5,     # 最小交易阈值
}

# 迭代参数
ITERATION_PARAMS = {
    'max_iterations': 100,           # 最大迭代次数
    'convergence_threshold': 1e-6,   # 收敛阈值
}

# 输出配置
OUTPUT_CONFIG = {
    'output_dir': str(OUTPUT_DIR),
    'trade_order_filename': 'trade_order_{date}.parquet',
    'position_filename': 'portfolio_position_{date}.parquet',
    'log_filename': 'optimization_log_{date}.parquet',
    'float_precision': 6,
    'encoding': 'utf-8',
}

# 数据路径配置
DATA_PATHS = {
    'exposure': 'debug/exposure_matrix.parquet',
    'factor_cov': 'model/factor_covariance.parquet',
    'specific_risk': 'model/specific_risk.parquet',
    'alpha': 'alpha_{date}.parquet',
}

# 默认市场
DEFAULT_MARKET = 'csi300'

# 默认组合净值（元）
DEFAULT_PORTFOLIO_VALUE = 1e8
