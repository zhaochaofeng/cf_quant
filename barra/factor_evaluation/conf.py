"""
因子评估配置常量
"""
from typing import Literal

# --- 截面IC ---
DEFAULT_IC_PERIOD = 1  # 默认IC周期（1日）
DEFAULT_IC_METHOD: Literal['spearman', 'pearson'] = 'spearman'

# --- 分层回测 ---
DEFAULT_N_GROUPS = 5  # 默认分组数
DEFAULT_WEIGHTING: Literal['equal', 'market_cap'] = 'equal'

# --- 信号衰减 ---
DEFAULT_MAX_DECAY_LAG = 100  # 收益率最大滞后期不能超过计算时间周期，否则超过的 forward_ret_{k} 元素全部为NaN
