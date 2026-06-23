"""
因子评估配置常量
"""


# --- 分层回测 ---
DEFAULT_N_GROUPS = 5  # 默认分组数

# --- 信号衰减 ---
DEFAULT_MAX_DECAY_LAG = 100  # 收益率最大滞后期不能超过计算时间周期(T)，否则超过的 forward_ret_{k} 元素全部为NaN
