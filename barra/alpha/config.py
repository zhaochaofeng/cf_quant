"""
多信号Alpha预测框架 - 配置常量
"""
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent

# 滚动窗口（交易日数）
# ROLLING_WINDOW = 750            # 3年，用于IC估计、协方差估计
# RESIDUAL_VOL_WINDOW = 500       # 2年，用于历史残差波动率
# SCENARIO_WINDOW = 500           # M=500，用于情形判断回归
# MIN_IC_WINDOW = 250             # IC估计最少所需天数
ROLLING_WINDOW = 200            # 3年，用于IC估计、协方差估计
RESIDUAL_VOL_WINDOW = 200       # 2年，用于历史残差波动率
SCENARIO_WINDOW = 200           # M=500，用于情形判断回归
MIN_IC_WINDOW = 200             # IC估计最少所需天数

# 情形判断阈值
R2_THRESHOLD = 0.2              # R^2 > 0.2 判定为 Case 2
P_VALUE_THRESHOLD = 0.05        # 系数b显著性阈值

# 残差收益率滞后期
IC_LAG = 2                      # t+2日残差收益率

# 新股判定：残差数据少于此天数视为新股，默认500
# NEW_STOCK_MIN_DAYS = 500
NEW_STOCK_MIN_DAYS = 200

# 残差波动率滚动窗口（交易日数），老股取最近N天的时间序列std
VOL_ROLLING_WINDOW = 50

# 数据源
SIGNAL_TABLE = 'monitor_return_rate'
RESIDUALS_PATH = str(PROJECT_ROOT / 'barra/risk_control/output/model/residuals.parquet')

# 输出
OUTPUT_DIR = str(PROJECT_ROOT / 'barra/alpha/output')

# Qlib
QLIB_PROVIDER_URI = '~/.qlib/qlib_data/custom_data_hfq'
