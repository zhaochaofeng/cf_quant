"""
多信号Alpha预测框架

框架流程：
1. 对每个信号独立计算单信号Alpha（按情形1/情形2公式）
2. 多信号时通过Cholesky正交化去除信号间冗余
3. 按IC加权合成最终Alpha

当前版本: K=1（单信号），多信号正交化接口已预留
"""

from .config import (
    ROLLING_WINDOW,
    RESIDUAL_VOL_WINDOW,
    SCENARIO_WINDOW,
    R2_THRESHOLD,
    IC_LAG,
    OUTPUT_DIR,
)
from .data_loader import AlphaDataLoader
from .signal_processor import SignalProcessor
from .residual_vol import ResidualVolEstimator
from .scenario_classifier import ScenarioClassifier
from .ic_estimator import ICEstimator
from .orthogonalizer import AlphaOrthogonalizer
from .alpha_engine import AlphaEngine
from .output import AlphaOutputManager

__version__ = '1.0.0'

__all__ = [
    # 配置
    'ROLLING_WINDOW',
    'RESIDUAL_VOL_WINDOW',
    'SCENARIO_WINDOW',
    'R2_THRESHOLD',
    'IC_LAG',
    'OUTPUT_DIR',
    # 核心模块
    'AlphaDataLoader',
    'SignalProcessor',
    'ResidualVolEstimator',
    'ScenarioClassifier',
    'ICEstimator',
    'AlphaOrthogonalizer',
    'AlphaEngine',
    'AlphaOutputManager',
]
