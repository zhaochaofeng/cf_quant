"""
Barra CNE6 风险模型包

完整的多因子风险模型实现，日频更新：
- 因子暴露矩阵构建（CNE6 38个风格因子 + 31个行业因子）
- 横截面回归估计因子收益率（WLS）
- 因子协方差矩阵估计（Barra 双半衰期 EWMA）
- 特异风险矩阵估计（ARMA + WLS 面板回归）
- 资产协方差矩阵计算（V = X·F·X^T + Δ）
- 风险归因分析（MCAR/RCAR/FMCAR/FRCAR）
- CSV + MySQL 输出

使用方式:
    # 命令行运行（推荐）
    python barra/risk_control/run.py --date 2025-03-01 --history-months 85

    # Python 调用
    from barra.risk_control.run import run
    results = run(calc_date='2025-03-01', history_months=85, n_jobs=4)
"""

from .config import (
    CNE6_STYLE_FACTORS,
    STYLE_FACTOR_LIST,
    INDUSTRY_MAPPING,
    MODEL_PARAMS,
    OUTPUT_CONFIG,
)

from .data_loader import DataLoader
from .portfolio import PortfolioManager
from .covariance import FactorCovarianceEstimator
from .specific_risk import SpecificRiskEstimator
from .risk_model import AssetCovarianceCalculator
from .risk_attribution import RiskAttributionAnalyzer
from .output import RiskOutputManager

__version__ = '2.0.0'

__all__ = [
    # 配置
    'CNE6_STYLE_FACTORS',
    'STYLE_FACTOR_LIST',
    'INDUSTRY_MAPPING',
    'MODEL_PARAMS',
    'OUTPUT_CONFIG',
    # 核心模块
    'DataLoader',
    'PortfolioManager',
    'FactorCovarianceEstimator',
    'SpecificRiskEstimator',
    'AssetCovarianceCalculator',
    'RiskAttributionAnalyzer',
    'RiskOutputManager',
]
