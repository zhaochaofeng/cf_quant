"""
Barra CNE6 风险模型包 - 内存优化版本

完整的多因子风险模型实现，包含：
- 因子暴露矩阵构建（CNE6 38个风格因子 + 31个行业因子）
- 横截面回归估计因子收益率
- 因子协方差矩阵估计
- 特异风险矩阵估计（ARMA + 面板回归）
- 风险归因分析（MCAR/RCAR/FMCAR/FRCAR）
- CSV文件输出

【内存优化说明】
针对8GB RAM内存约束进行了优化：
1. 使用float32替代float64（节省50%内存）
2. 分批处理数据（股票分批、日期分批）
3. 增量模式支持（磁盘缓存）
4. 自动并行进程数调整
5. 内存监控和警告

使用示例（内存优化版本）:
    # 方式1: 使用优化引擎（推荐，适用于8GB内存）
    from barra.risk_control import BarraRiskEngine
    
    engine = BarraRiskEngine(
        calc_date='2024-03-14',
        portfolio_input='random',
        output_dir='output',
        memory_threshold_gb=6.0,  # 内存阈值
        use_incremental=False     # 是否使用磁盘缓存
    )
    
    # 月频更新 - 使用分批处理
    engine.run_monthly_update(
        '2014-03-01', '2024-03-01',
        stock_batch_size=100,  # 每批处理100只股票
        date_batch_size=10     # 每批处理10天
    )
    
    # 日频计算
    engine.run_daily_risk()
    engine.save_results()
    
    # 方式2: 内存不足时使用增量模式（磁盘缓存）
    engine = BarraRiskEngine(
        calc_date='2024-03-14',
        use_incremental=True  # 启用磁盘缓存
    )
    engine.run_monthly_update('2014-03-01', '2024-03-01')

内存监控工具:
    from barra.risk_control import MemoryMonitor, suggest_workers_by_memory
    
    # 监控内存使用
    monitor = MemoryMonitor(threshold_gb=6.0)
    monitor.print_memory_status("当前状态")
    
    # 根据内存自动调整并行进程数
    n_jobs = suggest_workers_by_memory(
        max_workers=8,
        memory_per_worker_gb=1.0,
        reserve_memory_gb=2.0
    )
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

# 导入内存优化版本（默认使用）
from .memory_utils import (
    MemoryMonitor,
    optimize_memory,
    convert_to_float32,
    optimize_dataframe_memory,
    suggest_workers_by_memory,
)

from .factor_exposure_optimized import FactorExposureBuilder
from .cross_sectional_optimized import CrossSectionalRegression
from .barra_engine_optimized import BarraRiskEngine

# 如果需要原始版本，可以直接导入：
# from .factor_exposure import FactorExposureBuilder as FactorExposureBuilderOriginal
# from .cross_sectional import CrossSectionalRegression as CrossSectionalRegressionOriginal
# from .barra_engine import BarraRiskEngine as BarraRiskEngineOriginal

__version__ = '1.1.0-memory-optimized'

__all__ = [
    # 配置
    'CNE6_STYLE_FACTORS',
    'STYLE_FACTOR_LIST',
    'INDUSTRY_MAPPING',
    'MODEL_PARAMS',
    'OUTPUT_CONFIG',
    # 内存工具
    'MemoryMonitor',
    'optimize_memory',
    'convert_to_float32',
    'optimize_dataframe_memory',
    'suggest_workers_by_memory',
    # 核心模块（内存优化版本）
    'BarraRiskEngine',
    'DataLoader',
    'PortfolioManager',
    'FactorExposureBuilder',
    'CrossSectionalRegression',
    'FactorCovarianceEstimator',
    'SpecificRiskEstimator',
    'AssetCovarianceCalculator',
    'RiskAttributionAnalyzer',
    'RiskOutputManager',
]
