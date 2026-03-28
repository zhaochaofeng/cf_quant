"""
投资组合优化模块

基于均值-方差优化框架的主动投资组合构建系统。

主要组件：
- PortfolioEngine: 主引擎，编排完整优化流程
- PortfolioDataLoader: 数据加载器
- QPOptimizer: 凸二次规划优化器
- NoTradeZoneIterator: 无交易区域迭代算法
- TradeGenerator: 交易指令生成器
- PortfolioOutputManager: 输出管理器

使用示例：
    from barra.portfolio import PortfolioEngine
    
    engine = PortfolioEngine(calc_date='2026-03-28')
    result = engine.run(portfolio_value=1e8)
    
    # 查看交易指令
    print(result.trade_orders)
"""

from .config import (
    OPTIMIZATION_PARAMS,
    ITERATION_PARAMS,
    OUTPUT_CONFIG,
    DATA_PATHS,
    DEFAULT_MARKET,
    DEFAULT_PORTFOLIO_VALUE
)

from .data_loader import PortfolioDataLoader
from .optimizer import QPOptimizer, OptimizationResult, compute_mcva
from .no_trade_zone import NoTradeZoneIterator, IterationResult, build_asset_covariance
from .trade_generator import TradeGenerator
from .output import PortfolioOutputManager
from .portfolio_engine import PortfolioEngine, PortfolioResult

__all__ = [
    # 配置
    'OPTIMIZATION_PARAMS',
    'ITERATION_PARAMS',
    'OUTPUT_CONFIG',
    'DATA_PATHS',
    'DEFAULT_MARKET',
    'DEFAULT_PORTFOLIO_VALUE',
    
    # 核心类
    'PortfolioEngine',
    'PortfolioResult',
    'PortfolioDataLoader',
    'QPOptimizer',
    'OptimizationResult',
    'NoTradeZoneIterator',
    'IterationResult',
    'TradeGenerator',
    'PortfolioOutputManager',
    
    # 工具函数
    'compute_mcva',
    'build_asset_covariance',
]

__version__ = '1.0.0'
