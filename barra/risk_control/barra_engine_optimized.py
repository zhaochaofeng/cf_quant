"""
Barra CNE6 风险模型主引擎 - 内存优化版本
针对8GB RAM约束优化
"""
import pandas as pd
import numpy as np
import gc
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Union

# 导入各模块
from .data_loader import DataLoader
from .portfolio import PortfolioManager
from .memory_utils import (
    MemoryMonitor, optimize_memory, convert_to_float32,
    clear_variables, suggest_workers_by_memory
)

# 使用优化版本的模块
from .factor_exposure_optimized import FactorExposureBuilder
from .cross_sectional_optimized import CrossSectionalRegression
from .covariance import FactorCovarianceEstimator
from .specific_risk import SpecificRiskEstimator
from .risk_model import AssetCovarianceCalculator
from .risk_attribution import RiskAttributionAnalyzer
from .output import RiskOutputManager
from .config import MODEL_PARAMS, STYLE_FACTOR_LIST, INDUSTRY_MAPPING


class BarraRiskEngine:
    """Barra CNE6 风险模型引擎 - 内存优化版本"""
    
    def __init__(self, calc_date: str, 
                 portfolio_input: Union[str, Dict, pd.Series] = 'random',
                 market: str = 'csi300',
                 output_dir: str = 'output',
                 cache_dir: Optional[str] = None,
                 n_jobs: int = 1,
                 memory_threshold_gb: float = 6.0,
                 use_incremental: bool = False):
        """
        初始化风险模型引擎（内存优化版本）
        
        Args:
            calc_date: 计算日期，格式'YYYY-MM-DD'
            portfolio_input: 投资组合输入
            market: 市场代码
            output_dir: 输出目录
            cache_dir: 缓存目录
            n_jobs: 并行进程数（会根据内存自动调整）
            memory_threshold_gb: 内存阈值（GB）
            use_incremental: 是否使用增量模式（磁盘缓存）
        """
        self.calc_date = calc_date
        self.market = market
        self.use_incremental = use_incremental
        
        # 自动调整并行进程数
        self.n_jobs = suggest_workers_by_memory(
            max_workers=n_jobs,
            memory_per_worker_gb=1.0,
            reserve_memory_gb=2.0
        )
        print(f"根据可用内存自动调整并行进程数为: {self.n_jobs}")
        
        # 内存监控
        self.memory_monitor = MemoryMonitor(threshold_gb=memory_threshold_gb)
        
        # 初始化各模块
        self.data_loader = DataLoader(market=market)
        self.portfolio_manager = PortfolioManager(market=market)
        self.factor_builder = FactorExposureBuilder(
            cache_dir=cache_dir,
            memory_threshold_gb=memory_threshold_gb * 0.8
        )
        self.cross_sectional = CrossSectionalRegression(
            memory_threshold_gb=memory_threshold_gb * 0.7
        )
        self.covariance_estimator = FactorCovarianceEstimator(
            history_window=MODEL_PARAMS['history_window']
        )
        self.specific_risk_estimator = SpecificRiskEstimator(
            arma_order=(MODEL_PARAMS['arma_p'], MODEL_PARAMS['arma_q'])
        )
        self.asset_cov_calculator = AssetCovarianceCalculator()
        self.risk_attribution = RiskAttributionAnalyzer()
        self.output_manager = RiskOutputManager(output_dir=output_dir)
        
        # 加载组合
        self.portfolio_weights = self._load_portfolio(portfolio_input)
        
        # 存储中间结果
        self.factor_exposure = None
        self.factor_returns = None
        self.factor_covariance = None
        self.specific_risk = None
        self.asset_covariance = None
        self.benchmark_weights = None
        self.risk_results = None
        
        # 增量模式配置
        self.incremental_dir = Path(output_dir) / 'incremental_cache'
        if use_incremental:
            self.incremental_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_portfolio(self, portfolio_input) -> pd.Series:
        """加载投资组合"""
        if isinstance(portfolio_input, str) and portfolio_input.lower() == 'random':
            print(f"生成随机组合（50只股票）...")
            return self.portfolio_manager.generate_random_portfolio(
                self.calc_date, n_stocks=50, random_state=42
            )
        else:
            return self.portfolio_manager.load_portfolio(portfolio_input, self.calc_date)
    
    def run_monthly_update(self, start_date: str, end_date: str,
                          stock_batch_size: int = 100,
                          date_batch_size: int = 10) -> None:
        """
        月频更新：估计因子收益率、协方差矩阵、特异风险 - 内存优化版本
        
        Args:
            start_date: 历史数据开始日期
            end_date: 历史数据结束日期
            stock_batch_size: 股票批处理大小
            date_batch_size: 日期批处理大小
        """
        print("\n" + "=" * 70)
        print("开始月频模型更新（内存优化模式）...")
        print(f"历史数据区间: {start_date} 至 {end_date}")
        print(f"批处理配置: 股票{stock_batch_size}/批, 日期{date_batch_size}/批")
        
        self.memory_monitor.print_memory_status("月频更新开始前")
        
        # 1. 加载数据
        print("\n1. 加载数据...")
        instruments = self.data_loader.get_instruments(start_date, end_date)
        print(f"   股票数量: {len(instruments)}")
        
        # 估算内存使用
        estimated_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
        print(f"   预估交易日数: ~{estimated_days}")
        
        # 分批加载数据
        raw_data = self.data_loader.load_factor_data(instruments, start_date, end_date)
        returns_df = self.data_loader.load_returns(instruments, start_date, end_date)
        industry_df = self.data_loader.load_industry(instruments, start_date, end_date)
        market_cap_df = self.data_loader.load_market_cap(instruments, start_date, end_date)
        
        self.memory_monitor.print_memory_status("数据加载完成")
        
        # 2. 构建因子暴露矩阵
        print("\n2. 构建因子暴露矩阵...")
        
        if self.use_incremental:
            # 增量模式：使用磁盘缓存
            print("使用增量模式（磁盘缓存）...")
            self.factor_exposure = self.factor_builder.build_exposure_matrix_incremental(
                raw_data, industry_df, market_cap_df,
                output_dir=str(self.incremental_dir / 'exposure'),
                n_jobs=self.n_jobs,
                date_batch_size=date_batch_size
            )
        else:
            # 内存模式：分批处理
            self.factor_exposure = self.factor_builder.build_exposure_matrix(
                raw_data, industry_df, market_cap_df,
                n_jobs=self.n_jobs,
                stock_batch_size=stock_batch_size,
                date_batch_size=date_batch_size
            )
        
        # 释放原始数据内存
        clear_variables(raw_data)
        self.memory_monitor.print_memory_status("因子暴露矩阵构建完成")
        
        # 3. 横截面回归估计因子收益率
        print("\n3. 横截面回归...")
        
        if self.use_incremental:
            self.factor_returns = self.cross_sectional.fit_multi_periods_incremental(
                returns_df, self.factor_exposure, market_cap_df,
                output_dir=str(self.incremental_dir / 'regression'),
                dates_per_batch=date_batch_size
            )
        else:
            self.factor_returns = self.cross_sectional.fit_multi_periods(
                returns_df, self.factor_exposure, market_cap_df,
                dates_per_batch=date_batch_size
            )
        
        self.memory_monitor.print_memory_status("横截面回归完成")
        
        # 4. 估计因子协方差矩阵
        print("\n4. 估计因子协方差矩阵...")
        self.factor_covariance = self.covariance_estimator.estimate_sample_covariance(
            self.factor_returns
        )
        # 转换类型节省内存
        self.factor_covariance = convert_to_float32(self.factor_covariance)
        
        self.memory_monitor.print_memory_status("因子协方差估计完成")
        
        # 5. 估计特异风险矩阵
        print("\n5. 估计特异风险矩阵...")
        residuals_df = self.cross_sectional.get_residuals()
        
        # 如果需要，使用增量保存
        if self.use_incremental and len(residuals_df) > 100000:
            resid_file = self.incremental_dir / 'residuals.csv'
            self.cross_sectional.save_residuals_incremental(
                str(resid_file), chunk_size=50
            )
            # 重新加载（可选）
            residuals_df = pd.read_csv(resid_file, index_col=[0, 1])
        
        specific_risk_df = self.specific_risk_estimator.estimate_specific_risk(
            residuals_df, self.factor_exposure
        )
        
        # 转换为Series并优化类型
        self.specific_risk = pd.Series(
            specific_risk_df['specific_var'].values.astype(np.float32),
            index=specific_risk_df['instrument']
        )
        
        # 清理残差数据释放内存
        self.cross_sectional.clear_memory()
        clear_variables(residuals_df, specific_risk_df)
        
        self.memory_monitor.print_memory_status("特异风险估计完成")
        
        print("\n月频模型更新完成")
        print("=" * 70)
        
        # 最终内存优化
        optimize_memory()
    
    def run_daily_risk(self, exposure: Optional[pd.DataFrame] = None,
                      factor_cov: Optional[pd.DataFrame] = None,
                      specific_risk: Optional[pd.Series] = None) -> dict:
        """
        日频风险计算：计算MCAR/RCAR/FMCAR/FRCAR - 内存优化版本
        
        Args:
            exposure: 当日因子暴露矩阵（可选）
            factor_cov: 因子协方差矩阵（可选）
            specific_risk: 特异风险（可选）
            
        Returns:
            风险分析结果字典
        """
        print("\n" + "=" * 70)
        print(f"开始日频风险计算: {self.calc_date}（内存优化模式）")
        
        self.memory_monitor.print_memory_status("日频风险计算开始前")
        
        # 使用传入的参数或已有数据
        if exposure is not None:
            self.factor_exposure = exposure
        if factor_cov is not None:
            self.factor_covariance = factor_cov
        if specific_risk is not None:
            self.specific_risk = specific_risk
        
        # 检查数据是否就绪
        if self.factor_exposure is None:
            raise ValueError("因子暴露矩阵未就绪，请先运行月频更新")
        if self.factor_covariance is None:
            raise ValueError("因子协方差矩阵未就绪，请先运行月频更新")
        if self.specific_risk is None:
            raise ValueError("特异风险矩阵未就绪，请先运行月频更新")
        
        # 转换数据类型
        self.factor_covariance = convert_to_float32(self.factor_covariance)
        self.specific_risk = self.specific_risk.astype(np.float32)
        
        # 1. 获取基准权重
        print("\n1. 获取基准权重...")
        self.benchmark_weights = self.portfolio_manager.get_benchmark_weights(self.calc_date)
        print(f"   基准股票数: {len(self.benchmark_weights)}")
        
        self.memory_monitor.print_memory_status("基准权重获取完成")
        
        # 2. 计算资产协方差矩阵
        print("\n2. 计算资产协方差矩阵...")
        
        # 提取当日的因子暴露
        if isinstance(self.factor_exposure.index, pd.MultiIndex):
            today_exposure = self.factor_exposure.xs(self.calc_date, level=1)
        else:
            today_exposure = self.factor_exposure
        
        today_exposure = convert_to_float32(today_exposure)
        
        self.asset_covariance = self.asset_cov_calculator.calculate(
            today_exposure,
            self.factor_covariance,
            self.specific_risk
        )
        
        self.asset_covariance = convert_to_float32(self.asset_covariance)
        self.memory_monitor.print_memory_status("资产协方差计算完成")
        
        # 3. 风险归因分析
        print("\n3. 风险归因分析...")
        self.risk_results = self.risk_attribution.analyze_risk(
            self.asset_covariance,
            self.factor_covariance,
            today_exposure,
            self.portfolio_weights,
            self.benchmark_weights
        )
        
        # 转换结果类型
        for key in ['mcar', 'rcar', 'fmcar', 'frcar']:
            if key in self.risk_results and isinstance(self.risk_results[key], pd.Series):
                self.risk_results[key] = self.risk_results[key].astype(np.float32)
        
        self.memory_monitor.print_memory_status("风险归因完成")
        
        print("\n日频风险计算完成")
        print("=" * 70)
        
        # 清理临时数据
        clear_variables(today_exposure)
        optimize_memory()
        
        return self.risk_results
    
    def save_results(self) -> Dict[str, str]:
        """
        保存风险指标到CSV文件 - 内存优化版本
        
        Returns:
            保存的文件路径字典
        """
        if self.risk_results is None:
            raise ValueError("风险结果未计算，请先运行日频风险计算")
        
        print("\n保存风险指标到文件...")
        self.memory_monitor.print_memory_status("保存结果前")
        
        # 1. 保存股票风险指标
        stock_file = self.output_manager.save_stock_risk(
            self.risk_results['mcar'],
            self.risk_results['rcar'],
            self.calc_date
        )
        print(f"   股票风险指标: {stock_file}")
        
        # 2. 准备因子类型映射
        factor_types = self._get_factor_types()
        
        # 3. 保存因子风险指标
        factor_file = self.output_manager.save_factor_risk(
            self.risk_results['fmcar'],
            self.risk_results['frcar'],
            factor_types,
            self.calc_date
        )
        print(f"   因子风险指标: {factor_file}")
        
        self.memory_monitor.print_memory_status("结果保存完成")
        
        return {
            'stock_risk': stock_file,
            'factor_risk': factor_file,
        }
    
    def _get_factor_types(self) -> pd.Series:
        """获取因子类型映射"""
        factor_types = {}
        
        category_map = {
            'size': '规模',
            'volatility': '波动率',
            'liquidity': '流动性',
            'momentum': '动量',
            'quality_leverage': '质量-杠杆',
            'quality_earn_vol': '质量-盈利波动',
            'quality_earn_quality': '质量-盈利质量',
            'quality_profit': '质量-盈利能力',
            'quality_invest': '质量-投资质量',
            'value': '价值',
            'growth': '成长',
        }
        
        from .config import CNE6_STYLE_FACTORS
        for category, factors in CNE6_STYLE_FACTORS.items():
            for factor in factors:
                factor_types[factor] = category_map.get(category, category)
        
        for code, name in INDUSTRY_MAPPING.items():
            factor_types[name] = '行业'
        
        return pd.Series(factor_types)
    
    def get_risk_summary(self) -> pd.DataFrame:
        """
        获取风险汇总表
        
        Returns:
            风险汇总DataFrame
        """
        if self.risk_results is None:
            raise ValueError("风险结果未计算")
        
        summary = pd.DataFrame({
            '指标': ['组合总风险', '主动风险(跟踪误差)', 'RCAR之和', 'FRCAR之和'],
            '数值': [
                self.risk_results['total_risk'],
                self.risk_results['active_risk'],
                self.risk_results['rcar_sum'],
                self.risk_results['frcar_sum'],
            ]
        })
        
        return summary
    
    def print_risk_report(self):
        """打印风险报告"""
        if self.risk_results is None:
            print("风险结果未计算")
            return
        
        print("\n" + "=" * 70)
        print("Barra CNE6 风险分析报告")
        print(f"计算日期: {self.calc_date}")
        print("=" * 70)
        
        # 风险汇总
        summary = self.get_risk_summary()
        print("\n风险汇总:")
        print(summary.to_string(index=False))
        
        # 股票风险贡献前10
        print("\n\n股票风险贡献(RCAR)前10:")
        rcar_top10 = self.risk_results['rcar'].abs().nlargest(10)
        for stock, rcar in rcar_top10.items():
            actual_rcar = self.risk_results['rcar'].loc[stock]
            print(f"  {stock}: {actual_rcar:.6f}")
        
        # 因子风险贡献前10
        print("\n\n因子风险贡献(FRCAR)前10:")
        frcar_top10 = self.risk_results['frcar'].abs().nlargest(10)
        for factor, frcar in frcar_top10.items():
            actual_frcar = self.risk_results['frcar'].loc[factor]
            print(f"  {factor}: {actual_frcar:.6f}")
        
        print("\n" + "=" * 70)
    
    def clear_memory(self):
        """清理所有中间结果以释放内存"""
        print("\n清理引擎内存...")
        
        self.factor_exposure = None
        self.factor_returns = None
        self.factor_covariance = None
        self.specific_risk = None
        self.asset_covariance = None
        self.benchmark_weights = None
        self.risk_results = None
        
        self.cross_sectional.clear_memory()
        optimize_memory()
        
        self.memory_monitor.print_memory_status("内存清理完成")
    
    def get_memory_report(self) -> dict:
        """
        获取内存使用报告
        
        Returns:
            内存使用信息字典
        """
        return self.memory_monitor.check_memory("当前状态")
