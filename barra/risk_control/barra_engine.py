"""
Barra CNE6 风险模型主引擎
整合所有模块，提供统一接口
"""
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Union

# 导入各模块
from .data_loader import DataLoader
from .portfolio import PortfolioManager
from .factor_exposure import FactorExposureBuilder
from .cross_sectional import CrossSectionalRegression
from .covariance import FactorCovarianceEstimator
from .specific_risk import SpecificRiskEstimator
from .risk_model import AssetCovarianceCalculator
from .risk_attribution import RiskAttributionAnalyzer
from .output import RiskOutputManager
from .config import MODEL_PARAMS, STYLE_FACTOR_LIST, INDUSTRY_MAPPING


class BarraRiskEngine:
    """Barra CNE6 风险模型引擎"""
    
    def __init__(self, calc_date: str, 
                 portfolio_input: Union[str, Dict, pd.Series] = 'random',
                 market: str = 'csi300',
                 output_dir: str = 'output',
                 cache_dir: Optional[str] = None,
                 n_jobs: int = 1):
        """
        初始化风险模型引擎
        
        Args:
            calc_date: 计算日期，格式'YYYY-MM-DD'
            portfolio_input: 投资组合输入：
                - 'random': 随机生成50只股票组合
                - dict: 持仓权重字典 {instrument: weight}
                - pd.Series: 持仓权重Series
                - str: CSV文件路径
            market: 市场代码，默认'csi300'
            output_dir: 输出目录
            cache_dir: 缓存目录
            n_jobs: 并行进程数
        """
        self.calc_date = calc_date
        self.market = market
        self.n_jobs = n_jobs
        
        # 初始化各模块
        self.data_loader = DataLoader(market=market)
        self.portfolio_manager = PortfolioManager(market=market)
        self.factor_builder = FactorExposureBuilder(cache_dir=cache_dir)
        self.cross_sectional = CrossSectionalRegression()
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
    
    def _load_portfolio(self, portfolio_input) -> pd.Series:
        """加载投资组合"""
        if isinstance(portfolio_input, str) and portfolio_input.lower() == 'random':
            print(f"生成随机组合（50只股票）...")
            return self.portfolio_manager.generate_random_portfolio(
                self.calc_date, n_stocks=50, random_state=42
            )
        else:
            return self.portfolio_manager.load_portfolio(portfolio_input, self.calc_date)
    
    def run_monthly_update(self, start_date: str, end_date: str) -> None:
        """
        月频更新：估计因子收益率、协方差矩阵、特异风险
        
        这是月度任务，需要历史数据来计算模型参数
        
        Args:
            start_date: 历史数据开始日期
            end_date: 历史数据结束日期
        """
        print("\n" + "=" * 70)
        print("开始月频模型更新...")
        print(f"历史数据区间: {start_date} 至 {end_date}")
        
        # 1. 加载数据
        print("\n1. 加载数据...")
        instruments = self.data_loader.get_instruments(start_date, end_date)
        print(f"   股票数量: {len(instruments)}")
        
        raw_data = self.data_loader.load_factor_data(instruments, start_date, end_date)
        returns_df = self.data_loader.load_returns(instruments, start_date, end_date)
        industry_df = self.data_loader.load_industry(instruments, start_date, end_date)
        market_cap_df = self.data_loader.load_market_cap(instruments, start_date, end_date)
        
        # 保存中间结果目录
        import os
        debug_dir = Path('barra/risk_control/debug_output')
        debug_dir.mkdir(parents=True, exist_ok=True)
        
        # 2. 构建因子暴露矩阵
        print("\n2. 构建因子暴露矩阵...")
        self.factor_exposure = self.factor_builder.build_exposure_matrix(
            raw_data, industry_df, market_cap_df, n_jobs=self.n_jobs
        )
        # 保存因子暴露矩阵
        self.factor_exposure.to_csv(debug_dir / 'factor_exposure.csv')
        print(f"   因子暴露矩阵已保存: {debug_dir}/factor_exposure.csv")
        
        # 3. 横截面回归估计因子收益率
        print("\n3. 横截面回归...")
        self.factor_returns = self.cross_sectional.fit_multi_periods(
            returns_df, self.factor_exposure, market_cap_df
        )
        # 保存因子收益率
        self.factor_returns.to_csv(debug_dir / 'factor_returns.csv')
        print(f"   因子收益率已保存: {debug_dir}/factor_returns.csv")
        
        # 4. 估计因子协方差矩阵
        print("\n4. 估计因子协方差矩阵...")
        self.factor_covariance = self.covariance_estimator.estimate_sample_covariance(
            self.factor_returns
        )
        # 保存协方差矩阵
        self.factor_covariance.to_csv(debug_dir / 'factor_covariance.csv')
        print(f"   因子协方差矩阵已保存: {debug_dir}/factor_covariance.csv")
        
        # 5. 估计特异风险矩阵
        print("\n5. 估计特异风险矩阵...")
        residuals_df = self.cross_sectional.get_residuals()
        # 保存残差
        residuals_df.to_csv(debug_dir / 'residuals.csv')
        print(f"   残差已保存: {debug_dir}/residuals.csv")
        
        specific_risk_df = self.specific_risk_estimator.estimate_specific_risk(
            residuals_df, self.factor_exposure
        )
        
        # 将特异风险转换为Series
        self.specific_risk = pd.Series(
            specific_risk_df['specific_var'].values,
            index=specific_risk_df['instrument']
        )
        # 保存特异风险
        self.specific_risk.to_csv(debug_dir / 'specific_risk.csv')
        print(f"   特异风险已保存: {debug_dir}/specific_risk.csv")
        
        print("\n月频模型更新完成")
        print("=" * 70)
    
    def run_daily_risk(self, exposure: Optional[pd.DataFrame] = None,
                      factor_cov: Optional[pd.DataFrame] = None,
                      specific_risk: Optional[pd.Series] = None) -> dict:
        """
        日频风险计算：计算MCAR/RCAR/FMCAR/FRCAR
        
        这是每日任务，使用已有的模型参数
        
        Args:
            exposure: 当日因子暴露矩阵（可选，使用已有数据）
            factor_cov: 因子协方差矩阵（可选，使用已有数据）
            specific_risk: 特异风险（可选，使用已有数据）
            
        Returns:
            风险分析结果字典
        """
        print("\n" + "=" * 70)
        print(f"开始日频风险计算: {self.calc_date}")
        
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
        
        # 1. 获取基准权重
        print("\n1. 获取基准权重...")
        self.benchmark_weights = self.portfolio_manager.get_benchmark_weights(self.calc_date)
        print(f"   基准股票数: {len(self.benchmark_weights)}")
        
        # 2. 计算资产协方差矩阵
        print("\n2. 计算资产协方差矩阵...")
        # 提取当日的因子暴露
        if isinstance(self.factor_exposure.index, pd.MultiIndex):
            today_exposure = self.factor_exposure.xs(self.calc_date, level=1)
        else:
            today_exposure = self.factor_exposure
        
        self.asset_covariance = self.asset_cov_calculator.calculate(
            today_exposure,
            self.factor_covariance,
            self.specific_risk
        )
        
        # 3. 风险归因分析
        print("\n3. 风险归因分析...")
        self.risk_results = self.risk_attribution.analyze_risk(
            self.asset_covariance,
            self.factor_covariance,
            today_exposure,
            self.portfolio_weights,
            self.benchmark_weights
        )
        
        print("\n日频风险计算完成")
        print("=" * 70)
        
        return self.risk_results
    
    def save_results(self) -> Dict[str, str]:
        """
        保存风险指标到CSV文件
        
        Returns:
            保存的文件路径字典
        """
        if self.risk_results is None:
            raise ValueError("风险结果未计算，请先运行日频风险计算")
        
        print("\n保存风险指标到文件...")
        
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
        
        return {
            'stock_risk': stock_file,
            'factor_risk': factor_file,
        }
    
    def _get_factor_types(self) -> pd.Series:
        """获取因子类型映射"""
        factor_types = {}
        
        # 风格因子
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
        
        # 行业因子
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
