"""
Barra CNE6 风险模型主引擎
整合所有模块，提供统一接口
"""
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Union, List

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

# 导入工具函数
import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
from utils import get_trade_cal_inter


def get_monthly_first_trade_days(start_date: str, end_date: str) -> List[str]:
    """
    获取时间区间内每月的第一个交易日
    
    Args:
        start_date: 开始日期，格式'YYYY-MM-DD'
        end_date: 结束日期，格式'YYYY-MM-DD'
        
    Returns:
        每月第一个交易日列表（已排序），格式['YYYY-MM-DD', ...]
    """
    # 获取所有交易日
    trade_dates = get_trade_cal_inter(start_date, end_date)
    
    # 按年月分组，取每月第一个交易日
    dates_df = pd.DataFrame({'date': pd.to_datetime(trade_dates)})
    dates_df['year_month'] = dates_df['date'].dt.to_period('M')
    monthly_first = dates_df.groupby('year_month')['date'].min()
    
    return [d.strftime('%Y-%m-%d') for d in monthly_first.tolist()]


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
        
        这是月度任务，需要历史数据来计算模型参数。
        使用全量历史数据以保证因子计算有足够的时间窗口。
        
        Args:
            start_date: 历史数据开始日期
            end_date: 历史数据结束日期
        """
        print("\n" + "=" * 70)
        print("开始月频模型更新...")
        print(f"历史数据区间: {start_date} 至 {end_date}")
        
        # 1. 加载数据（一次性获取全量历史数据，保证因子计算有足够的时间窗口）
        print("\n1. 加载全量历史数据...")
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
        
        # 3. 筛选月初样本并对齐数据
        print("\n3. 筛选月初样本并对齐数据...")
        regression_dates = get_monthly_first_trade_days(start_date, end_date)
        regression_dates_ts = pd.to_datetime(regression_dates)
        
        # 筛选三个数据集的月初样本
        self.factor_exposure = self.factor_exposure[
            self.factor_exposure.index.get_level_values(1).isin(regression_dates_ts)
        ]
        returns_df = returns_df[
            returns_df.index.get_level_values(1).isin(regression_dates_ts)
        ]
        market_cap_df = market_cap_df[
            market_cap_df.index.get_level_values(1).isin(regression_dates_ts)
        ]
        
        # 检查是否为空
        if self.factor_exposure.empty or returns_df.empty or market_cap_df.empty:
            print(f"   ⚠️  警告：筛选后的数据为空")
            print(f"      因子暴露为空: {self.factor_exposure.empty}")
            print(f"      收益率为空: {returns_df.empty}")
            print(f"      市值为空: {market_cap_df.empty}")
        
        print(f"   对齐后的数据:")
        print(f"      日期数量: {len(regression_dates)}")
        print(f"      因子暴露: {self.factor_exposure.shape}")
        print(f"      收益率: {returns_df.shape}")
        print(f"      市值: {market_cap_df.shape}")
        
        # 4. 横截面回归估计因子收益率
        print("\n4. 横截面回归...")
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
        
        # 提取与残差日期对齐的因子暴露（重要！因为横截面回归可能跳过某些日期）
        residual_dates = residuals_df.index.get_level_values(1).unique()
        aligned_factor_exposure = self.factor_exposure[
            self.factor_exposure.index.get_level_values(1).isin(residual_dates)
        ]
        print(f"   残差日期数量: {len(residual_dates)}")
        print(f"   对齐后的因子暴露: {aligned_factor_exposure.shape}")
        
        # 特异风险协方差矩阵 Delta
        specific_risk_df = self.specific_risk_estimator.estimate_specific_risk(
            residuals_df, aligned_factor_exposure
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
    
    def save_model_data(self, model_dir: str) -> Dict[str, str]:
        """
        保存模型数据到Parquet文件，供日频计算使用
        
        Args:
            model_dir: 模型数据保存目录
            
        Returns:
            保存的文件路径字典
        """
        from datetime import datetime
        
        model_path = Path(model_dir)
        model_path.mkdir(parents=True, exist_ok=True)
        
        # 使用calc_date作为文件后缀
        date_suffix = self.calc_date.replace('-', '')
        
        saved_files = {}
        
        # 1. 保存因子暴露矩阵（最新一期）
        if self.factor_exposure is not None:
            # 提取最新日期的因子暴露
            if isinstance(self.factor_exposure.index, pd.MultiIndex):
                latest_date = self.factor_exposure.index.get_level_values(1).max()
                latest_exposure = self.factor_exposure.xs(latest_date, level=1)
            else:
                latest_exposure = self.factor_exposure
            
            exposure_file = model_path / f'factor_exposure_{date_suffix}.parquet'
            latest_exposure.to_parquet(exposure_file)
            saved_files['factor_exposure'] = str(exposure_file)
            print(f"   因子暴露矩阵已保存: {exposure_file}")
        
        # 2. 保存因子收益率历史
        if self.factor_returns is not None and not self.factor_returns.empty:
            returns_file = model_path / f'factor_returns_{date_suffix}.parquet'
            self.factor_returns.to_parquet(returns_file)
            saved_files['factor_returns'] = str(returns_file)
            print(f"   因子收益率已保存: {returns_file}")
        
        # 3. 保存因子协方差矩阵
        if self.factor_covariance is not None:
            cov_file = model_path / f'factor_covariance_{date_suffix}.parquet'
            self.factor_covariance.to_parquet(cov_file)
            saved_files['factor_covariance'] = str(cov_file)
            print(f"   因子协方差矩阵已保存: {cov_file}")
        
        # 4. 保存特异风险
        if self.specific_risk is not None:
            specific_file = model_path / f'specific_risk_{date_suffix}.parquet'
            self.specific_risk.to_frame('specific_var').to_parquet(specific_file)
            saved_files['specific_risk'] = str(specific_file)
            print(f"   特异风险已保存: {specific_file}")
        
        # 5. 保存模型元数据
        metadata = {
            'calc_date': self.calc_date,
            'market': self.market,
            'n_factors': len(self.factor_covariance) if self.factor_covariance is not None else 0,
            'n_stocks': len(self.specific_risk) if self.specific_risk is not None else 0,
            'saved_files': saved_files,
            'timestamp': datetime.now().isoformat()
        }
        
        import json
        metadata_file = model_path / f'metadata_{date_suffix}.json'
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        saved_files['metadata'] = str(metadata_file)
        print(f"   模型元数据已保存: {metadata_file}")
        
        return saved_files
    
    def load_model_data(self, model_dir: str, calc_date: str = None) -> bool:
        """
        从Parquet文件加载模型数据，供日频计算使用
        
        Args:
            model_dir: 模型数据目录路径
            calc_date: 计算日期，如果为None则使用self.calc_date
            
        Returns:
            是否成功加载
        """
        from datetime import datetime
        
        if calc_date is None:
            calc_date = self.calc_date
        
        model_path = Path(model_dir)
        if not model_path.exists():
            print(f"警告：模型数据目录不存在: {model_dir}")
            return False
        
        # 尝试找到匹配calc_date的模型文件
        date_suffix = calc_date.replace('-', '')
        
        # 如果没有精确匹配，找最新的模型文件
        exposure_files = list(model_path.glob('factor_exposure_*.parquet'))
        if not exposure_files:
            print(f"警告：未找到模型数据文件")
            return False
        
        # 如果精确日期存在，使用精确日期；否则使用最新日期
        exposure_file = model_path / f'factor_exposure_{date_suffix}.parquet'
        if not exposure_file.exists():
            # 使用最新的模型文件
            exposure_file = sorted(exposure_files)[-1]
            # 从文件名提取日期
            date_str = exposure_file.stem.split('_')[-1]
            loaded_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
            print(f"   未找到 {calc_date} 的模型数据，使用最新模型: {loaded_date}")
        else:
            loaded_date = calc_date
        
        date_suffix = loaded_date.replace('-', '')
        
        try:
            # 1. 加载因子暴露矩阵
            exposure_file = model_path / f'factor_exposure_{date_suffix}.parquet'
            if exposure_file.exists():
                self.factor_exposure = pd.read_parquet(exposure_file)
                print(f"   因子暴露矩阵已加载: {exposure_file.name} ({self.factor_exposure.shape})")
            else:
                print(f"   警告：因子暴露矩阵文件不存在")
                return False
            
            # 2. 加载因子收益率历史（可选，用于分析）
            returns_file = model_path / f'factor_returns_{date_suffix}.parquet'
            if returns_file.exists():
                self.factor_returns = pd.read_parquet(returns_file)
                print(f"   因子收益率历史已加载: {returns_file.name} ({self.factor_returns.shape})")
            
            # 3. 加载因子协方差矩阵（关键）
            cov_file = model_path / f'factor_covariance_{date_suffix}.parquet'
            if cov_file.exists():
                self.factor_covariance = pd.read_parquet(cov_file)
                print(f"   因子协方差矩阵已加载: {cov_file.name} ({self.factor_covariance.shape})")
            else:
                print(f"   警告：协方差矩阵文件不存在")
                return False
            
            # 4. 加载特异风险（关键）
            specific_file = model_path / f'specific_risk_{date_suffix}.parquet'
            if specific_file.exists():
                specific_df = pd.read_parquet(specific_file)
                self.specific_risk = specific_df['specific_var']
                print(f"   特异风险已加载: {specific_file.name} ({len(self.specific_risk)})")
            else:
                print(f"   警告：特异风险文件不存在")
                return False
            
            # 5. 加载并显示元数据
            metadata_file = model_path / f'metadata_{date_suffix}.json'
            if metadata_file.exists():
                import json
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                print(f"   模型元数据: calc_date={metadata['calc_date']}, n_factors={metadata['n_factors']}")
            
            print(f"✓ 模型数据加载成功")
            return True
            
        except Exception as e:
            print(f"✗ 模型数据加载失败: {str(e)}")
            return False
    
    def calculate_daily_exposure(self, calc_date: str, n_jobs: int = 1) -> pd.DataFrame:
        """
        计算指定日期的因子暴露矩阵（用于日频风险计算）
        
        日频计算需要：
        1. 使用最新的模型参数（协方差矩阵F、特异风险Δ）- 从月频模型加载
        2. 重新计算指定日期的因子暴露 X_t - 本方法实现
        3. 组合使用：V = X_t * F * X_t^T + Δ
        
        Args:
            calc_date: 计算日期，格式'YYYY-MM-DD'
            n_jobs: 并行进程数
            
        Returns:
            因子暴露矩阵，index=instrument, columns=factors
        """
        print(f"\n计算 {calc_date} 的因子暴露矩阵...")
        
        # 1. 加载当日数据
        print("   加载当日市场数据...")
        instruments = self.data_loader.get_instruments(calc_date, calc_date)
        print(f"   股票数量: {len(instruments)}")
        
        raw_data = self.data_loader.load_factor_data(instruments, calc_date, calc_date)
        industry_df = self.data_loader.load_industry(instruments, calc_date, calc_date)
        market_cap_df = self.data_loader.load_market_cap(instruments, calc_date, calc_date)
        
        # 2. 构建因子暴露矩阵（使用缓存目录保存中间结果）
        exposure = self.factor_builder.build_exposure_matrix(
            raw_data, industry_df, market_cap_df, 
            save_path=f'barra/risk_control/cache/exposure_{calc_date}.csv',
            n_jobs=n_jobs
        )
        
        # 3. 提取指定日期的暴露
        if isinstance(exposure.index, pd.MultiIndex):
            daily_exposure = exposure.xs(calc_date, level=1)
        else:
            daily_exposure = exposure
        
        print(f"   因子暴露矩阵: {daily_exposure.shape}")
        return daily_exposure
    
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
        rcar = self.risk_results['rcar']
        if len(rcar) > 0 and rcar.dtype != object:
            rcar_top10 = rcar.abs().nlargest(min(10, len(rcar)))
            for stock, rcar_val in rcar_top10.items():
                actual_rcar = rcar.loc[stock]
                print(f"  {stock}: {actual_rcar:.6f}")
        else:
            print("  无数据或数据类型不支持")
        
        # 因子风险贡献前10
        print("\n\n因子风险贡献(FRCAR)前10:")
        frcar = self.risk_results['frcar']
        if len(frcar) > 0:
            # 确保数据类型为数值型
            frcar_numeric = pd.to_numeric(frcar, errors='coerce')
            if not frcar_numeric.isna().all():
                frcar_top10 = frcar_numeric.abs().nlargest(min(10, len(frcar_numeric)))
                for factor, frcar_val in frcar_top10.items():
                    actual_frcar = frcar.loc[factor]
                    print(f"  {factor}: {float(actual_frcar):.6f}")
            else:
                print("  所有FRCAR值为空")
        else:
            print("  无数据")
        
        print("\n" + "=" * 70)
