"""
Barra CNE6 风险模型主引擎
整合所有模块，提供统一计算接口
"""
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

from .config import MODEL_PARAMS, BENCHMARK_CONFIG
from .covariance import FactorCovarianceEstimator
from .cross_sectional import CrossSectionalRegression
from .data_loader import DataLoader
from .factor_exposure import FactorExposureBuilder
from .output import RiskOutputManager
from .specific_risk import SpecificRiskEstimator

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
from utils import (
    get_trade_cal_inter, LoggerFactory, calculate_excess_returns, winsorize,
)

logger = LoggerFactory.get_logger(__name__)


class BarraRiskEngine:
    """Barra CNE6 风险模型引擎"""

    def __init__(self, calc_date: str,
                 market: str = 'csi300',
                 output_dir: str = 'output',
                 n_jobs: int = 1):
        """
        初始化风险模型引擎

        Args:
            calc_date: 计算日期，格式'YYYY-MM-DD'
            market: 市场代码，默认'csi300'
            output_dir: 输出目录
            n_jobs: 并行进程数
        """
        self.calc_date = calc_date
        self.market = market
        self.n_jobs = n_jobs

        # 初始化各模块
        self.data_loader = DataLoader(market=market)
        self.factor_builder = FactorExposureBuilder()
        self.cross_sectional = CrossSectionalRegression()
        self.covariance_estimator = FactorCovarianceEstimator()
        self.specific_risk_estimator = SpecificRiskEstimator(
            panel_window=MODEL_PARAMS['panel_regression_window'],
        )
        self.output_manager = RiskOutputManager(output_dir=output_dir)

        # 中间结果
        self.factor_exposure = None
        self.factor_returns = None
        self.factor_covariance = None

    def run(self, start_date: str, end_date: str,
            use_cache: bool = False) -> None:
        """
        运行模型估计：因子暴露→横截面回归(b)→协方差(F)→特异风险(Delta)

        Args:
            start_date: 历史数据开始日期
            end_date: 历史数据结束日期
            use_cache: 是否使用缓存数据
        """
        logger.info('=' * 70)
        logger.info('开始风险模型更新...')
        logger.info(f'历史数据区间: {start_date} 至 {end_date}')

        # 1. 加载数据
        logger.info('1. 加载全量历史数据...')
        instruments = self.data_loader.get_instruments(start_date, end_date)
        logger.info(f'   股票数量: {len(instruments)}')

        if use_cache:
            logger.info('使用缓存数据...')
            raw_data = self.output_manager.load_data(
                'debug/raw_data.parquet', type='parquet')
            returns_df = self.output_manager.load_data(
                'debug/returns_data.parquet', type='parquet')
            industry_df = self.output_manager.load_data(
                'debug/industry_data.parquet', type='parquet')
            market_cap_df = self.output_manager.load_data(
                'debug/market_cap_data.parquet', type='parquet')
        else:
            raw_data = self.data_loader.load_fields_data(
                instruments, start_date, end_date,
                extend_start=MODEL_PARAMS['data_extend_years'],
                extend_freq='Y')
            returns_df = self.data_loader.load_returns(
                instruments, start_date, end_date)
            benchmark_df = self.data_loader.load_returns(
                BENCHMARK_CONFIG['BENCHMARK'], start_date, end_date)
            returns_df = calculate_excess_returns(returns_df, benchmark_df)
            logger.info(f'   超额收益计算完成，shape: {returns_df.shape}')

            industry_df = self.data_loader.load_industry(
                instruments, start_date, end_date)
            market_cap_df = self.data_loader.load_market_cap(
                instruments, start_date, end_date)

            self.output_manager.save_data(
                raw_data, 'debug/raw_data.parquet', type='parquet')
            self.output_manager.save_data(
                returns_df, 'debug/returns_data.parquet', type='parquet')
            self.output_manager.save_data(
                industry_df, 'debug/industry_data.parquet', type='parquet')
            self.output_manager.save_data(
                market_cap_df, 'debug/market_cap_data.parquet', type='parquet')

        logger.info(f'raw_data shape: {raw_data.shape}')
        logger.info(f'returns_df shape: {returns_df.shape}')
        logger.info(f'industry_df shape: {industry_df.shape}')
        logger.info(f'market_cap_df shape: {market_cap_df.shape}')

        # NOTE: raw_data 保留扩展区间，供 BETA(504d) 等因子滚动窗口使用
        # 因子计算完成后再由 build_exposure_matrix 内部截断到与 industry_df 一致( [start_date, end_date]）

        com_dates = get_trade_cal_inter(
            start_date, end_date)
        com_dates = pd.to_datetime(com_dates)

        # 构建因子暴露矩阵
        logger.info('2. 构建因子暴露矩阵...')
        if use_cache:
            logger.info('加载缓存的因子暴露矩阵...')
            self.factor_exposure = self.output_manager.load_data(
                'debug/exposure_matrix.parquet', type='parquet')
        else:
            self.factor_exposure = self.factor_builder.build_exposure_matrix(
                raw_data, industry_df, market_cap_df,
                n_jobs=self.n_jobs,
                output_manager=self.output_manager,
                com_dates=com_dates,
            )
        nan_ratio = (self.factor_exposure.isna().sum(axis=0) / self.factor_exposure.shape[0]).sort_values(ascending=False)
        logger.info('因子缺失值比例：{}'.format(nan_ratio))
        if nan_ratio.max() > 0.5:
            err_msg = '因子缺失值比例过高，请检查数据'
            logger.error(err_msg)
            raise ValueError(err_msg)

        self.factor_exposure = self.factor_exposure[
            self.factor_exposure.index.get_level_values('datetime').isin(
                com_dates)
        ]
        returns_df = returns_df[
            returns_df.index.get_level_values('datetime').isin(
                com_dates)
        ]
        market_cap_df = market_cap_df[
            market_cap_df.index.get_level_values('datetime').isin(com_dates)
        ]

        if self.factor_exposure.empty or returns_df.empty or market_cap_df.empty:
            logger.warning('筛选后的数据为空')
            logger.warning(f'  因子暴露为空: {self.factor_exposure.empty}')
            logger.warning(f'  收益率为空: {returns_df.empty}')
            logger.warning(f'  市值为空: {market_cap_df.empty}')

        logger.info(f'   对齐后 - 日期数: {len(com_dates)}, '
                     f'因子暴露: {self.factor_exposure.shape}, '
                     f'收益率: {returns_df.shape}, '
                     f'市值: {market_cap_df.shape}')

        # 横截面回归估计因子收益率(b)
        # Fix: factor_exposure 暂时以 0 填充 NaN
        logger.info('3. 横截面回归...')
        self.factor_returns = self.cross_sectional.fit_multi_periods(
            returns_df, self.factor_exposure.fillna(0.0), market_cap_df, method='constrained')
        self.output_manager.save_data(
            self.factor_returns, 'model/factor_returns.parquet', type='parquet')

        # 估计因子收益率协方差矩阵(F)
        logger.info('4. 估计因子收益率协方差矩阵（F）...')
        # 对因子收益率（b）去极值（时间维度）
        factor_returns_winsorized = winsorize(
            self.factor_returns, method='median')
        self.factor_covariance = \
            self.covariance_estimator.estimate_barra_covariance(
                factor_returns_winsorized,
                half_life_corr=MODEL_PARAMS['half_life_corr'],
                half_life_var=MODEL_PARAMS['half_life_var'],
                init_periods=MODEL_PARAMS['ewma_init_periods'],
            )
        self.output_manager.save_data(
            self.factor_covariance,
            'model/factor_covariance.parquet', type='parquet')

        # 估计特异风险矩阵
        logger.info('5. 估计特异风险收益率协方差矩阵(Delta)...')
        residuals_df = self.cross_sectional.get_residuals()
        self.output_manager.save_data(
            residuals_df, 'model/residuals.parquet', type='parquet')

        logger.info(f'residuals_df: {residuals_df.shape}')
        logger.info(f'factor_exposure: {self.factor_exposure.shape}')

        # Fix: 暂时将 factor_exposure 的NaN用 0 填充
        specific_risk_df = self.specific_risk_estimator.estimate_specific_risk(
            residuals_df, self.factor_exposure.fillna(0.0))
        self.output_manager.save_data(
            specific_risk_df, 'model/specific_risk.parquet', type='parquet')

        logger.info('风险模型更新完成')
        logger.info('=' * 70)
