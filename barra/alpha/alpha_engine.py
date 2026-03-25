"""
Alpha引擎 - 编排完整的多信号Alpha预测流水线
"""
import pandas as pd
import numpy as np
from pathlib import Path

from .config import (
    ROLLING_WINDOW, RESIDUAL_VOL_WINDOW, OUTPUT_DIR, IC_LAG,
)
from .data_loader import AlphaDataLoader
from .signal_processor import SignalProcessor
from .residual_vol import ResidualVolEstimator
from .scenario_classifier import ScenarioClassifier
from .ic_estimator import ICEstimator
from .orthogonalizer import AlphaOrthogonalizer
from .output import AlphaOutputManager
from utils import LoggerFactory, dt

logger = LoggerFactory.get_logger(__name__)


class AlphaEngine:
    """多信号Alpha预测引擎

    编排完整的Alpha预测流水线：
    数据加载 -> 标准化 -> 残差波动率 -> 情形判断 -> IC估计
    -> 单信号Alpha -> (正交化合成) -> 输出
    """

    def __init__(self, market: str = 'csi300', output_dir: str = OUTPUT_DIR):
        """初始化

        Args:
            market: 市场代码
            output_dir: 输出目录
        """
        self.market = market
        self.data_loader = AlphaDataLoader(market=market)
        self.signal_processor = SignalProcessor()
        self.residual_vol_estimator = ResidualVolEstimator()
        self.scenario_classifier = ScenarioClassifier()
        self.ic_estimator = ICEstimator()
        self.orthogonalizer = AlphaOrthogonalizer()
        self.output_manager = AlphaOutputManager(output_dir=output_dir)

    def run(self, calc_date: str, portfolio: str = 'default') -> pd.DataFrame:
        """执行完整Alpha预测流水线

        Args:
            calc_date: 计算日期，如 '2026-03-06'
            portfolio: 持仓组合名称

        Returns:
            DataFrame(index=instrument, column='alpha')
        """
        logger.info('=' * 60)
        logger.info(f'Alpha预测开始: calc_date={calc_date}')

        # Step 1: 计算数据窗口
        start_date = dt.subtract_months(calc_date, 36)
        logger.info(f'数据窗口: [{start_date}, {calc_date}]')

        # Step 2: 加载数据
        logger.info('Step 1: 加载数据...')
        signal_df = self.data_loader.load_signal(start_date, calc_date)
        residuals = self.data_loader.load_residuals()

        # 过滤残差到窗口范围内
        resid_dates = residuals.index.get_level_values('datetime')
        residuals = residuals.loc[
            (resid_dates >= pd.Timestamp(start_date)) &
            (resid_dates <= pd.Timestamp(calc_date))
        ]
        n_resid_days = residuals.index.get_level_values('datetime').nunique()
        logger.info(f'残差数据: {n_resid_days}天, {residuals.shape[0]}条')
        if n_resid_days < RESIDUAL_VOL_WINDOW:
            raise ValueError(
                f'残差数据不足: 仅{n_resid_days}天，需要至少{RESIDUAL_VOL_WINDOW}天。'
                f'请先运行 barra/risk_control 积累足够的残差数据'
            )

        # 加载行业和市值
        instruments = signal_df.index.get_level_values('instrument').unique().tolist()
        ind_mv_df = self.data_loader.load_industry_and_market_cap(
            instruments, start_date, calc_date
        )
        industry_df = ind_mv_df[['industry_code']]
        market_cap_df = ind_mv_df[['circ_mv']]

        # Step 3: 横截面z-score标准化
        logger.info('Step 2: 横截面z-score标准化...')
        z_cs = self.signal_processor.cross_sectional_zscore(signal_df)

        # Step 4: 残差波动率估计
        logger.info('Step 3: 残差波动率估计...')
        omega = self.residual_vol_estimator.estimate_all(
            residuals, industry_df, market_cap_df, calc_date
        )

        # Step 5: 情形判断
        logger.info('Step 4: 情形判断...')
        case = self.scenario_classifier.classify(signal_df, omega, calc_date)

        # Step 6: IC估计
        logger.info('Step 5: IC估计...')
        ic = self.ic_estimator.compute_ic(z_cs, residuals, calc_date)

        # Step 7: 单信号Alpha
        logger.info('Step 6: 单信号Alpha计算...')
        single_alpha = self.signal_processor.compute_single_alpha(
            z_cs, omega, ic, case, calc_date
        )

        # 当前K=1，直接使用单信号Alpha作为最终结果
        final_alpha = single_alpha

        # Step 8: 保存结果
        logger.info('Step 7: 保存结果...')
        self.output_manager.save_alpha(final_alpha, calc_date)
        self.output_manager.save_to_mysql(final_alpha, calc_date, portfolio)

        # 保存诊断信息
        diagnostics = self._build_diagnostics(
            final_alpha, omega, ic, case, calc_date
        )
        self.output_manager.save_diagnostics(diagnostics, calc_date)

        # 摘要
        logger.info('=' * 60)
        logger.info(f'计算日期: {calc_date}')
        logger.info(f'情形: Case {case}')
        logger.info(f'IC: {ic:.6f}')
        logger.info(f'Alpha统计: mean={final_alpha["alpha"].mean():.6f}, '
                    f'std={final_alpha["alpha"].std():.6f}, '
                    f'min={final_alpha["alpha"].min():.6f}, '
                    f'max={final_alpha["alpha"].max():.6f}')
        logger.info(f'股票数: {len(final_alpha)}')
        logger.info('=' * 60)

        return final_alpha

    def run_with_data(
        self,
        signals: dict[str, pd.DataFrame],
        residuals: pd.DataFrame,
        industry_df: pd.DataFrame,
        market_cap_df: pd.DataFrame,
        calc_date: str,
        portfolio: str = 'default'
    ) -> pd.DataFrame:
        """使用预加载数据执行Alpha预测（支持多信号）

        Args:
            signals: {信号名: DataFrame(MultiIndex(instrument,datetime), col='g')}
            residuals: 残差收益率，MultiIndex(instrument, datetime), col='residual'
            industry_df: 行业数据，MultiIndex(instrument, datetime), col='industry_code'
            market_cap_df: 市值数据，MultiIndex(instrument, datetime), col='circ_mv'
            calc_date: 计算日期
            portfolio: 持仓组合名称

        Returns:
            DataFrame(index=instrument, column='alpha')
        """
        K = len(signals)
        logger.info('=' * 60)
        logger.info(f'Alpha预测开始: calc_date={calc_date}, K={K}个信号')

        # Step 1: 残差波动率估计
        logger.info('Step 1: 残差波动率估计...')
        omega = self.residual_vol_estimator.estimate_all(
            residuals, industry_df, market_cap_df, calc_date
        )

        # Step 2: 逐信号处理
        single_alphas_today = {}   # {name: Series(instrument -> alpha)} 当日
        alpha_histories = {}       # {name: DataFrame(MultiIndex, col='alpha')} 历史

        for name, signal_df in signals.items():
            logger.info(f'--- 信号 [{name}] ---')

            # 横截面z-score
            z_cs = self.signal_processor.cross_sectional_zscore(signal_df)

            # 情形判断
            case = self.scenario_classifier.classify(signal_df, omega, calc_date)

            # IC估计
            ic = self.ic_estimator.compute_ic(z_cs, residuals, calc_date)

            # 当日单信号Alpha
            alpha_today = self.signal_processor.compute_single_alpha(
                z_cs, omega, ic, case, calc_date
            )
            single_alphas_today[name] = alpha_today['alpha']

            # 历史Alpha（正交化需要）
            if K > 1:
                alpha_hist = self.signal_processor.compute_alpha_history(
                    z_cs, omega, ic, case
                )
                alpha_histories[name] = alpha_hist

        # Step 3: 合成
        if K == 1:
            name = list(single_alphas_today.keys())[0]
            final_alpha = pd.DataFrame(
                {'alpha': single_alphas_today[name]}
            )
        else:
            logger.info('Step 3: 多信号正交化合成...')
            final_alpha_series = self.orthogonalizer.fit_and_transform(
                alpha_histories, single_alphas_today,
                residuals, calc_date
            )
            final_alpha = final_alpha_series.to_frame('alpha')

        final_alpha.index.name = 'instrument'

        # Step 4: 保存
        logger.info('Step 4: 保存结果...')
        self.output_manager.save_alpha(final_alpha, calc_date)
        self.output_manager.save_to_mysql(final_alpha, calc_date, portfolio)

        # 摘要
        logger.info('=' * 60)
        logger.info(f'计算日期: {calc_date}, 信号数: {K}')
        logger.info(f'Alpha统计: mean={final_alpha["alpha"].mean():.6f}, '
                    f'std={final_alpha["alpha"].std():.6f}, '
                    f'min={final_alpha["alpha"].min():.6f}, '
                    f'max={final_alpha["alpha"].max():.6f}')
        logger.info(f'股票数: {len(final_alpha)}')
        logger.info('=' * 60)

        return final_alpha

    def _build_diagnostics(
        self,
        alpha: pd.DataFrame,
        omega: pd.Series,
        ic: float,
        case: int,
        calc_date: str
    ) -> pd.DataFrame:
        """构建诊断信息

        Args:
            alpha: 最终Alpha
            omega: 残差波动率
            ic: 信息系数
            case: 情形
            calc_date: 计算日期

        Returns:
            诊断信息DataFrame
        """
        instruments = alpha.index
        diag = pd.DataFrame(index=instruments)
        diag['alpha'] = alpha['alpha']
        diag['omega'] = omega.reindex(instruments)
        diag['ic'] = ic
        diag['case'] = case
        diag['calc_date'] = calc_date
        return diag
