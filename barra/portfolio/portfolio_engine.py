"""
投资组合优化主引擎
"""
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from barra.portfolio.config import (
    OPTIMIZATION_PARAMS, DEFAULT_PORTFOLIO_VALUE
)
from barra.portfolio.data_loader import PortfolioDataLoader
from barra.portfolio.no_trade_zone import NoTradeZoneIterator, build_asset_covariance
from barra.portfolio.optimizer import QPOptimizer
from barra.portfolio.output import PortfolioOutputManager
from barra.portfolio.trade_generator import TradeGenerator
from config import BENCHMARK_CONFIG
from utils import LoggerFactory, PickleIO

logger = LoggerFactory.get_logger(__name__)


class PortfolioEngine:
    """投资组合优化主引擎

    编排完整的优化流程：
    1. 数据加载与对齐
    2. 构建资产协方差矩阵
    3. QP优化求解（可选）
    4. 无交易区域迭代
    5. 生成交易指令
    6. 保存结果

    使用示例：
        engine = PortfolioEngine(calc_date='2026-03-28')
        result = engine.run()
    """

    def __init__(
        self,
        calc_date: str,
        market: str = BENCHMARK_CONFIG['market'],
        risk_output_dir: str = None,
        output_dir: str = None,
        portfolio_name: str = 'default',
        position: str = 'zero',
        **optimization_params
    ):
        """初始化引擎

        Args:
            calc_date: 计算日期 'YYYY-MM-DD'
            market: 市场代码
            risk_output_dir: 风险模型输出目录
            output_dir: 组合优化输出目录
            portfolio_name: 组合名称
            position: 持仓信息来源
            **optimization_params: 优化参数（覆盖默认配置）
        """
        self.calc_date = calc_date
        self.market = market
        self.portfolio_name = portfolio_name
        self.position = position

        # 合并优化参数
        self.params = OPTIMIZATION_PARAMS.copy()
        self.params.update(optimization_params)

        # 初始化组件
        self.data_loader = PortfolioDataLoader(
            market=market,
            risk_output_dir=risk_output_dir,
            portfolio_name=portfolio_name
        )
        self.output_manager = PortfolioOutputManager()
        self.trade_generator = TradeGenerator()
        self.debug_output_dir = f'{output_dir}/debug'
        os.makedirs(self.debug_output_dir, exist_ok=True)

        # 结果缓存
        self.data = None
        self.V = None
        self.iteration_result = None

    @staticmethod
    def benchmark_neutralize_alpha(
        alpha: np.ndarray,
        benchmark_weights: np.ndarray,
        beta: np.ndarray,
    ) -> np.ndarray:
        """对Alpha进行基准中性化: α_n^neutral = α_n - β_n * Σ(b_k * α_k)

        Args:
            alpha: 原始Alpha向量 (N,)
            benchmark_weights: 基准权重向量 (N,)
            beta: 股票Beta（对基准的敏感性）向量 (N,)

        Returns:
            中性化后的Alpha向量 (N,)
        """
        alpha_benchmark = np.sum(benchmark_weights * alpha)
        alpha_neutral = alpha - beta * alpha_benchmark

        # 验证中性化后基准加权平均 ≈ 0
        weighted_sum = np.sum(benchmark_weights * alpha_neutral)
        logger.info(
            f'Alpha基准中性化: '
            f'alpha_B={alpha_benchmark:.6f}, '
            f'Σ(b_n * α_n^neutral)={weighted_sum:.4e}, '
            f'range [{alpha.min():.4f}, {alpha.max():.4f}] → '
            f'[{alpha_neutral.min():.4f}, {alpha_neutral.max():.4f}]'
        )

        return alpha_neutral

    @staticmethod
    def factor_neutralize_alpha(
        alpha: np.ndarray,
        exposure: pd.DataFrame,
        specific_variance: pd.Series,
        regularization: float = 1e-6,
    ) -> np.ndarray:
        """对Alpha进行行业及风险因子中性化（GLS投影）

        α_sp = α - X(X^T Δ⁻¹ X)⁻¹ X^T Δ⁻¹ α

        将Alpha投影到因子暴露矩阵X的列空间补空间上，
        剔除能被共同因子解释的部分，保留特异Alpha。

        Args:
            alpha: 原始Alpha向量 (N,)
            exposure: 因子暴露矩阵 (N×K DataFrame)
            specific_variance: 特异风险方差 (N, Series)
            regularization: 正则化参数，防止X^T Δ⁻¹ X奇异

        Returns:
            特异Alpha向量 (N,)
        """
        X = exposure.values
        Δ_inv = 1.0 / specific_variance.values  # (N,)

        # X^T Δ⁻¹ X = X^T @ diag(Δ_inv) @ X = (X.T * Δ_inv) @ X
        Xt_W = X.T * Δ_inv  # (K, N)
        Xt_W_X = Xt_W @ X   # (K, K)

        # 正则化
        Xt_W_X += regularization * np.eye(Xt_W_X.shape[0])

        # θ = solve(X^T Δ⁻¹ X, X^T Δ⁻¹ α), α_sp = α - X @ θ
        Xt_W_alpha = X.T @ (Δ_inv * alpha)
        theta = np.linalg.solve(Xt_W_X, Xt_W_alpha)
        alpha_sp = alpha - X @ theta

        # 检验指标
        from scipy.stats import spearmanr
        sp_corr, _ = spearmanr(alpha, alpha_sp)
        vol_alpha = np.std(alpha)
        vol_alpha_sp = np.std(alpha_sp)
        logger.info(
            f'Alpha因子中性化: '
            f'alpha_vol={vol_alpha:.4e} → alpha_sp_vol={vol_alpha_sp:.4e}, '
            f'vol_reduction={vol_alpha_sp / vol_alpha:.2%}, '
            f'spearman_rank_corr={sp_corr:.4f}'
        )

        return alpha_sp

    def run(
        self,
        portfolio_value: float = DEFAULT_PORTFOLIO_VALUE,
        use_qp_init: bool = False,
        save_to_mysql: bool = False,
    ):
        """执行完整优化流程

        Args:
            portfolio_value: 组合净值（元）
            use_qp_init: 是否用QP解作为迭代初始值
            save_to_mysql: 是否保存到MySQL

        """
        logger.info('=' * 60)
        logger.info(f'投资组合优化开始: calc_date={self.calc_date}')
        logger.info('=' * 60)

        # Step 1: 加载并对齐数据
        logger.info('Step 1: 加载数据...')
        self.data = self.data_loader.align_all_data(
            calc_date=self.calc_date, position=self.position
        )
        PickleIO.write(self.data, f'{self.debug_output_dir}/data.pkl')

        # Step 2: 构建资产协方差矩阵
        logger.info('Step 2: 构建协方差矩阵...')
        self.V = build_asset_covariance(
            exposure=self.data['exposure'],
            factor_cov=self.data['factor_cov'],
            specific_risk=self.data['specific_risk']
        )
        logger.info('V shape: {}'.format(self.V.shape))
        PickleIO.write(self.V, f'{self.debug_output_dir}/V.pkl')

        # 准备numpy数组
        cash = self.data['cash']
        hold_value = self.data['current_position'] * self.data['prices']   # 持仓金额
        if self.position == 'zero':
            hold_weight = pd.Series(np.zeros_like(hold_value), index=hold_value.index)
            total_value = cash
        else:
            hold_weight = hold_value / hold_value.sum()  # 持仓权重
            total_value = cash + hold_value.sum().item()
        logger.info(f'total_value: {total_value}')
        PickleIO.write(hold_weight, f'{self.debug_output_dir}/hold_weight.pkl')
        alpha = self.data['alpha'].values
        w_b = self.data['benchmark_weights'].values

        # Alpha基准中性化: α_n^neutral = α_n - β_n * Σ(b_k * α_k)
        if self.params.get('benchmark_neutralize_alpha', True):
            beta = self.data['beta'].values
            alpha = self.benchmark_neutralize_alpha(alpha, w_b, beta)

        # Alpha行业及风险因子中性化: α_sp = α - X(X^T Δ⁻¹ X)⁻¹ X^T Δ⁻¹ α
        if self.params.get('factor_neutralize_alpha', True):
            alpha = self.factor_neutralize_alpha(
                alpha,
                self.data['exposure'],
                self.data['specific_risk']
            )

        h_cur = hold_weight.values - w_b

        # Step 3: 最优持仓（可选）
        logger.info('Step 3: QP求解最优持仓...')
        optimizer = QPOptimizer()
        qp_result = optimizer.solve(alpha, self.V, h_cur, w_b)
        h_star = qp_result.h_optimal
        logger.info(f'QP解: active_risk={qp_result.active_risk:.4f}')
        PickleIO.write(h_star, f'{self.debug_output_dir}/h_star.pkl')

        # Step 4: 无交易区域迭代
        # logger.info('Step 4: 无交易区域迭代...')
        # iterator = NoTradeZoneIterator()
        # self.iteration_result = iterator.iterate(alpha, self.V, h_cur, w_b, self.debug_output_dir)

        # Step 5: 生成交易指令
        logger.info('Step 5: 生成交易指令...')
        # h_final = pd.Series(self.iteration_result.h_final, index=self.data['instruments'])
        h_final = pd.Series(h_star, index=self.data['instruments'])
        h_cur_series = pd.Series(h_cur, index=self.data['instruments'])

        trade_orders = self.trade_generator.generate(
            h_final=h_final,
            h_cur=h_cur_series,
            portfolio_value=total_value,
            prices=self.data['prices'],
            w_b=self.data['benchmark_weights']
        )
        PickleIO.write(trade_orders, f'{self.debug_output_dir}/trade_orders.pkl')

        # Step 6: 保存结果
        # 保存到MySQL
        if save_to_mysql:
            self.output_manager.save_to_mysql(
                trade_orders, self.calc_date, self.portfolio_name,
                total_value, self.data['current_position']
            )

        logger.info('=' * 60)
        logger.info('投资组合优化完成')
        logger.info('=' * 60)
