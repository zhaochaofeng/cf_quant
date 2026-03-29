"""
无交易区域迭代算法模块
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional
from dataclasses import dataclass

from barra.portfolio.config import OPTIMIZATION_PARAMS, ITERATION_PARAMS
from barra.portfolio.optimizer import compute_mcva
from utils import LoggerFactory

logger = LoggerFactory.get_logger(__name__)


@dataclass
class IterationResult:
    """迭代结果数据类"""
    h_final: np.ndarray             # 最终主动头寸
    marginal_contributions: np.ndarray  # 边际贡献
    iterations: int                 # 迭代次数
    converged: bool                 # 是否收敛
    active_risk: float              # 主动风险
    in_no_trade_zone: np.ndarray    # 是否在无交易区域


class NoTradeZoneIterator:
    """无交易区域迭代算法
    
    核心思想：
    当交易成本为线性时，最优解满足 MCVA = 0。
    但由于实际交易成本包含固定部分，引入无交易区域：
    - 若 MCVA_n ∈ [-c_s, c_b]，不交易
    - 若 MCVA_n > c_b，买入
    - 若 MCVA_n < -c_s，卖出
    
    迭代调整直至收敛。
    """
    
    def __init__(
        self,
        risk_aversion: float = None,
        buy_cost: Optional[np.ndarray] = None,
        sell_cost: Optional[np.ndarray] = None,
        max_iterations: int = None,
        convergence_threshold: float = None
    ):
        """初始化迭代器
        
        Args:
            risk_aversion: 风险厌恶系数 λ
            buy_cost: 买入成本率向量 c_b
            sell_cost: 卖出成本率向量 c_s
            max_iterations: 最大迭代次数
            convergence_threshold: 收敛阈值
        """
        opt_params = OPTIMIZATION_PARAMS.copy()
        iter_params = ITERATION_PARAMS.copy()
        
        self.risk_aversion = risk_aversion or opt_params['risk_aversion']
        self.max_iterations = max_iterations or iter_params['max_iterations']
        self.convergence_threshold = convergence_threshold or iter_params['convergence_threshold']
        
        # 成本向量
        self.buy_cost_scalar = opt_params['buy_cost_rate']
        self.sell_cost_scalar = opt_params['sell_cost_rate']
        self.buy_cost = buy_cost
        self.sell_cost = sell_cost
    
    def iterate(
        self,
        alpha: np.ndarray,
        V: np.ndarray,
        h_cur: np.ndarray,
        w_b: np.ndarray,
        h_init: Optional[np.ndarray] = None
    ) -> IterationResult:
        """执行无交易区域迭代
        
        迭代步骤：
        1. 计算 MCVA = α - 2λVh
        2. 判断每只股票是否在无交易区域
        3. 调整边界外的头寸
        4. 施加现金中性约束
        5. 施加卖空约束
        6. 检查收敛条件
        
        Args:
            alpha: Alpha向量 (N,)
            V: 资产协方差矩阵 (N, N)
            h_cur: 当前主动头寸 (N,)
            w_b: 基准权重 (N,)
            h_init: 初始头寸（可选，默认用h_cur）
            
        Returns:
            IterationResult: 迭代结果
        """
        N = len(alpha)
        logger.info(f'开始无交易区域迭代: N={N}, max_iter={self.max_iterations}')
        
        # 处理成本向量
        c_b = self.buy_cost if self.buy_cost is not None else np.full(N, self.buy_cost_scalar)
        c_s = self.sell_cost if self.sell_cost is not None else np.full(N, self.sell_cost_scalar)
        
        # 预计算对角元素（用于单股票调整）
        V_diag = np.diag(V)
        # 避免除零
        V_diag = np.maximum(V_diag, 1e-12)
        
        # 初始化头寸
        h = h_init.copy() if h_init is not None else h_cur.copy()
        
        # 阻尼因子（步长限制，防止数值爆炸）
        damping = 0.5
        
        # 迭代
        converged = False
        iteration = 0
        
        for iteration in range(1, self.max_iterations + 1):
            # 计算边际贡献
            mcva = compute_mcva(alpha, V, h, self.risk_aversion)
            
            # 计算调整量
            delta_h = np.zeros(N)
            
            # 买入情形: MCVA > c_b
            buy_mask = mcva > c_b
            delta_h[buy_mask] = (mcva[buy_mask] - c_b[buy_mask]) / (2 * self.risk_aversion * V_diag[buy_mask])
            
            # 卖出情形: MCVA < -c_s
            sell_mask = mcva < -c_s
            delta_h[sell_mask] = (mcva[sell_mask] + c_s[sell_mask]) / (2 * self.risk_aversion * V_diag[sell_mask])
            
            # 应用阻尼因子（限制步长）
            delta_h = delta_h * damping
            
            # 更新头寸
            h_new = h + delta_h
            
            # 施加现金中性约束: sum(h) = 0
            h_new = h_new - np.mean(h_new)
            
            # 施加卖空约束: h >= -w_b
            h_new = np.maximum(h_new, -w_b)
            
            # 限制头寸范围（防止数值溢出）
            h_new = np.clip(h_new, -1.0, 1.0)
            
            # 检查数值溢出
            if not np.all(np.isfinite(h_new)):
                logger.warning(f'数值溢出检测，使用前一次有效结果')
                break
            
            # 检查收敛
            change = np.linalg.norm(h_new - h)
            
            if change < self.convergence_threshold:
                converged = True
                logger.info(f'迭代收敛: iter={iteration}, change={change:.2e}')
                break
            
            h = h_new
        
        if not converged:
            logger.warning(f'迭代未收敛: iter={iteration}, change={change:.2e}')
        
        # 最终计算
        mcva_final = compute_mcva(alpha, V, h, self.risk_aversion)
        active_risk = np.sqrt(h @ V @ h)
        
        # 判断是否在无交易区域
        in_no_trade_zone = (mcva_final >= -c_s) & (mcva_final <= c_b)
        
        logger.info(f'迭代完成: iter={iteration}, converged={converged}, '
                   f'active_risk={active_risk:.4f}, '
                   f'in_zone_ratio={in_no_trade_zone.mean():.2%}')
        
        return IterationResult(
            h_final=h,
            marginal_contributions=mcva_final,
            iterations=iteration,
            converged=converged,
            active_risk=active_risk,
            in_no_trade_zone=in_no_trade_zone
        )


def build_asset_covariance(
    exposure: pd.DataFrame,
    factor_cov: pd.DataFrame,
    specific_risk: pd.Series,
    regularization: float = 1e-6
) -> np.ndarray:
    """构建资产协方差矩阵 V = X*F*X^T + Δ

    Args:
        exposure: 因子暴露矩阵 X, shape(N, K)
        factor_cov: 因子协方差矩阵 F, shape(K, K)
        specific_risk: 特异风险方差 Δ的对角元素, shape(N,)
        regularization: 正则化参数，添加到对角线改善条件数

    Returns:
        资产协方差矩阵 V, shape(N, N)
    """
    # 对齐因子
    common_factors = exposure.columns.intersection(factor_cov.index)
    common_instruments = exposure.index.intersection(specific_risk.index)

    X = exposure.loc[common_instruments, common_factors].values
    F = factor_cov.loc[common_factors, common_factors].values
    delta = specific_risk.loc[common_instruments].values

    N = len(common_instruments)

    # 计算 X @ F @ X.T
    XFXT = X @ F @ X.T

    # 加上特异风险对角矩阵 + 正则化项
    V = XFXT + np.diag(delta) + regularization * np.eye(N)

    return V
