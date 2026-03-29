"""
凸二次规划优化模块（cvxpy实现）
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional
from dataclasses import dataclass

from barra.portfolio.config import OPTIMIZATION_PARAMS
from utils import LoggerFactory

logger = LoggerFactory.get_logger(__name__)

# 尝试导入cvxpy
try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    logger.warning('cvxpy未安装，QP优化功能不可用')


@dataclass
class OptimizationResult:
    """优化结果数据类"""
    h_optimal: np.ndarray       # 最优主动头寸
    b: np.ndarray               # 买入权重变化
    s: np.ndarray               # 卖出权重变化
    active_risk: float          # 主动风险
    objective_value: float      # 目标函数值
    status: str                 # 求解状态
    iterations: int             # 求解迭代次数


class QPOptimizer:
    """凸二次规划优化器

    求解以下优化问题：
        min  0.5 * h' * (2λV) * h - α'h + c_b'b + c_s's

    约束条件：
        1. 交易量关系: h - h_cur = b - s
        2. 现金中性: sum(h) = 0
        3. 禁止卖空: h >= -w_b
        4. 换手率限制: sum(b) + sum(s) <= T_max
        5. 个股上限: h <= U
        6. 非负性: b >= 0, s >= 0
    """

    def __init__(
        self,
        risk_aversion: float = None,
        buy_cost: Optional[np.ndarray] = None,
        sell_cost: Optional[np.ndarray] = None,
        max_turnover: float = None,
        max_active_position: float = None
    ):
        """初始化优化器

        Args:
            risk_aversion: 风险厌恶系数 λ
            buy_cost: 买入成本率向量 c_b（长度为N的向量）
            sell_cost: 卖出成本率向量 c_s（长度为N的向量）
            max_turnover: 换手率上限 T_max
            max_active_position: 个股主动头寸上限 U
        """
        if not CVXPY_AVAILABLE:
            raise ImportError('cvxpy未安装，请先安装: pip install cvxpy')

        params = OPTIMIZATION_PARAMS.copy()

        self.risk_aversion = risk_aversion or params['risk_aversion']
        self.max_turnover = max_turnover or params['max_turnover']
        self.max_active_position = max_active_position or params['max_active_position']

        # 成本向量（可以是标量或向量）
        self.buy_cost_scalar = params['buy_cost_rate']
        self.sell_cost_scalar = params['sell_cost_rate']
        self.buy_cost = buy_cost
        self.sell_cost = sell_cost

    def solve(
        self,
        alpha: np.ndarray,
        V: np.ndarray,
        h_cur: np.ndarray,
        w_b: np.ndarray
    ) -> OptimizationResult:
        """求解二次规划问题

        Args:
            alpha: Alpha向量 (N,)
            V: 资产协方差矩阵 (N, N)
            h_cur: 当前主动头寸 (N,)
            w_b: 基准权重 (N,)

        Returns:
            OptimizationResult: 优化结果
        """
        N = len(alpha)
        logger.info(f'开始QP优化: N={N}, λ={self.risk_aversion}')

        # 处理成本向量
        c_b = self.buy_cost if self.buy_cost is not None else np.full(N, self.buy_cost_scalar)
        c_s = self.sell_cost if self.sell_cost is not None else np.full(N, self.sell_cost_scalar)

        # 个股上限向量
        U = np.full(N, self.max_active_position)

        # 检查协方差矩阵正定性并正则化
        V_processed = self._ensure_positive_definite(V)

        # 构建cvxpy问题
        h = cp.Variable(N)
        b = cp.Variable(N, nonneg=True)
        s = cp.Variable(N, nonneg=True)

        # 目标函数: min 0.5 * h' * Q * h - alpha @ h + c_b @ b + c_s @ s
        # Q = 2 * self.risk_aversion * V_processed
        Q = 2 * self.risk_aversion * V_processed

        # 使用psd_wrap包装Q以确保数值稳定性
        try:
            objective = cp.Minimize(
                0.5 * cp.quad_form(h, cp.psd_wrap(Q)) - alpha @ h + c_b @ b + c_s @ s
            )
        except AttributeError:
            # 旧版本cvxpy可能没有psd_wrap
            objective = cp.Minimize(
                0.5 * cp.quad_form(h, Q) - alpha @ h + c_b @ b + c_s @ s
            )

        # 约束条件
        constraints = [
            h - h_cur == b - s,              # 交易量关系
            cp.sum(h) == 0,                  # 现金中性
            h >= -w_b,                       # 禁止卖空
            h <= U,                          # 个股上限
            cp.sum(b) + cp.sum(s) <= self.max_turnover  # 换手率限制
        ]

        # 求解
        problem = cp.Problem(objective, constraints)

        # 尝试多个求解器，从最稳健的开始
        solvers_to_try = [
            (cp.CLARABEL, {'verbose': False}),
            (cp.SCS, {'verbose': False, 'max_iters': 10000}),
            (cp.OSQP, {'verbose': False, 'max_iter': 10000}),
            (cp.ECOS, {'verbose': False})
        ]

        solved = False
        for solver, kwargs in solvers_to_try:
            try:
                problem.solve(solver=solver, **kwargs)
                if problem.status in ['optimal', 'optimal_inaccurate']:
                    solved = True
                    logger.info(f'QP求解成功: solver={solver}, status={problem.status}')
                    break
                else:
                    logger.debug(f'{solver}返回: {problem.status}')
            except Exception as e:
                logger.debug(f'{solver}求解失败: {e}')
                continue

        if not solved:
            # 最后尝试默认求解器
            try:
                problem.solve(verbose=False)
            except Exception as e:
                logger.error(f'所有求解器都失败: {e}')

        # 检查求解状态
        if problem.status not in ['optimal', 'optimal_inaccurate']:
            logger.warning(f'QP求解状态: {problem.status}')

        # 提取结果
        h_optimal = h.value if h.value is not None else h_cur
        b_optimal = b.value if b.value is not None else np.zeros(N)
        s_optimal = s.value if s.value is not None else np.zeros(N)

        # 计算主动风险
        active_risk = np.sqrt(h_optimal @ V @ h_optimal)

        logger.info(f'QP求解完成: status={problem.status}, '
                   f'active_risk={active_risk:.4f}, '
                   f'turnover={np.sum(b_optimal) + np.sum(s_optimal):.4f}')

        return OptimizationResult(
            h_optimal=h_optimal,
            b=b_optimal,
            s=s_optimal,
            active_risk=active_risk,
            objective_value=problem.value,
            status=problem.status,
            iterations=problem.solver_stats.num_iters if hasattr(problem, 'solver_stats') else 0
        )

    def _ensure_positive_definite(self, V: np.ndarray) -> np.ndarray:
        """确保协方差矩阵正定且条件数合理

        Args:
            V: 协方差矩阵

        Returns:
            正定的协方差矩阵，条件数改善
        """
        N = V.shape[0]

        # 添加正则化项改善条件数
        # 先检测当前最小特征值
        eigenvalues = np.linalg.eigvalsh(V)
        min_eigenvalue = eigenvalues.min()
        max_eigenvalue = eigenvalues.max()

        # 计算条件数
        cond_number = max_eigenvalue / max(min_eigenvalue, 1e-15)

        # 如果条件数太大或最小特征值太小，添加正则化
        if cond_number > 1e8 or min_eigenvalue < 1e-6:
            # 添加正则化项使最小特征值至少为1e-6
            regularization = max(1e-6 - min_eigenvalue, 1e-6)
            V = V + regularization * np.eye(N)
            logger.info(f'协方差矩阵正则化: min_eig={min_eigenvalue:.2e}, cond={cond_number:.2e}, '
                       f'regularization={regularization:.2e}')

        return V


def compute_mcva(
    alpha: np.ndarray,
    V: np.ndarray,
    h: np.ndarray,
    risk_aversion: float
) -> np.ndarray:
    """计算边际贡献 MCVA = α - 2λVh

    Args:
        alpha: Alpha向量 (N,)
        V: 资产协方差矩阵 (N, N)
        h: 主动头寸向量 (N,)
        risk_aversion: 风险厌恶系数 λ

    Returns:
        边际贡献向量 (N,)
    """
    return alpha - 2 * risk_aversion * (V @ h)
