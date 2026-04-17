"""因子协方差矩阵估计模块"""
import numpy as np
import pandas as pd

from utils import LoggerFactory

logger = LoggerFactory.get_logger(__name__)


class FactorCovarianceEstimator:
    """因子协方差矩阵估计器"""

    def __init__(self):
        self.covariance_matrix = None
        self.mean_returns = None

    def estimate_sample_covariance(
            self, factor_returns: pd.DataFrame) -> pd.DataFrame:
        """
        估计样本协方差矩阵（基准方法）

        F = 1/(T-1) * sum((b_t - b_bar) * (b_t - b_bar)^T)

        Args:
            factor_returns: 因子收益率，index=date, columns=factors

        Returns:
            协方差矩阵 DataFrame
        """
        logger.info('估计样本协方差矩阵...')
        col_len = factor_returns.shape[1]
        cov_matrix = factor_returns.cov(min_periods=col_len)
        self.covariance_matrix = cov_matrix
        self.mean_returns = factor_returns.mean()
        logger.info(f'样本协方差完成，{len(factor_returns)}期，'
                     f'维度: {cov_matrix.shape}')
        return cov_matrix

    def estimate_barra_covariance(
            self, factor_returns: pd.DataFrame,
            half_life_corr: int = 252,
            half_life_var: int = 42,
            init_periods: int = 20) -> pd.DataFrame:
        """
        Barra 双半衰期 EWMA 协方差估计

        方差与相关系数分离估计，半衰期分别平滑：
        - 长半衰期 H_C 控制相关系数矩阵（稳定结构）
        - 短半衰期 H_D 控制方差（快速响应波动聚集）

        Args:
            factor_returns: 因子收益率，index=date, columns=factors
            half_life_corr: 相关系数半衰期 H_C（交易日）
            half_life_var: 方差半衰期 H_D（交易日）
            init_periods: 初始化等权样本协方差窗口 m

        Returns:
            协方差矩阵 DataFrame (K x K)
        """
        logger.info(f'估计 Barra EWMA 协方差矩阵 '
                     f'(H_C={half_life_corr}, H_D={half_life_var}, '
                     f'm={init_periods})...')

        T, K = factor_returns.shape
        if T <= init_periods:
            raise ValueError(
                f'样本数({T})须大于初始化窗口({init_periods})')

        values = factor_returns.values  # (T, K)

        # 衰减因子
        lambda_c = 0.5 ** (1.0 / half_life_corr)
        lambda_d = 0.5 ** (1.0 / half_life_var)

        # 初始化：前 m 期等权样本协方差
        F_raw = np.cov(values[:init_periods].T, ddof=1)  # (K, K)
        var_smooth = np.diag(F_raw).copy()  # (K,)

        # 迭代更新 (t = m, m+1, ..., T-1)
        for t in range(init_periods, T):
            b_t = values[t]  # (K,)
            outer = np.outer(b_t, b_t)  # (K, K)

            # 更新原始协方差矩阵（长半衰期）
            F_raw = lambda_c * F_raw + (1 - lambda_c) * outer

            # 提取基础方差并二次平滑（短半衰期）
            V_t = np.diag(F_raw)
            var_smooth = lambda_d * var_smooth + (1 - lambda_d) * V_t

        # 提取相关系数矩阵 C
        diag_raw = np.diag(F_raw)
        denom = np.sqrt(np.outer(diag_raw, diag_raw))
        # 防止除零
        denom = np.where(denom < 1e-20, 1e-20, denom)
        C = F_raw / denom

        # 构建标准差矩阵 D
        D = np.diag(np.sqrt(np.maximum(var_smooth, 1e-20)))

        # 合成最终协方差矩阵 F = D @ C @ D
        F_final = D @ C @ D

        # 正定性检验
        F_final = self._ensure_positive_definite(F_final)

        cov_df = pd.DataFrame(
            F_final,
            index=factor_returns.columns,
            columns=factor_returns.columns,
        )
        self.covariance_matrix = cov_df
        self.mean_returns = factor_returns.mean()

        logger.info(f'Barra EWMA 协方差完成，{T}期，维度: {cov_df.shape}')
        return cov_df

    @staticmethod
    def _ensure_positive_definite(matrix: np.ndarray,
                                  min_eigenvalue: float = 1e-10
                                  ) -> np.ndarray:
        """
        确保矩阵正定，非正定时调整特征值

        Args:
            matrix: 输入矩阵
            min_eigenvalue: 最小特征值阈值

        Returns:
            正定矩阵
        """
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        if np.all(eigenvalues > 0):
            return matrix
        logger.warning(f'矩阵非正定，最小特征值={eigenvalues.min():.6e}，'
                        f'进行特征值调整')
        eigenvalues = np.maximum(eigenvalues, min_eigenvalue)
        return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    def get_covariance_matrix(self) -> pd.DataFrame:
        """获取协方差矩阵"""
        if self.covariance_matrix is None:
            raise ValueError('协方差矩阵尚未估计')
        return self.covariance_matrix

    def get_factor_volatility(self) -> pd.Series:
        """获取因子波动率（协方差矩阵对角线的平方根）"""
        if self.covariance_matrix is None:
            raise ValueError('协方差矩阵尚未估计')
        vol = np.sqrt(np.diag(self.covariance_matrix))
        return pd.Series(vol, index=self.covariance_matrix.index)

    def get_correlation_matrix(self) -> pd.DataFrame:
        """获取相关系数矩阵"""
        if self.covariance_matrix is None:
            raise ValueError('协方差矩阵尚未估计')
        cov = self.covariance_matrix.values
        std = np.sqrt(np.diag(cov))
        std = np.where(std < 1e-20, 1e-20, std)
        corr = cov / np.outer(std, std)
        return pd.DataFrame(
            corr,
            index=self.covariance_matrix.index,
            columns=self.covariance_matrix.columns,
        )
