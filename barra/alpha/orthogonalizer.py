"""
正交化模块 - 多信号Cholesky正交化与IC加权合成

当K>1时启用，K=1时跳过
"""
import pandas as pd
import numpy as np
from scipy.linalg import solve_triangular

from .config import ROLLING_WINDOW, IC_LAG, MIN_IC_WINDOW
from utils import LoggerFactory

logger = LoggerFactory.get_logger(__name__)


class AlphaOrthogonalizer:
    """多信号Alpha正交化器

    流程：
    1. 估计Alpha协方差矩阵 Sigma_alpha (K x K)
    2. Cholesky分解: Sigma = L @ L^T
    3. 正交化变换: y = L^{-1} (alpha - mean)
    4. 估计正交化信号IC: gamma_j = Corr(y_j(t), theta(t+2))
    5. 合成: alpha_n = sum(gamma_j * y_{j,n})
    """

    def __init__(self, window: int = ROLLING_WINDOW):
        """初始化

        Args:
            window: 滚动窗口（交易日）
        """
        self.window = window

    def fit_and_transform(
        self,
        alpha_history: dict[str, pd.DataFrame],
        alpha_current: dict[str, pd.Series],
        residuals: pd.DataFrame,
        as_of_date: str
    ) -> pd.Series:
        """从历史Alpha矩阵到最终合成Alpha

        Args:
            alpha_history: {signal_name: DataFrame(MultiIndex(instrument,datetime), col='alpha')}
                过去window天的单信号alpha历史
            alpha_current: {signal_name: Series(instrument -> alpha)}
                当日的单信号alpha
            residuals: 残差收益率，MultiIndex(instrument, datetime), col='residual'
            as_of_date: 计算截止日期

        Returns:
            Series(instrument -> final_alpha)
        """
        signal_names = list(alpha_history.keys())
        K = len(signal_names)
        logger.info(f'正交化: {K}个信号')

        # Step 1: 构建历史Alpha矩阵 (N*T x K)
        alpha_matrix, valid_index = self._build_alpha_matrix(alpha_history, as_of_date)

        # Step 2: 估计协方差矩阵
        alpha_mean = alpha_matrix.mean(axis=0)
        sigma = self._estimate_covariance(alpha_matrix)

        # Step 3: Cholesky分解
        L = self._cholesky_decompose(sigma)

        # Step 4: 正交化历史数据
        y_matrix = self._orthogonalize(alpha_matrix, alpha_mean, L)

        # Step 5: 估计正交化IC
        y_df = pd.DataFrame(y_matrix, index=valid_index, columns=signal_names)
        gamma = self._compute_gamma(y_df, residuals, as_of_date)

        # Step 6: 正交化当日Alpha并合成
        alpha_cur_vec = self._build_current_vector(alpha_current, signal_names)
        y_current = self._orthogonalize_vector(
            alpha_cur_vec.values, alpha_mean, L
        )
        final_alpha = y_current @ gamma

        result = pd.Series(final_alpha, index=alpha_cur_vec.index, name='alpha')
        logger.info(f'正交化合成完成: {len(result)}只股票')
        return result

    def _build_alpha_matrix(
        self, alpha_history: dict[str, pd.DataFrame], as_of_date: str
    ) -> tuple[np.ndarray, pd.MultiIndex]:
        """构建历史Alpha矩阵

        Args:
            alpha_history: 各信号的历史alpha
            as_of_date: 截止日期

        Returns:
            (matrix: shape (M, K), valid_index: MultiIndex)
        """
        as_of_ts = pd.Timestamp(as_of_date)
        signal_names = list(alpha_history.keys())

        # 合并所有信号的alpha为一个DataFrame
        merged = None
        for name in signal_names:
            df = alpha_history[name].copy()
            # 过滤日期
            dates = df.index.get_level_values('datetime')
            df = df.loc[dates <= as_of_ts]
            # 取最近window天
            recent_dates = dates[dates <= as_of_ts].unique().sort_values()
            recent_dates = recent_dates[-self.window:]
            df = df.loc[df.index.get_level_values('datetime').isin(recent_dates)]
            df.columns = [name]

            if merged is None:
                merged = df
            else:
                merged = merged.join(df, how='inner')

        # 删除含NaN的行
        merged = merged.dropna()
        return merged.values, merged.index

    def _estimate_covariance(self, alpha_matrix: np.ndarray) -> np.ndarray:
        """估计Alpha协方差矩阵 (K x K)

        Args:
            alpha_matrix: shape (M, K)

        Returns:
            协方差矩阵 (K x K)
        """
        sigma = np.cov(alpha_matrix, rowvar=False)
        if sigma.ndim == 0:
            sigma = np.array([[sigma]])
        return sigma

    def _cholesky_decompose(self, sigma: np.ndarray) -> np.ndarray:
        """Cholesky分解，返回下三角矩阵L

        Sigma = L @ L^T

        Args:
            sigma: 协方差矩阵 (K x K)

        Returns:
            下三角矩阵L (K x K)
        """
        # 加小对角正则化保证正定
        K = sigma.shape[0]
        eps = 1e-10
        sigma_reg = sigma + eps * np.eye(K)

        try:
            L = np.linalg.cholesky(sigma_reg)
        except np.linalg.LinAlgError:
            # 增大正则化
            logger.warning('Cholesky分解失败，增大正则化')
            sigma_reg = sigma + 1e-6 * np.eye(K)
            L = np.linalg.cholesky(sigma_reg)

        return L

    def _orthogonalize(
        self, alpha_matrix: np.ndarray, alpha_mean: np.ndarray, L: np.ndarray
    ) -> np.ndarray:
        """正交化变换: y = L^{-1} (alpha - mean)

        Args:
            alpha_matrix: shape (M, K)
            alpha_mean: shape (K,)
            L: 下三角矩阵 (K, K)

        Returns:
            正交化矩阵 (M, K)
        """
        centered = alpha_matrix - alpha_mean
        # 对每一行求解 L @ y = centered
        # 等价于 y = L^{-1} @ centered^T，逐列求解
        y = solve_triangular(L, centered.T, lower=True).T
        return y

    def _orthogonalize_vector(
        self, alpha_vec: np.ndarray, alpha_mean: np.ndarray, L: np.ndarray
    ) -> np.ndarray:
        """对单个Alpha向量做正交化

        Args:
            alpha_vec: shape (N, K)，N只股票的K维alpha
            alpha_mean: shape (K,)
            L: 下三角矩阵 (K, K)

        Returns:
            shape (N, K)
        """
        centered = alpha_vec - alpha_mean
        y = solve_triangular(L, centered.T, lower=True).T
        return y

    def _compute_gamma(
        self, y_df: pd.DataFrame, residuals: pd.DataFrame, as_of_date: str
    ) -> np.ndarray:
        """计算正交化信号的IC

        gamma_j = Corr(y_j(t), theta(t+lag))

        Args:
            y_df: 正交化后的信号，MultiIndex(instrument, datetime), columns=signal_names
            residuals: 残差收益率
            as_of_date: 截止日期

        Returns:
            gamma向量 (K,)
        """
        from .ic_estimator import ICEstimator
        ic_est = ICEstimator(window=self.window, lag=IC_LAG)

        K = y_df.shape[1]
        gamma = np.zeros(K)

        for j, col in enumerate(y_df.columns):
            y_j = y_df[[col]].copy()
            y_j.columns = ['z_cs']  # ICEstimator期望此列名
            gamma[j] = ic_est.compute_ic(y_j, residuals, as_of_date)

        logger.info(f'正交化IC (gamma): {gamma}')
        return gamma

    def _build_current_vector(
        self, alpha_current: dict[str, pd.Series], signal_names: list[str]
    ) -> pd.DataFrame:
        """构建当日Alpha向量矩阵

        Args:
            alpha_current: {signal_name: Series(instrument -> alpha)}
            signal_names: 信号名列表（保证顺序）

        Returns:
            DataFrame(instrument, K columns)
        """
        merged = None
        for name in signal_names:
            s = alpha_current[name].rename(name)
            if merged is None:
                merged = s.to_frame()
            else:
                merged = merged.join(s.to_frame(), how='inner')
        return merged.dropna()
