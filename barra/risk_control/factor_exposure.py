"""
因子暴露矩阵构建模块
包含：原始因子计算、去极值、中性化、正交化、标准化、行业因子合并
"""
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from .output import RiskOutputManager

from utils.multiprocess import multiprocessing_wrapper

from .config import STYLE_FACTOR_LIST, FACTOR_FUNCTIONS, INDUSTRY_MAPPING, INDUSTRY_NAMES
from utils import LoggerFactory, winsorize, neutralize, standardize
logger = LoggerFactory.get_logger(__name__)

class FactorExposureBuilder:
    """因子暴露矩阵构建器"""

    def calculate_raw_factors(self, raw_data: pd.DataFrame,
                              n_jobs: int = 1
                              ) -> pd.DataFrame:
        """
        计算原始CNE6因子值
        
        Args:
            raw_data: 原始数据DataFrame，包含所有必要字段
            n_jobs: 并行进程数
            
        Returns:
            DataFrame, index=(instrument, datetime), columns=因子名称
        """
        logger.info("开始计算原始因子值...")
        raw_data = raw_data.sort_index()
        
        # 准备并行计算任务
        factor_results = {}
        
        if n_jobs > 1:
            # 并行计算
            func_calls = []
            for factor_name in STYLE_FACTOR_LIST:
                if factor_name in FACTOR_FUNCTIONS:
                    func_calls.append((
                        self._compute_single_factor,
                        (raw_data, factor_name, FACTOR_FUNCTIONS[factor_name])
                    ))
            
            results = multiprocessing_wrapper(func_calls, n=n_jobs)
            for factor_name, result in results:
                if result is not None:
                    factor_results[factor_name] = result
        else:
            # 串行计算
            for factor_name in STYLE_FACTOR_LIST:
                if factor_name in FACTOR_FUNCTIONS:
                    _, result = self._compute_single_factor(
                        raw_data, factor_name, FACTOR_FUNCTIONS[factor_name]
                    )
                    if result is not None:
                        factor_results[factor_name] = result
        
        # 合并所有因子
        factor_df = pd.DataFrame(index=raw_data.index)
        for factor_name, series in factor_results.items():
            factor_df[factor_name] = series
        
        logger.info(f"原始因子计算完成，共{len(factor_df.columns)}个因子")

        return factor_df
    
    def _compute_single_factor(self, raw_data: pd.DataFrame, 
                               factor_name: str, 
                               factor_func) -> tuple:
        """
        计算单个因子（用于并行）
        raw_data：原始字段数据
        factor_name：因子名称
        factor_func：因子计算函数
        
        Returns:
            (factor_name, series)
        """
        try:
            result = factor_func(raw_data)
            if result is not None and not result.empty:
                series = result.iloc[:, 0]  # 取第一列
                return factor_name, series
        except Exception as e:
            err_msg = f"因子{factor_name}计算失败: {str(e)}"
            logger.error(err_msg)
            raise Exception(err_msg)

    def neutralize_factors(self, factor_df: pd.DataFrame,
                          industry_df: pd.DataFrame,
                          market_cap_df: pd.DataFrame) -> pd.DataFrame:
        """
        行业/市值中性化
        对每个因子，用行业和对数市值做回归，取残差
        去掉一个行业，添加截距项

        Args:
            factor_df: 因子数据，index=(instrument, datetime), columns=因子名
            industry_df: 行业数据，index=(instrument, datetime)
            market_cap_df: 市值数据，index=(instrument, datetime), columns包含'circ_mv'

        Returns:
            中性化后的因子数据
        """
        logger.info("进行行业/市值中性化 ...")

        # 行业哑变量：去掉第一列避免行业因子与截距项出现多重共线性
        industry_dummies = get_industry_dummies(industry_df, drop_first=True, prefix='ind')
        industry_dummies.drop(columns=['ind_nan'], inplace=True)

        # 对数市值. CNE6 的 LNCAP 因子已经捕获了 对数流通市值 信息
        log_mv = np.log(market_cap_df[['circ_mv']].clip(lower=1e-10))
        log_mv.columns = ['log_mv']

        # 基础自变量：行业哑变量 + 对数市值
        base_x = industry_dummies.join(log_mv, how='inner')

        # 对每个因子分别进行中性化
        result_df = pd.DataFrame(index=factor_df.index)

        for factor_name in factor_df.columns:
            y = factor_df[factor_name]

            # 对齐索引，去除 NaN
            common_idx = y.index.intersection(base_x.index)
            y_aligned = y.loc[common_idx]
            x_aligned = base_x.loc[common_idx]
            valid_mask = y_aligned.notna() & x_aligned.notna().all(axis=1)

            if valid_mask.sum() / factor_df.shape[1] < 0.5:
                err_msg = f"因子{factor_name}有效样本低于 50%"
                logger.error(err_msg)
                raise Exception(err_msg)

            resid = neutralize(
                y_aligned[valid_mask], x_aligned.loc[valid_mask], intercept=True
            )

            # 残差写回，NaN 位置保留
            result_col = pd.Series(np.nan, index=factor_df.index, dtype=float)
            result_col.loc[resid.index] = resid
            result_df[factor_name] = result_col

        return result_df

    def verify_orthogonality(self, factor_df: pd.DataFrame, 
                            threshold: float = 0.1) -> bool:
        """
        验证因子正交性
        
        Args:
            factor_df: 因子数据
            threshold: 相关系数阈值，超过则可能存在严重多重共线性
            
        Returns:
            是否通过正交性检验
        """
        logger.info("验证因子正交性...")
        # 计算相关系数矩阵
        corr_matrix = factor_df.corr()
        
        # 对角线元素填充为0
        # 检查非对角元素
        np.fill_diagonal(corr_matrix.values, 0)
        max_corr = corr_matrix.abs().max().max()
        
        logger.info(f"绝对值最大的相关系数: {max_corr:.4f}")
        
        if max_corr > threshold:
            logger.warning(f"警告：存在相关系数超过阈值{threshold}的因子对")
            # 打印高相关因子对
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > threshold:
                        high_corr_pairs.append((
                            corr_matrix.columns[i], 
                            corr_matrix.columns[j], 
                            corr_matrix.iloc[i, j]
                        ))
            high_corr_pairs = sorted(high_corr_pairs, key=lambda x: abs(x[-1]), reverse=True)
            logger.info('高相关因子对: {}'.format(len(high_corr_pairs)))
            for f1, f2, corr in high_corr_pairs:
                logger.info(f"  {f1} - {f2}: {corr:.4f}")
            return False
        
        logger.info("正交性检验通过")
        return True

    def verify_vif(self, factor_df: pd.DataFrame,
                   threshold: float = 10.0) -> bool:
        """
        方差膨胀因子(VIF)检验，检测因子间多重共线性

        对每个因子 k，以其为因变量、其余因子为自变量做 OLS，
        VIF_k = 1 / (1 - R^2_k)，超过阈值发出警告。

        Args:
            factor_df: 因子数据，columns=因子名
            threshold: VIF 阈值，默认10.0

        Returns:
            是否通过 VIF 检验（所有因子 VIF < threshold）
        """
        logger.info('VIF 检验...')
        df = factor_df.dropna()
        if df.empty:
            logger.warning('VIF 检验: 无有效数据')
            return False
        if df.shape[0] / factor_df.shape[0] < 0.5:
            err_msg = 'VIF 检验: 有效数据占比 {} 低于 50%'.format(df.shape[0] / factor_df.shape[0])
            logger.warning(err_msg)
            raise Exception(err_msg)

        factors = df.columns.tolist()
        X_all = df.values
        vif_dict = {}
        for i, name in enumerate(factors):
            y = X_all[:, i]
            idx = list(range(len(factors)))
            idx.remove(i)
            X = np.column_stack([np.ones(X_all.shape[0]), X_all[:, idx]])  # 添加截距项
            try:
                # beta = np.linalg.lstsq(X, y, rcond=None)[0]
                from utils import WLS
                beta = WLS(y, X, intercept=False, verbose=False, backend='statsmodels')
                sse = np.sum((y - X @ beta) ** 2)   # 残差平方和
                sst = np.sum((y - y.mean()) ** 2)   # 总平方和
                r2 = 1 - sse / sst if sst > 0 else 0.0
                vif = 1.0 / (1.0 - r2) if r2 < 1.0 else np.inf
            except Exception:
                vif = np.inf
            vif_dict[name] = vif

        passed = True
        for name, vif in sorted(vif_dict.items(), key=lambda x: -x[1]):
            if vif > threshold:
                logger.warning(f'  VIF 超阈值: {name} = {vif:.2f}')
                passed = False

        if passed:
            logger.info(f'VIF 检验通过 (max={max(vif_dict.values()):.2f})')
        else:
            logger.warning('VIF 检验未通过，存在多重共线性')

        return passed

    def merge_industry_factors(self, style_factors: pd.DataFrame,
                               industry_dummies: pd.DataFrame) -> pd.DataFrame:
        """
        合并风格因子和行业因子
        
        Args:
            style_factors: 风格因子数据（已预处理）
            industry_dummies: 行业数据，index=(instrument, datetime)
            
        Returns:
            合并后的因子暴露矩阵
        """
        logger.info("合并行业因子...")
        # 合并风格因子和行业因子
        merged = style_factors.join(industry_dummies, how='outer')
        
        logger.info(f"合并完成，共{len(style_factors.columns)}个风格因子 + "
                    f"{len(industry_dummies.columns)}个行业因子 = "
                    f"{len(merged.columns)}个因子")
        
        return merged
    
    def build_exposure_matrix(self, raw_data: pd.DataFrame,
                             industry_df: pd.DataFrame,
                             market_cap_df: pd.DataFrame,
                             n_jobs: int = 1,
                             output_manager: RiskOutputManager = None,
                             com_dates: pd.DatetimeIndex = None,
                              ) -> pd.DataFrame:
        """
        构建完整的因子暴露矩阵

        执行完整的预处理流程：
        1. 计算原始因子
        2. 去极值
        3. 行业/市值中性化（移除因子间正交化）
        4. 标准化
        5. 合并行业因子

        Args:
            raw_data: 原始数据
            industry_df: 行业数据
            market_cap_df: 市值数据
            save_path: 保存路径
            n_jobs: 并行进程数
            output_manager: 输出管理器
            com_dates: 实际使用的因子日期


        Returns:
            完整的因子暴露矩阵
        """
        logger.info("=" * 60)
        logger.info("开始构建因子暴露矩阵...")

        # 1. 计算原始因子
        raw_factors = self.calculate_raw_factors(raw_data, n_jobs=n_jobs)
        output_manager.save_data(raw_factors, 'debug/raw_factors.parquet', type='parquet')

        # 过滤日期
        logger.info('过滤日期 ...')
        shape = raw_factors.shape
        raw_factors = raw_factors[
            raw_factors.index.get_level_values('datetime').isin(com_dates)
        ]
        logger.info('raw_factors 过滤前 shape：{}, 过滤后 shape：{}'.format(shape, raw_factors.shape))

        # 2. 去极值
        winsorized = winsorize(raw_factors, method='median', level='datetime')
        output_manager.save_data(winsorized, 'debug/winsorized.parquet', type='parquet')

        # 正交性检验
        logger.info('中性化前正交检验 ...')
        self.verify_orthogonality(winsorized, threshold=0.5)

        # 3. 行业、市值中性化
        neutralized = self.neutralize_factors(winsorized, industry_df, market_cap_df)
        output_manager.save_data(neutralized, 'debug/neutralized.parquet', type='parquet')

        # 4. 标准化
        standardized = standardize(neutralized, method='zscore', level='datetime')
        output_manager.save_data(standardized, 'debug/standardized.parquet', type='parquet')

        # 5. 验证正交性 + VIF 检验（可选，仅用于监控）
        self.verify_orthogonality(standardized, threshold=0.5)
        self.verify_vif(standardized)

        # 6. 合并行业因子
        industry_dummies = get_industry_dummies(industry_df, drop_first=False, prefix='ind')
        industry_dummies.drop(columns=['ind_nan'], inplace=True)
        exposure_matrix = self.merge_industry_factors(standardized, industry_dummies)
        output_manager.save_data(exposure_matrix, 'debug/exposure_matrix.parquet', type='parquet')

        logger.info("因子暴露矩阵构建完成")
        logger.info("=" * 60)
        
        return exposure_matrix


def get_industry_dummies(industry_df: pd.DataFrame, drop_first=False, prefix='ind') -> pd.DataFrame:
    """行业哑变量"""
    industry_dummies = pd.get_dummies(
        industry_df.iloc[:, 0], prefix=prefix, drop_first=drop_first, dtype='float'
    )
    # 行业代码转名称
    industry_name_map = {f"ind_{code}": name
                         for code, name in INDUSTRY_MAPPING.items()}
    industry_dummies = industry_dummies.rename(columns=industry_name_map)

    # 补齐缺失的行业列（CSI300 可能不含某些行业的股票）
    # all_names = INDUSTRY_NAMES if not drop_first else INDUSTRY_NAMES[1:]
    # for name in all_names:
    #     if name not in industry_dummies.columns:
    #         industry_dummies[name] = 0.0

    return industry_dummies

