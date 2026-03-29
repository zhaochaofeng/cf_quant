"""
投资组合优化数据加载模块
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union, Dict, List

from barra.portfolio.config import (
    RISK_OUTPUT_DIR, ALPHA_OUTPUT_DIR, DATA_PATHS, DEFAULT_MARKET
)
from utils import LoggerFactory

logger = LoggerFactory.get_logger(__name__)


class PortfolioDataLoader:
    """组合优化数据加载器
    
    加载并对齐以下数据：
    - Alpha预测值
    - 风险模型（因子暴露、因子协方差、特异风险）
    - 基准权重
    - 当前持仓
    - 股票价格
    """
    
    def __init__(
        self,
        market: str = DEFAULT_MARKET,
        risk_output_dir: Optional[str] = None,
        alpha_output_dir: Optional[str] = None
    ):
        """初始化数据加载器
        
        Args:
            market: 市场代码，如 'csi300'
            risk_output_dir: 风险模型输出目录
            alpha_output_dir: Alpha输出目录
        """
        self.market = market
        self.risk_output_dir = Path(risk_output_dir or RISK_OUTPUT_DIR)
        self.alpha_output_dir = Path(alpha_output_dir or ALPHA_OUTPUT_DIR)
        
    def load_alpha(self, calc_date: str) -> pd.Series:
        """加载Alpha预测值
        
        Args:
            calc_date: 计算日期，如 '2026-03-28'
            
        Returns:
            Series(index=instrument, name='alpha')
        """
        date_str = calc_date.replace('-', '')
        filepath = self.alpha_output_dir / DATA_PATHS['alpha'].format(date=date_str)
        
        if not filepath.exists():
            raise FileNotFoundError(f'Alpha文件不存在: {filepath}')
        
        df = pd.read_parquet(filepath)
        
        # 确保index为instrument
        if 'instrument' in df.columns:
            df = df.set_index('instrument')
        
        alpha = df['alpha'] if 'alpha' in df.columns else df.iloc[:, 0]
        alpha.name = 'alpha'
        
        logger.info(f'加载Alpha: {len(alpha)}只股票, 文件={filepath.name}')
        return alpha
    
    def load_risk_model(self, calc_date: str = None) -> Dict[str, Union[pd.DataFrame, pd.Series]]:
        """加载风险模型数据
        
        Args:
            calc_date: 计算日期（用于提取对应日期的因子暴露）
            
        Returns:
            {
                'exposure': DataFrame(index=instrument, columns=factors),
                'factor_cov': DataFrame(index=factors, columns=factors),
                'specific_risk': Series(index=instrument)
            }
        """
        # 加载因子暴露矩阵
        exposure_path = self.risk_output_dir / DATA_PATHS['exposure']
        if not exposure_path.exists():
            raise FileNotFoundError(f'因子暴露文件不存在: {exposure_path}')
        
        exposure_df = pd.read_parquet(exposure_path)
        
        # 提取指定日期的数据（MultiIndex: instrument, datetime）
        if isinstance(exposure_df.index, pd.MultiIndex):
            if calc_date is not None:
                # 使用指定日期
                try:
                    exposure_df = exposure_df.xs(calc_date, level='datetime')
                    logger.info(f'因子暴露提取日期: {calc_date}')
                except KeyError:
                    # 如果指定日期不存在，使用最新日期
                    latest_date = exposure_df.index.get_level_values('datetime').max()
                    exposure_df = exposure_df.xs(latest_date, level='datetime')
                    logger.warning(f'指定日期{calc_date}不存在，使用最新日期: {latest_date}')
                    calc_date = str(latest_date)
            else:
                # 使用最新日期
                latest_date = exposure_df.index.get_level_values('datetime').max()
                exposure_df = exposure_df.xs(latest_date, level='datetime')
                logger.info(f'因子暴露提取日期: {latest_date}')
        
        # 加载因子协方差矩阵
        factor_cov_path = self.risk_output_dir / DATA_PATHS['factor_cov']
        if not factor_cov_path.exists():
            raise FileNotFoundError(f'因子协方差文件不存在: {factor_cov_path}')
        
        factor_cov_df = pd.read_parquet(factor_cov_path)
        
        # 加载特异风险
        specific_risk_path = self.risk_output_dir / DATA_PATHS['specific_risk']
        if not specific_risk_path.exists():
            raise FileNotFoundError(f'特异风险文件不存在: {specific_risk_path}')
        
        specific_risk_df = pd.read_parquet(specific_risk_path)
        
        # 转换为Series（提取对角元素）
        if isinstance(specific_risk_df, pd.DataFrame):
            # 如果是对角矩阵格式，提取对角元素
            if specific_risk_df.shape[0] == specific_risk_df.shape[1]:
                specific_risk = pd.Series(
                    specific_risk_df.values.diagonal(),
                    index=specific_risk_df.index,
                    name='specific_var'
                )
            else:
                # 否则取第一列
                specific_risk = specific_risk_df.iloc[:, 0]
        else:
            specific_risk = specific_risk_df
            specific_risk.name = 'specific_var'
        
        logger.info(f'加载风险模型: exposure={exposure_df.shape}, '
                   f'factor_cov={factor_cov_df.shape}, '
                   f'specific_risk={len(specific_risk)}')
        
        return {
            'exposure': exposure_df,
            'factor_cov': factor_cov_df,
            'specific_risk': specific_risk,
            'calc_date': calc_date
        }
    
    def load_benchmark_weights(self, calc_date: str) -> pd.Series:
        """计算基准权重（沪深300成分股流通市值加权）
        
        Args:
            calc_date: 计算日期
            
        Returns:
            Series(index=instrument, name='weight')
        """
        from qlib.data import D
        
        # 获取成分股
        instruments = D.instruments(market=self.market)
        
        # 获取流通市值
        df = D.features(
            instruments,
            fields=['$circ_mv'],
            start_time=calc_date,
            end_time=calc_date
        )
        
        if df.empty:
            raise ValueError(f'无法获取基准成分股数据: {calc_date}')
        if isinstance(df.index, pd.MultiIndex):
            df = df.droplevel(level='datetime')
        # 计算权重
        circ_mv = df['$circ_mv'].dropna()
        weights = circ_mv / circ_mv.sum()
        weights.name = 'weight'
        
        logger.info(f'基准权重: {len(weights)}只股票, 日期={calc_date}')
        return weights
    
    def load_current_position(
        self,
        calc_date: str,
        position_input: Union[str, Dict, pd.Series] = 'zero',
        instruments: Optional[pd.Index] = None
    ) -> pd.Series:
        """加载当前持仓
        
        Args:
            calc_date: 计算日期
            position_input: 持仓输入，支持：
                - 'zero': 零持仓
                - 'mysql': 从MySQL读取
                - dict: {instrument: weight}
                - pd.Series: 直接使用
                - str(CSV路径): 从CSV读取
            instruments: 股票代码索引（用于对齐）
            
        Returns:
            Series(index=instrument, name='weight')
        """
        if position_input == 'zero':
            # 零持仓
            if instruments is None:
                raise ValueError('零持仓模式需要提供instruments参数')
            position = pd.Series(0.0, index=instruments, name='weight')
            logger.info('当前持仓: 零持仓')
            
        elif position_input == 'mysql':
            # 从MySQL读取（TODO: 实现数据库读取）
            raise NotImplementedError('MySQL持仓读取尚未实现')
            
        elif isinstance(position_input, pd.Series):
            # 直接使用Series
            position = position_input.copy()
            position.name = 'weight'
            logger.info(f'当前持仓: {len(position)}只股票（Series输入）')
            
        elif isinstance(position_input, dict):
            # 字典转Series
            position = pd.Series(position_input, name='weight')
            position.index.name = 'instrument'
            logger.info(f'当前持仓: {len(position)}只股票（字典输入）')
            
        elif isinstance(position_input, str) and position_input.endswith('.csv'):
            # 从CSV读取
            df = pd.read_csv(position_input)
            if 'instrument' in df.columns and 'weight' in df.columns:
                position = df.set_index('instrument')['weight']
            else:
                position = df.iloc[:, 1]
                position.index = df.iloc[:, 0]
            position.name = 'weight'
            logger.info(f'当前持仓: {len(position)}只股票（CSV输入）')
            
        else:
            raise ValueError(f'不支持的持仓输入类型: {type(position_input)}')
        
        return position
    
    def load_stock_prices(
        self,
        instruments: List[str],
        calc_date: str
    ) -> pd.Series:
        """加载股票价格
        
        Args:
            instruments: 股票代码列表
            calc_date: 计算日期
            
        Returns:
            Series(index=instrument, name='price')
        """
        from qlib.data import D
        
        df = D.features(
            instruments,
            fields=['$close'],
            start_time=calc_date,
            end_time=calc_date
        )

        if df.empty:
            raise ValueError(f'无法获取股票价格数据: {calc_date}')
        df = df.droplevel(level='datetime')
        prices = df['$close'].dropna()
        prices.name = 'price'
        
        logger.info(f'加载股价: {len(prices)}只股票')

        return prices
    
    def align_all_data(
        self,
        calc_date: str,
        position_input: Union[str, Dict, pd.Series] = 'zero'
    ) -> Dict:
        """对齐所有数据，返回完整数据集
        
        Args:
            calc_date: 计算日期
            position_input: 当前持仓输入
            
        Returns:
            {
                'instruments': Index, 股票代码
                'alpha': Series,
                'exposure': DataFrame,
                'factor_cov': DataFrame,
                'specific_risk': Series,
                'benchmark_weights': Series,
                'current_position': Series,
                'prices': Series
            }
        """
        logger.info('=' * 50)
        logger.info('开始数据加载与对齐...')
        
        # 1. 加载Alpha
        alpha = self.load_alpha(calc_date)
        
        # 2. 加载风险模型（使用相同的calc_date）
        risk_model = self.load_risk_model(calc_date)
        exposure = risk_model['exposure']
        factor_cov = risk_model['factor_cov']
        specific_risk = risk_model['specific_risk']
        
        # 3. 加载基准权重（使用相同的calc_date）
        benchmark_weights = self.load_benchmark_weights(calc_date)

        # 4. 计算共同股票（取交集）
        common_instruments = (
            alpha.index
            .intersection(exposure.index)
            .intersection(specific_risk.index)
            .intersection(benchmark_weights.index)
        )
        common_instruments = common_instruments.sort_values()
        
        logger.info(f'数据对齐: 共{len(common_instruments)}只股票')
        
        # 5. 对齐各数据
        aligned_alpha = alpha.reindex(common_instruments)
        aligned_exposure = exposure.reindex(common_instruments)
        aligned_factor_cov = factor_cov
        aligned_specific_risk = specific_risk.reindex(common_instruments)
        aligned_benchmark = benchmark_weights.reindex(common_instruments, fill_value=0.0)
        
        # 6. 加载当前持仓并对其
        current_position = self.load_current_position(
            calc_date, position_input, common_instruments
        )
        aligned_position = current_position.reindex(common_instruments, fill_value=0.0)
        
        # 7. 加载股价
        prices = self.load_stock_prices(common_instruments.tolist(), calc_date)
        aligned_prices = prices.reindex(common_instruments)

        # 检查缺失值
        nan_check = {
            'alpha': aligned_alpha.isna().sum(),
            'exposure': aligned_exposure.isna().sum().sum(),
            'specific_risk': aligned_specific_risk.isna().sum(),
            'prices': aligned_prices.isna().sum()
        }
        logger.info(f'缺失值检查: {nan_check}')
        
        logger.info('数据加载与对齐完成')
        logger.info('=' * 50)

        align_data = {
            'instruments': common_instruments,
            'alpha': aligned_alpha,
            'exposure': aligned_exposure,
            'factor_cov': aligned_factor_cov,
            'specific_risk': aligned_specific_risk,
            'benchmark_weights': aligned_benchmark,
            'current_position': aligned_position,
            'prices': aligned_prices,
            'calc_date': risk_model['calc_date']
        }
        # logger.info(align_data)
        return align_data
