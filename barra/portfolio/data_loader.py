"""
投资组合优化数据加载模块
"""
from pathlib import Path
from typing import Optional, Union, List, Dict

import pandas as pd

from barra.portfolio.config import DATA_PATHS
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
        market: str = 'csi300',
        risk_output_dir: Optional[str] = None,
        portfolio_name: str = 'default',
    ):
        """初始化数据加载器

        Args:
            market: 市场代码，如 'csi300'
            risk_output_dir: 风险模型输出目录
            portfolio_name: 组合名称
        """
        self.market = market
        self.risk_output_dir = Path(risk_output_dir)
        self.portfolio_name = portfolio_name

    def load_alpha(self, calc_date: str) -> pd.Series:
        """从MySQL加载Alpha预测值

        Args:
            calc_date: 计算日期，如 '2026-03-28'

        Returns:
            Series(index=instrument, name='alpha')
        """
        from utils import sql_engine

        engine = sql_engine()
        sql = """
            SELECT qlib_code AS instrument, alpha
            FROM alpha
            WHERE day = %(calc_date)s
        """
        df = pd.read_sql(sql, engine, params={'calc_date': calc_date})

        if df.empty:
            raise FileNotFoundError(f'Alpha数据不存在: calc_date={calc_date}')

        df = df.set_index('instrument')
        alpha = df['alpha']
        alpha.name = 'alpha'

        logger.info(f'加载Alpha(MySQL): {len(alpha)}只股票, 日期={calc_date}')
        return alpha
    
    def load_risk_model(self, calc_date: str) -> Dict[str, Union[pd.DataFrame, pd.Series]]:
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
        # MultiIndex (instrument, datetime)，提取指定日期
        exposure_df = exposure_df.xs(calc_date, level='datetime')

        # 加载因子协方差矩阵
        factor_cov_path = self.risk_output_dir / DATA_PATHS['factor_cov']
        if not factor_cov_path.exists():
            raise FileNotFoundError(f'因子协方差文件不存在: {factor_cov_path}')

        factor_cov_df = pd.read_parquet(factor_cov_path)

        # 加载特异风险
        specific_risk_path = self.risk_output_dir / DATA_PATHS['specific_risk']
        if not specific_risk_path.exists():
            raise FileNotFoundError(f'特异风险文件不存在: {specific_risk_path}')

        specific_risk = pd.read_parquet(specific_risk_path).iloc[:, 0]
        specific_risk.name = 'specific_var'

        logger.info(f'加载风险模型: exposure={exposure_df.shape}, '
                   f'factor_cov={factor_cov_df.shape}, '
                   f'specific_risk={len(specific_risk)}')

        return {
            'exposure': exposure_df,
            'factor_cov': factor_cov_df,
            'specific_risk': specific_risk,
        }
    
    def load_benchmark_weights(self, calc_date: str) -> pd.Series:
        """计算基准权重（沪深300成分股流通市值 占比）
        
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
        instruments: Optional[List[str]] = None,
        calc_date: str = None
    ) -> pd.Series:
        """加载当前持仓

        Args:
            instruments: 股票代码索引
        Returns:
            Series(index=instrument, name='weight')
        """
        if instruments is None:
            raise ValueError('零持仓模式需要提供instruments参数')
        try:
            # 从MySQL读取最新持仓
            from utils import sql_engine
            engine = sql_engine()
            sql = """
                            SELECT qlib_code AS instrument, total_weight, day
                            FROM portfolio
                            WHERE portfolio = %(portfolio)s
                              AND day = (
                                  SELECT MAX(day) FROM portfolio WHERE portfolio = %(portfolio)s
                              )
                              AND day < %(calc_date)s
                        """
            df = pd.read_sql(sql, engine, params={'portfolio': self.portfolio_name, 'calc_date': calc_date})
            latest_day = df['day'].iloc[0]
            position = df.set_index('instrument')['total_weight']
            position.name = 'weight'
            logger.info(f'当前持仓(MySQL): {len(position)}只股票, '
                        f'portfolio={self.portfolio_name}, '
                        f'latest_day={latest_day}')
        except Exception as e:
            position = pd.Series(0.0, index=instruments, name='weight')
            logger.info('当前持仓: 零持仓')

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
    ) -> Dict:
        """对齐所有数据，返回完整数据集
        
        Args:
            calc_date: 计算日期
            
        Returns:
            {
                'instruments': Index,         股票代码
                'alpha': Series,              Alpha 值
                'exposure': DataFrame,        因子曝光数据。N * K
                'factor_cov': DataFrame,      因子收益率相关矩阵 K * K
                'specific_risk': Series,      特异风险矩阵 Delta N * N
                'benchmark_weights': Series,  基准权重
                'current_position': Series,   当前持仓
                'prices': Series              股票价格
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
        specific_risk = risk_model['specific_risk']  # Series
        
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
        aligned_benchmark = aligned_benchmark / aligned_benchmark.sum()    # 权重归一化
        
        # 6. 加载当前持仓并对齐
        current_position = self.load_current_position(common_instruments, calc_date)
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

        # 每个元素都独立，仅总体 instrument 索引对齐
        align_data = {
            'instruments': common_instruments,
            'alpha': aligned_alpha,
            'exposure': aligned_exposure,
            'factor_cov': aligned_factor_cov,
            'specific_risk': aligned_specific_risk,
            'benchmark_weights': aligned_benchmark,
            'current_position': aligned_position,
            'prices': aligned_prices,
            'calc_date': calc_date
        }
        # logger.info(align_data)
        return align_data
