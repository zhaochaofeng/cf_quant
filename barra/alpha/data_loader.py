"""
数据加载模块 - 信号、残差、行业市值数据
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

from .config import SIGNAL_TABLE, RESIDUALS_PATH
from utils import sql_engine, LoggerFactory

logger = LoggerFactory.get_logger(__name__)


class AlphaDataLoader:
    """Alpha预测数据加载器

    加载信号(MySQL)、残差收益率(parquet)、行业市值(Qlib)
    """

    def __init__(self, market: str = 'csi300'):
        """初始化

        Args:
            market: 市场代码
        """
        self.market = market
        self._qlib_loader = None

    @property
    def qlib_loader(self):
        """延迟初始化risk_control DataLoader"""
        if self._qlib_loader is None:
            from barra.risk_control.data_loader import DataLoader
            self._qlib_loader = DataLoader(market=self.market)
        return self._qlib_loader

    def load_signal(self, start_time: str, end_time: str) -> pd.DataFrame:
        """从MySQL加载原始预测信号

        Args:
            start_time: 开始日期，如 '2023-01-01'
            end_time: 结束日期，如 '2026-03-06'

        Returns:
            MultiIndex(instrument, datetime), column='g'
        """
        engine = sql_engine()
        sql = (
            f"SELECT qlib_code AS instrument, day AS datetime, score AS g "
            f"FROM {SIGNAL_TABLE} "
            f"WHERE day >= '{start_time}' AND day <= '{end_time}'"
        )
        df = pd.read_sql(sql, engine)
        if df.empty:
            raise ValueError(f'{SIGNAL_TABLE} 在 [{start_time}, {end_time}] 无数据')

        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index(['instrument', 'datetime'])
        # 去除重复条目（取最后一条）
        dup_count = df.index.duplicated().sum()
        if dup_count > 0:
            logger.warning(f'信号数据存在 {dup_count} 条重复，已去重')
            df = df[~df.index.duplicated(keep='last')]
        df = df.sort_index()
        logger.info(f'信号数据加载完成: {df.shape}')
        return df

    def load_residuals(self, residuals_path: Optional[str] = None) -> pd.DataFrame:
        """加载残差收益率

        Args:
            residuals_path: parquet文件路径，默认使用config中的路径

        Returns:
            MultiIndex(instrument, datetime), column='residual'
        """
        path = Path(residuals_path or RESIDUALS_PATH)
        if not path.exists():
            raise FileNotFoundError(f'残差文件不存在: {path}')

        df = pd.read_parquet(path)
        # 确保索引格式正确
        if not isinstance(df.index, pd.MultiIndex):
            raise ValueError('残差数据索引需为 MultiIndex(instrument, datetime)')

        df = df.sort_index()
        logger.info(f'残差数据加载完成: {df.shape}')
        return df

    def load_industry_and_market_cap(
        self, instruments: list[str], start_time: str, end_time: str
    ) -> pd.DataFrame:
        """加载行业和流通市值数据

        Args:
            instruments: 股票列表
            start_time: 开始日期
            end_time: 结束日期

        Returns:
            MultiIndex(instrument, datetime), columns=['industry_code', 'circ_mv']
        """
        industry_df = self.qlib_loader.load_industry(instruments, start_time, end_time)
        market_cap_df = self.qlib_loader.load_market_cap(instruments, start_time, end_time)

        # 合并，只取circ_mv
        df = industry_df.join(market_cap_df[['circ_mv']], how='outer')
        logger.info(f'行业市值数据加载完成: {df.shape}')
        return df

    def get_trade_dates(self, start_time: str, end_time: str) -> list[str]:
        """获取交易日列表"""
        return self.qlib_loader.get_trade_dates(start_time, end_time)

    def get_instruments(self, start_time: str, end_time: str) -> list[str]:
        """获取股票列表"""
        return self.qlib_loader.get_instruments(start_time, end_time)
