'''
    Barra 模块基类
'''


import pandas as pd
import qlib
from qlib.data import D
from typing import List
from config import BENCHMARK_CONFIG, PROVIDER_URI
from utils import LoggerFactory

logger = LoggerFactory.get_logger(__name__)


class BaseDataLoader:
    """
        加载数据
    """

    def __init__(self, market: str = BENCHMARK_CONFIG['market']):
        self.market = market

    def load_instruments(self, start_time: str, end_time: str) -> List[str]:
        """
        获取股票列表

        Args:
            start_time: 开始时间
            end_time: 结束时间

        Returns:
            股票代码列表
        """
        instruments = D.instruments(market=self.market)
        instruments = D.list_instruments(
            instruments,
            start_time=start_time,
            end_time=end_time,
            as_list=True
        )
        logger.info(f'市场: {self.market}, 开始时间: {start_time}, 结束时间: {end_time}, 股票数量: {len(instruments)}')
        return instruments

    @staticmethod
    def load_industry(instruments: List[str], start_time: str, end_time: str) -> pd.DataFrame:
        """
        加载行业分类数据

        Args:
            instruments: 股票列表
            start_time: 开始时间
            end_time: 结束时间

        Returns:
            DataFrame, index=(instrument, datetime), columns=['industry_code']
        """
        fields = ['$ind_one']
        df = D.features(instruments, fields, start_time=start_time, end_time=end_time)
        df.columns = ['industry_code']
        # 将行业代码转为字符串
        df['industry_code'] = df['industry_code'].astype(str).str.replace('.0', '', regex=False)
        return df

    @staticmethod
    def load_market_cap(instruments: List[str], start_time: str, end_time: str) -> pd.DataFrame:
        """
        加载市值数据

        Args:
            instruments: 股票列表
            start_time: 开始时间
            end_time: 结束时间

        Returns:
            DataFrame, index=(instrument, datetime), columns=['circ_mv', 'total_mv']
        """
        # 流通市值、总市值
        fields = ['$circ_mv', '$total_mv']
        df = D.features(instruments, fields, start_time=start_time, end_time=end_time)
        df.columns = ['circ_mv', 'total_mv']
        df = df * 10000  # 万元转元
        return df




