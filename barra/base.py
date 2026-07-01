'''
    Barra 模块基类
'''


from typing import List

import pandas as pd
from qlib.data import D

from config import BENCHMARK_CONFIG
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


    def load_signal(self, start_time: str, end_time: str) -> pd.DataFrame:
        """从MySQL加载原始预测信号

        Args:
            start_time: 开始日期，如 '2023-01-01'
            end_time: 结束日期，如 '2026-03-06'

        Returns:
            MultiIndex(instrument, datetime), column='g'
        """
        from utils import sql_engine
        engine = sql_engine()
        sql = (
            f"SELECT qlib_code AS instrument, day AS datetime, score AS g "
            f"FROM monitor_return_rate "
            f"WHERE day >= '{start_time}' AND day <= '{end_time}' AND model='lightgbm_alpha_csi300'"
        )
        df = pd.read_sql(sql, engine)
        if df.empty:
            raise ValueError(f'信号数据在 [{start_time}, {end_time}] 无数据')

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

    @staticmethod
    def load_rate(start_time: str, end_time: str) -> pd.Series:
        """从MySQL加载 无风险利率

                Args:
                    start_time: 开始日期，如 '2023-01-01'
                    end_time: 结束日期，如 '2026-03-06'

                Returns:
                    Index(datetime), column='rate'
                """
        from utils import sql_engine
        engine = sql_engine()
        sql = (
            f"SELECT date AS datetime, on_rate as rate "
            f"FROM shibor "
            f"WHERE date >= '{start_time}' AND date <= '{end_time}'"
        )
        df = pd.read_sql(sql, engine)
        if df.empty:
            raise ValueError(f'无风险利率数据在 [{start_time}, {end_time}] 为空')
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index(['datetime'])
        df = df['rate']   # 转化为 Series
        # 去除重复条目（取最后一条）
        dup_count = df.index.duplicated().sum()
        if dup_count > 0:
            logger.warning(f'无风险利率数据存在 {dup_count} 条重复，已去重')
            df = df[~df.index.duplicated(keep='last')]
        df = df.sort_index()
        logger.info(f'数据加载完成: {df.shape}')
        df.sort_index(inplace=True)
        df = df * 0.01   # 百分数转化为小数形式
        return df


    @staticmethod
    def load_benchmark_ret(start_time: str, end_time: str,
                           benchmark: str = BENCHMARK_CONFIG['BENCHMARK'], k: int=1) -> pd.Series:
        """从MySQL加载 基准收益率

        Args:
            start_time: 开始日期，如 '2023-01-01'
            end_time: 结束日期，如 '2026-03-06'
            benchmark: 基准代码，默认为 CSI300
            k: 收益率计算期数

        Returns:
            Index(datetime), column='bm_ret'
        """
        close = D.features([benchmark], ['$close'], start_time=start_time, end_time=end_time)
        close = close.droplevel(level='instrument')
        close = close.iloc[:,0]
        ret = close.shift(-k-1) / close.shift(-1) - 1
        ret.name = 'bm_ret'
        return ret
