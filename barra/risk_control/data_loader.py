"""
数据加载模块 - 集成qlib框架
"""
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from qlib.data import D
from typing import List, Optional, Tuple, Union
from datetime import datetime

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
from utils import LoggerFactory

logger = LoggerFactory.get_logger(__name__)

from .config import QLIB_FIELDS, FIELD_GROUPS


class DataLoader:
    """数据加载器"""
    
    def __init__(self, market: str = 'csi300'):
        """
        初始化数据加载器
        
        Args:
            market: 市场代码，默认'csi300'
        """
        self.market = market
        
    def get_instruments(self, start_time: str, end_time: str) -> List[str]:
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
        return instruments
    
    def load_returns(self, instruments: Union[List[str], str], start_time: str, end_time: str) -> pd.DataFrame:
        """
        加载股票收益率数据
        
        Args:
            instruments: 股票列表
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            DataFrame, index=(instrument, datetime), columns=['return']
        """
        if isinstance(instruments, str):
            instruments = [instruments]
        fields = ['$change']
        df = D.features(instruments, fields, start_time=start_time, end_time=end_time)
        df.columns = ['return']
        return df
    
    def load_market_cap(self, instruments: List[str], start_time: str, end_time: str) -> pd.DataFrame:
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
    
    def load_industry(self, instruments: List[str], start_time: str, end_time: str) -> pd.DataFrame:
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
    
    def load_fields_data(self, instruments: List[str], start_time: str, end_time: str,
                         extend_start: int = 0, extend_freq: str = 'Y',
                         batch_size: int = 100) -> pd.DataFrame:
        """
        加载原始字段数据（用于CNE6因子计算）

        采用分批加载策略，避免一次性加载过多数据导致内存溢出或超时。
        支持时间延长：因子计算需回溯历史数据时，自动将 start_time 前移。

        Args:
            instruments: 股票列表
            start_time: 开始时间
            end_time: 结束时间
            extend_start: 时间前移量（默认0，不前移）
            extend_freq: 前移单位，'Y'=年, 'D'=日（默认'Y'）
            batch_size: 每批加载的股票数量

        Returns:
            DataFrame, 包含所有必要的原始字段
        """
        import time

        # 计算扩展后的 start_time
        if extend_start > 0:
            if extend_freq == 'Y':
                offset = pd.DateOffset(years=extend_start)
            elif extend_freq == 'D':
                offset = pd.DateOffset(days=extend_start)
            else:
                raise ValueError(f'不支持的频率: {extend_freq}')
            actual_start = (pd.Timestamp(start_time) - offset).strftime('%Y-%m-%d')
            logger.info(f'时间延长: {start_time} → {actual_start} '
                        f'(前移{extend_start}{extend_freq})')
        else:
            actual_start = start_time

        logger.info(f'采用分批加载策略，共 {len(instruments)} 只股票，分 {len(FIELD_GROUPS)} 组字段加载')
        
        all_data = []
        
        # 按股票分批加载
        for batch_idx in range(0, len(instruments), batch_size):
            batch_instruments = instruments[batch_idx:batch_idx + batch_size]
            batch_num = batch_idx // batch_size + 1
            total_batches = (len(instruments) + batch_size - 1) // batch_size
            
            logger.info(f'加载第 {batch_num}/{total_batches} 批股票 ({len(batch_instruments)} 只)...')
            
            batch_data = None
            
            # 按字段组加载
            for group_idx, group in enumerate(FIELD_GROUPS):
                group_start = time.time()
                try:
                    df = D.features(batch_instruments, group['fields'],
                                  start_time=actual_start, end_time=end_time)
                    
                    if batch_data is None:
                        batch_data = df
                    else:
                        # 合并数据 - 按索引join
                        batch_data = batch_data.join(df, how='outer')
                    
                    logger.info(f'  组 {group_idx+1}/{len(FIELD_GROUPS)} ({group["name"]}): '
                                f'{df.shape[1]} 字段, {time.time()-group_start:.2f}秒')
                    
                except Exception as e:
                    err_msg = f'组 {group_idx+1} ({group["name"]}) 加载失败: {e}'
                    logger.error(err_msg)
                    raise Exception(err_msg)
            
            if batch_data is not None:
                all_data.append(batch_data)
                logger.info(f'第 {batch_num} 批完成: {batch_data.shape}')
        
        # 合并所有批次的数据
        if len(all_data) == 0:
            raise ValueError('没有成功加载任何数据')
        
        logger.info(f'合并 {len(all_data)} 批数据...')
        final_df = pd.concat(all_data, axis=0)
        logger.info(f'最终数据: {final_df.shape}')
        
        return final_df
    
    def get_trade_dates(self, start_time: str, end_time: str, frequency: str = 'day') -> List[str]:
        """
        获取交易日列表
        
        Args:
            start_time: 开始时间
            end_time: 结束时间
            frequency: 频率，'day'或'month'
            
        Returns:
            交易日列表
        """
        from qlib.data import D
        dates = D.calendar(start_time=start_time, end_time=end_time, freq=frequency)
        return [str(d) for d in dates]
