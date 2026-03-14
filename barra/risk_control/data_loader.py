"""
数据加载模块 - 集成qlib框架
"""
import pandas as pd
import numpy as np
from qlib.data import D
from typing import List, Optional, Tuple
from datetime import datetime

from .config import QLIB_FIELDS, INDUSTRY_MAPPING


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
    
    def load_returns(self, instruments: List[str], start_time: str, end_time: str) -> pd.DataFrame:
        """
        加载股票收益率数据
        
        Args:
            instruments: 股票列表
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            DataFrame, index=(instrument, datetime), columns=['return']
        """
        fields = [QLIB_FIELDS['change']]
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
        fields = [QLIB_FIELDS['circ_mv'], QLIB_FIELDS['total_mv']]
        df = D.features(instruments, fields, start_time=start_time, end_time=end_time)
        df.columns = ['circ_mv', 'total_mv']
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
        fields = [QLIB_FIELDS['industry']]
        df = D.features(instruments, fields, start_time=start_time, end_time=end_time)
        df.columns = ['industry_code']
        # 将行业代码转为字符串
        df['industry_code'] = df['industry_code'].astype(str).str.replace('.0', '', regex=False)
        return df
    
    def load_factor_data(self, instruments: List[str], start_time: str, end_time: str) -> pd.DataFrame:
        """
        加载因子原始数据（用于CNE6因子计算）
        
        Args:
            instruments: 股票列表
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            DataFrame, 包含所有必要的原始字段
        """
        # 基础交易数据
        fields = [
            '$ind_one',
            '$change',
            '$close',
            '$circ_mv',
            '$total_mv',
            '$total_share',
            '$amount',
            # 资产负债表
            'P($$oth_eqt_tools_p_shr_q)',
            'P($$total_ncl_q)',
            'P($$total_hldr_eqy_exc_min_int_q)',
            'P($$total_assets_q)',
            'P($$total_liab_q)',
            'P($$money_cap_q)',
            # 利润表
            'P($$revenue_q)',
            'P($$n_income_attr_p_q)',
            'P($$oper_cost_q)',
            'P($$basic_eps_q)',
            'P($$ebit_q)',
            # 现金流量表
            'P($$n_cashflow_act_q)',
            'P($$depr_fa_coga_dpba_q)',
            'P($$amort_intang_assets_q)',
            'P($$lt_amort_deferred_exp_q)',
            'P($$c_pay_acq_const_fiolta_q)',
            # 借款相关
            'P($$st_borr_q)',
            'P($$lt_borr_q)',
            'P($$non_cur_liab_due_1y_q)',
            'P($$bond_payable_q)',
            # TTM数据
            'PTTM($$revenue_q)',
            'PTTM($$n_income_attr_p_q)',
            'PTTM($$n_cashflow_act_q)',
        ]
        
        df = D.features(instruments, fields, start_time=start_time, end_time=end_time)
        return df
    
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
