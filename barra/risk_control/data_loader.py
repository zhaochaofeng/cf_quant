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
    
    def load_factor_data(self, instruments: List[str], start_time: str, end_time: str, 
                         batch_size: int = 100) -> pd.DataFrame:
        """
        加载因子原始数据（用于CNE6因子计算）
        
        采用分批加载策略，避免一次性加载过多数据导致内存溢出或超时
        
        Args:
            instruments: 股票列表
            start_time: 开始时间
            end_time: 结束时间
            batch_size: 每批加载的股票数量
            
        Returns:
            DataFrame, 包含所有必要的原始字段
        """
        import time
        
        # 字段分组 - 按数据类型分组加载
        field_groups = [
            # 第1组: 基础交易数据（7个字段）
            {
                'name': '基础交易数据',
                'fields': [
                    '$ind_one', '$change', '$close', '$circ_mv', 
                    '$total_mv', '$total_share', '$amount'
                ]
            },
            # 第2组: 资产负债表（6个字段）
            {
                'name': '资产负债表',
                'fields': [
                    'P($$oth_eqt_tools_p_shr_q)', 'P($$total_ncl_q)',
                    'P($$total_hldr_eqy_exc_min_int_q)', 'P($$total_assets_q)',
                    'P($$total_liab_q)', 'P($$money_cap_q)'
                ]
            },
            # 第3组: 利润表（5个字段）
            {
                'name': '利润表',
                'fields': [
                    'P($$revenue_q)', 'P($$n_income_attr_p_q)',
                    'P($$oper_cost_q)', 'P($$basic_eps_q)', 'P($$ebit_q)'
                ]
            },
            # 第4组: 现金流量表（5个字段）
            {
                'name': '现金流量表',
                'fields': [
                    'P($$n_cashflow_act_q)', 'P($$depr_fa_coga_dpba_q)',
                    'P($$amort_intang_assets_q)', 'P($$lt_amort_deferred_exp_q)',
                    'P($$c_pay_acq_const_fiolta_q)'
                ]
            },
            # 第5组: 借款相关（4个字段）
            {
                'name': '借款相关',
                'fields': [
                    'P($$st_borr_q)', 'P($$lt_borr_q)',
                    'P($$non_cur_liab_due_1y_q)', 'P($$bond_payable_q)'
                ]
            },
            # 第6组: TTM数据（3个字段）
            {
                'name': 'TTM数据',
                'fields': [
                    'PTTM($$revenue_q)', 'PTTM($$n_income_attr_p_q)',
                    'PTTM($$n_cashflow_act_q)'
                ]
            },
        ]
        
        print(f"   采用分批加载策略，共 {len(instruments)} 只股票，分 {len(field_groups)} 组字段加载")
        
        all_data = []
        
        # 按股票分批加载
        for batch_idx in range(0, len(instruments), batch_size):
            batch_instruments = instruments[batch_idx:batch_idx + batch_size]
            batch_num = batch_idx // batch_size + 1
            total_batches = (len(instruments) + batch_size - 1) // batch_size
            
            print(f"   加载第 {batch_num}/{total_batches} 批股票 ({len(batch_instruments)} 只)...")
            
            batch_data = None
            
            # 按字段组加载
            for group_idx, group in enumerate(field_groups):
                group_start = time.time()
                try:
                    df = D.features(batch_instruments, group['fields'], 
                                  start_time=start_time, end_time=end_time)
                    
                    if batch_data is None:
                        batch_data = df
                    else:
                        # 合并数据 - 按索引join
                        batch_data = batch_data.join(df, how='outer')
                    
                    print(f"     组 {group_idx+1}/{len(field_groups)} ({group['name']}): "
                          f"{df.shape[1]} 字段, {time.time()-group_start:.2f}秒")
                    
                except Exception as e:
                    print(f"     组 {group_idx+1} ({group['name']}) 加载失败: {e}")
                    # 继续加载其他组
                    continue
            
            if batch_data is not None:
                all_data.append(batch_data)
                print(f"   第 {batch_num} 批完成: {batch_data.shape}")
        
        # 合并所有批次的数据
        if len(all_data) == 0:
            raise ValueError("没有成功加载任何数据")
        
        print(f"   合并 {len(all_data)} 批数据...")
        final_df = pd.concat(all_data, axis=0)
        print(f"   最终数据: {final_df.shape}")
        
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
