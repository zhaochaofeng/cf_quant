"""
交易指令生成模块
"""
import numpy as np
import pandas as pd
from typing import Dict

from barra.portfolio.config import OPTIMIZATION_PARAMS
from utils import LoggerFactory

logger = LoggerFactory.get_logger(__name__)


class TradeGenerator:
    """交易指令生成器
    
    将权重变化转换为具体的交易指令：
    - 计算交易金额
    - 计算交易股数（向下取整到100股整数倍）
    - 过滤小于阈值的交易
    """
    
    def __init__(
        self,
        min_trade_threshold: float = None,
        lot_size: int = 100
    ):
        """初始化生成器
        
        Args:
            min_trade_threshold: 最小交易阈值（权重变化）
            lot_size: 每手股数，A股为100股
        """
        params = OPTIMIZATION_PARAMS.copy()
        self.min_trade_threshold = min_trade_threshold or params['min_trade_threshold']
        self.lot_size = lot_size
    
    def generate(
        self,
        h_final: pd.Series,
        h_cur: pd.Series,
        portfolio_value: float,
        prices: pd.Series,
        w_b: pd.Series
    ) -> pd.DataFrame:
        """生成交易指令
        
        Args:
            h_final: 最终主动头寸 Series(instrument)
            h_cur: 当前主动头寸 Series(instrument)
            portfolio_value: 组合净值（元）
            prices: 股票价格 Series(instrument)
            w_b: 基准权重 Series(instrument)
            
        Returns:
            DataFrame(columns=[
                'instrument', 'direction', 'weight_change',
                'amount', 'shares', 'price', 'active_weight', 'total_weight'
            ])
        """
        logger.info('开始生成交易指令...')
        
        # 对齐索引
        instruments = h_final.index
        h_final = h_final.reindex(instruments)
        h_cur = h_cur.reindex(instruments, fill_value=0.0)
        prices = prices.reindex(instruments)
        w_b = w_b.reindex(instruments, fill_value=0.0)
        
        # 计算权重变化
        delta_h = h_final - h_cur
        delta_h.name = 'weight_change'
        
        # 计算交易金额
        amounts = np.abs(delta_h.values) * portfolio_value
        
        # 计算交易股数（向下取整到lot_size整数倍）
        shares = self._calculate_shares(amounts, prices.values)
        
        # 确定交易方向
        directions = np.where(delta_h > self.min_trade_threshold, 'buy',
                     np.where(delta_h < -self.min_trade_threshold, 'sell', 'hold'))
        
        # 计算总权重
        total_weight = w_b + h_final
        
        # 构建结果DataFrame
        result = pd.DataFrame({
            'instrument': instruments,
            'direction': directions,
            'weight_change': delta_h.values,
            'amount': amounts,
            'shares': shares,
            'price': prices.values,
            'active_weight': h_final.values,
            'total_weight': total_weight.values
        })
        
        # 过滤持仓为0且不交易的股票
        result = result[~((result['direction'] == 'hold') & 
                          (result['total_weight'] == 0))].copy()
        
        # 统计
        buy_count = (result['direction'] == 'buy').sum()
        sell_count = (result['direction'] == 'sell').sum()
        hold_count = (result['direction'] == 'hold').sum()
        
        logger.info(f'交易指令生成完成: 买入={buy_count}, 卖出={sell_count}, 持有={hold_count}')
        
        return result.reset_index(drop=True)
    
    def _calculate_shares(self, amounts: np.ndarray, prices: np.ndarray) -> np.ndarray:
        """计算交易股数（向下取整到lot_size整数倍）
        
        Args:
            amounts: 交易金额数组
            prices: 股价数组
            
        Returns:
            交易股数数组
        """
        # 避免除零
        safe_prices = np.where(prices > 0, prices, np.inf)
        
        # 计算原始股数
        raw_shares = amounts / safe_prices
        
        # 向下取整到lot_size整数倍
        shares = (raw_shares // self.lot_size) * self.lot_size
        
        # 转为整数
        shares = shares.astype(int)
        
        return shares
    
    def generate_position_summary(
        self,
        trade_orders: pd.DataFrame,
        portfolio_value: float
    ) -> pd.DataFrame:
        """生成持仓摘要
        
        Args:
            trade_orders: 交易指令DataFrame
            portfolio_value: 组合净值
            
        Returns:
            DataFrame(columns=[
                'instrument', 'active_weight', 'total_weight',
                'shares', 'market_value', 'weight_pct'
            ])
        """
        df = trade_orders.copy()
        
        # 计算市值
        df['market_value'] = df['shares'] * df['price']
        
        # 计算权重百分比
        df['weight_pct'] = df['total_weight'] * 100
        
        # 选择输出列
        result = df[['instrument', 'active_weight', 'total_weight', 
                     'shares', 'market_value', 'weight_pct']].copy()
        
        # 只保留有持仓的股票
        result = result[result['total_weight'] != 0].copy()
        
        # 按权重排序
        result = result.sort_values('total_weight', ascending=False).reset_index(drop=True)
        
        logger.info(f'持仓摘要: {len(result)}只股票')
        
        return result
