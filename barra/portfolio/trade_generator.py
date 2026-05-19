"""
交易指令生成模块
"""
import numpy as np
import pandas as pd

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
        lot_size: int = 100
    ):
        """初始化生成器
        
        Args:
            min_trade_threshold: 最小交易阈值（权重变化）
            lot_size: 每手股数，A股为100股
        """
        params = OPTIMIZATION_PARAMS.copy()
        self.min_trade_threshold = params['min_trade_threshold']
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
