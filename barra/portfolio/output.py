"""
投资组合优化输出管理模块
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict

from barra.portfolio.config import OUTPUT_CONFIG
from utils import LoggerFactory

logger = LoggerFactory.get_logger(__name__)


class PortfolioOutputManager:
    """组合优化输出管理器
    
    负责保存以下输出：
    - 交易指令文件（parquet/CSV）
    - 持仓文件（parquet/CSV）
    - 优化诊断日志（parquet）
    - MySQL数据库（可选）
    """

    def save_to_mysql(
        self,
        trade_orders: pd.DataFrame,
        calc_date: str,
        portfolio_name: str,
        total_value: float,
        current_position: pd.Series
    ):
        """保存持仓信息到MySQL数据库

        Args:
            trade_orders: 交易信息
            calc_date: 计算日期
            portfolio_name: 组合名称
            total_value: 组合总值
            current_position: 当前持仓
        Returns:
            是否成功
        """
        instrument = trade_orders['instrument']
        trade_orders.set_index('instrument', inplace=True)
        trade_shares = trade_orders['trade_shares'].copy()
        direction = trade_orders['direction']

        # 处理卖出的股票
        sell_mask = direction == 'sell'
        # 对于卖出的股票，如果卖出数量超过当前持仓，则限制为当前持仓数量
        if sell_mask.any():
            # 获取卖出股票对应的当前持仓
            sell_instruments = trade_shares[sell_mask].index
            current_hold = current_position.reindex(sell_instruments).fillna(0)

            # 限制卖出数量不超过持仓
            excess_mask = trade_shares[sell_mask] > current_hold
            if excess_mask.any():
                trade_shares.loc[sell_mask & excess_mask] = current_hold[excess_mask]

        # 将卖出数量转为负数
        trade_shares.loc[sell_mask] = -trade_shares.loc[sell_mask]

        # 新持仓
        new_position = current_position.reindex(instrument).fillna(0) + trade_shares
        hold_value = new_position * trade_orders['price']
        cash = total_value - hold_value.sum().item()
        try:
            from utils import MySQLDB

            sql = '''INSERT INTO portfolio
                     (day, portfolio, qlib_code, active_weight, total_weight,
                      hold_shares, trade_shares, direction, price, cash)
                     VALUES (%(day)s, %(portfolio)s, %(qlib_code)s,
                             %(active_weight)s, %(total_weight)s,
                             %(hold_shares)s, %(trade_shares)s, 
                             %(direction)s, %(price)s, %(cash)s
                             )
                     ON DUPLICATE KEY UPDATE
                     active_weight = VALUES(active_weight),
                     total_weight = VALUES(total_weight),
                     hold_shares = VALUES(hold_shares),
                     trade_shares = VALUES(trade_shares),
                     direction = VALUES(direction),
                     price = VALUES(price),
                     cash = VALUES(cash)
                     '''

            params = []
            trade_orders = trade_orders.reset_index()
            for _, row in trade_orders.iterrows():
                params.append({
                    'day': calc_date,
                    'portfolio': portfolio_name,
                    'qlib_code': row['instrument'],
                    'active_weight': float(row['active_weight']),
                    'total_weight': float(row['total_weight']),
                    'hold_shares': int(new_position[row['instrument']]),
                    'trade_shares': int(trade_shares[row['instrument']]),
                    'direction': row['direction'],
                    'price': float(row['price']),
                    'cash': float(cash)
                })

            with MySQLDB() as db:
                db.executemany(sql, params)

            logger.info(f'持仓数据已保存到MySQL: {calc_date}, portfolio={portfolio_name}, '
                       f'共{len(params)}条')

        except Exception as e:
            err_msg = f'MySQL保存失败: {e}'
            logger.error(err_msg)
            raise Exception(err_msg)

    def save_factor_alpha_to_mysql(
        self,
        factor_alpha: pd.DataFrame,
        calc_date: str,
    ):
        """保存因子阿尔法到MySQL

        Args:
            factor_alpha: DataFrame(index=因子名, columns=['alpha_F'])
            calc_date: 计算日期 YYYY-MM-DD
        """

        sql = '''
            INSERT INTO factor_alpha (day, name, alpha_F)
            VALUES (%(day)s, %(name)s, %(alpha_F)s)
            ON DUPLICATE KEY UPDATE alpha_F = VALUES(alpha_F)
        '''

        params = []
        for name, row in factor_alpha.iterrows():
            params.append({
                'day': calc_date,
                'name': name,
                'alpha_F': row['alpha_F'],
            })

        try:
            from utils import MySQLDB
            with MySQLDB() as db:
                db.executemany(sql, params)

            logger.info(f'因子阿尔法已保存到MySQL: calc_date={calc_date}, '
                       f'共{len(params)}个因子')

        except Exception as e:
            err_msg = f'因子阿尔法MySQL保存失败: {e}'
            logger.error(err_msg)
            raise Exception(err_msg)
