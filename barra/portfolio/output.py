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
    
    def __init__(self, output_dir: str = None):
        """初始化输出管理器
        
        Args:
            output_dir: 输出目录路径
        """
        self.output_dir = Path(output_dir or OUTPUT_CONFIG['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.float_precision = OUTPUT_CONFIG['float_precision']
        self.encoding = OUTPUT_CONFIG['encoding']
    
    def save_trade_orders(
        self,
        trade_orders: pd.DataFrame,
        calc_date: str,
        filename: Optional[str] = None
    ) -> str:
        """保存交易指令
        
        Args:
            trade_orders: 交易指令DataFrame
            calc_date: 计算日期
            filename: 自定义文件名
            
        Returns:
            保存的文件路径
        """
        # 添加日期列
        df = trade_orders.copy()
        df['calc_date'] = calc_date
        
        # 格式化浮点数
        float_cols = ['weight_change', 'amount', 'price', 'active_weight', 'total_weight']
        for col in float_cols:
            if col in df.columns:
                df[col] = df[col].round(self.float_precision)
        
        # 生成文件名
        if filename is None:
            date_str = calc_date.replace('-', '')
            filename = OUTPUT_CONFIG['trade_order_filename'].format(date=date_str)
        
        filepath = self.output_dir / filename
        
        # 保存parquet
        df.to_parquet(filepath, index=False)
        logger.info(f'交易指令已保存: {filepath}')
        
        # 同时保存CSV
        csv_path = filepath.with_suffix('.csv')
        df.to_csv(csv_path, index=False, encoding=self.encoding)
        logger.info(f'交易指令CSV已保存: {csv_path}')
        
        return str(filepath)
    
    def save_position(
        self,
        position: pd.DataFrame,
        calc_date: str,
        filename: Optional[str] = None
    ) -> str:
        """保存持仓信息
        
        Args:
            position: 持仓DataFrame
            calc_date: 计算日期
            filename: 自定义文件名
            
        Returns:
            保存的文件路径
        """
        # 添加日期列
        df = position.copy()
        df['calc_date'] = calc_date
        
        # 格式化浮点数
        float_cols = ['active_weight', 'total_weight', 'market_value', 'weight_pct']
        for col in float_cols:
            if col in df.columns:
                df[col] = df[col].round(self.float_precision)
        
        # 生成文件名
        if filename is None:
            date_str = calc_date.replace('-', '')
            filename = OUTPUT_CONFIG['position_filename'].format(date=date_str)
        
        filepath = self.output_dir / filename
        
        # 保存parquet
        df.to_parquet(filepath, index=False)
        logger.info(f'持仓信息已保存: {filepath}')
        
        # 同时保存CSV
        csv_path = filepath.with_suffix('.csv')
        df.to_csv(csv_path, index=False, encoding=self.encoding)
        logger.info(f'持仓信息CSV已保存: {csv_path}')
        
        return str(filepath)
    
    def save_optimization_log(
        self,
        instruments: pd.Index,
        alpha: pd.Series,
        mcva: np.ndarray,
        in_no_trade_zone: np.ndarray,
        v_diag: np.ndarray,
        calc_date: str,
        filename: Optional[str] = None
    ) -> str:
        """保存优化诊断日志
        
        Args:
            instruments: 股票代码索引
            alpha: Alpha预测值
            mcva: 边际贡献
            in_no_trade_zone: 是否在无交易区域
            v_diag: 协方差对角元素
            calc_date: 计算日期
            filename: 自定义文件名
            
        Returns:
            保存的文件路径
        """
        df = pd.DataFrame({
            'instrument': instruments,
            'alpha': alpha.values,
            'mcva': mcva,
            'in_no_trade_zone': in_no_trade_zone,
            'v_diag': v_diag,
            'calc_date': calc_date
        })
        
        # 格式化浮点数
        float_cols = ['alpha', 'mcva', 'v_diag']
        for col in float_cols:
            df[col] = df[col].round(self.float_precision)
        
        # 生成文件名
        if filename is None:
            date_str = calc_date.replace('-', '')
            filename = OUTPUT_CONFIG['log_filename'].format(date=date_str)
        
        filepath = self.output_dir / filename
        
        # 保存parquet
        df.to_parquet(filepath, index=False)
        logger.info(f'优化诊断日志已保存: {filepath}')
        
        return str(filepath)
    
    def save_to_mysql(
        self,
        trade_orders: pd.DataFrame,
        position: pd.DataFrame,
        calc_date: str,
        portfolio: str = 'default'
    ) -> bool:
        """保存到MySQL数据库
        
        Args:
            trade_orders: 交易指令
            position: 持仓信息
            calc_date: 计算日期
            portfolio: 组合名称
            
        Returns:
            是否成功
        """
        try:
            from utils import MySQLDB
            
            # 保存交易指令
            with MySQLDB() as db:
                # 删除已有数据
                db.execute(
                    'DELETE FROM portfolio_order WHERE day = %s AND portfolio = %s',
                    (calc_date, portfolio)
                )
                
                # 插入新数据
                for _, row in trade_orders.iterrows():
                    if row['direction'] != 'hold':
                        db.execute(
                            '''INSERT INTO portfolio_order 
                               (day, portfolio, qlib_code, direction, weight_change, amount, shares, price)
                               VALUES (%s, %s, %s, %s, %s, %s, %s, %s)''',
                            (calc_date, portfolio, row['instrument'], row['direction'],
                             row['weight_change'], row['amount'], row['shares'], row['price'])
                        )
                
                # 删除已有持仓数据
                db.execute(
                    'DELETE FROM portfolio_position WHERE day = %s AND portfolio = %s',
                    (calc_date, portfolio)
                )
                
                # 插入新持仓数据
                for _, row in position.iterrows():
                    db.execute(
                        '''INSERT INTO portfolio_position 
                           (day, portfolio, qlib_code, active_weight, total_weight, shares, market_value)
                           VALUES (%s, %s, %s, %s, %s, %s, %s)''',
                        (calc_date, portfolio, row['instrument'],
                         row['active_weight'], row['total_weight'], row['shares'], row['market_value'])
                    )
            
            logger.info(f'数据已保存到MySQL: {calc_date}, portfolio={portfolio}')
            return True
            
        except Exception as e:
            logger.warning(f'MySQL保存失败: {e}')
            return False
    
    def load_trade_orders(self, calc_date: str) -> Optional[pd.DataFrame]:
        """加载交易指令
        
        Args:
            calc_date: 计算日期
            
        Returns:
            DataFrame或None
        """
        date_str = calc_date.replace('-', '')
        filename = OUTPUT_CONFIG['trade_order_filename'].format(date=date_str)
        filepath = self.output_dir / filename
        
        if filepath.exists():
            return pd.read_parquet(filepath)
        return None
    
    def load_position(self, calc_date: str) -> Optional[pd.DataFrame]:
        """加载持仓信息
        
        Args:
            calc_date: 计算日期
            
        Returns:
            DataFrame或None
        """
        date_str = calc_date.replace('-', '')
        filename = OUTPUT_CONFIG['position_filename'].format(date=date_str)
        filepath = self.output_dir / filename
        
        if filepath.exists():
            return pd.read_parquet(filepath)
        return None
