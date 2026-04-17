"""
输出管理模块 - 风险指标CSV + MySQL输出
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config import OUTPUT_CONFIG
from utils import LoggerFactory
from utils.db_mysql import MySQLDB

logger = LoggerFactory.get_logger(__name__)


class RiskOutputManager:
    """风险指标输出管理器"""
    
    def __init__(self, output_dir: str = 'output'):
        """
        初始化输出管理器
        
        Args:
            output_dir: 输出目录路径
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def save_stock_risk(self, mcar: pd.Series, rcar: pd.Series, 
                       calc_date: str, filename: Optional[str] = None) -> str:
        """
        保存股票风险指标到CSV
        
        Args:
            mcar: 主动风险边际贡献，Series index=instrument
            rcar: 主动风险贡献，Series index=instrument
            calc_date: 计算日期
            filename: 自定义文件名，默认使用stock_risk_YYYYMMDD.csv
            
        Returns:
            保存的文件路径
        """
        # 合并数据
        df = pd.DataFrame({
            'instrument': mcar.index,
            'mcar': mcar.values,
            'rcar': rcar.values,
            'calc_date': calc_date
        })
        
        # 按instrument升序排序
        df = df.sort_values('instrument').reset_index(drop=True)
        
        # 格式化float精度
        precision = OUTPUT_CONFIG['float_precision']
        df['mcar'] = df['mcar'].round(precision)
        df['rcar'] = df['rcar'].round(precision)
        
        # 生成文件名
        if filename is None:
            filename = OUTPUT_CONFIG['stock_risk_filename'].format(date=calc_date.replace('-', ''))
        
        filepath = self.output_dir / filename
        
        # 保存到CSV
        df.to_csv(filepath, index=False, encoding=OUTPUT_CONFIG['encoding'])
        
        return str(filepath)
    
    def save_factor_risk(self, fmcar: pd.Series, frcar: pd.Series,
                        factor_types: pd.Series, calc_date: str,
                        filename: Optional[str] = None) -> str:
        """
        保存因子风险指标到CSV
        
        Args:
            fmcar: 因子主动风险边际贡献，Series index=factor_name
            frcar: 因子主动风险贡献，Series index=factor_name
            factor_types: 因子类型，Series index=factor_name
            calc_date: 计算日期
            filename: 自定义文件名，默认使用factor_risk_YYYYMMDD.csv
            
        Returns:
            保存的文件路径
        """
        # 合并数据
        df = pd.DataFrame({
            'factor_name': fmcar.index,
            'fmcar': fmcar.values,
            'frcar': frcar.values,
            'factor_type': factor_types.reindex(fmcar.index).values,
            'calc_date': calc_date
        })
        
        # 按factor_type分组，组内按factor_name升序排序
        df = df.sort_values(['factor_type', 'factor_name']).reset_index(drop=True)
        
        # 格式化float精度
        precision = OUTPUT_CONFIG['float_precision']
        df['fmcar'] = df['fmcar'].round(precision)
        df['frcar'] = df['frcar'].round(precision)
        
        # 生成文件名
        if filename is None:
            filename = OUTPUT_CONFIG['factor_risk_filename'].format(date=calc_date.replace('-', ''))
        
        filepath = self.output_dir / filename
        
        # 保存到CSV
        df.to_csv(filepath, index=False, encoding=OUTPUT_CONFIG['encoding'])
        
        return str(filepath)
    
    def load_stock_risk(self, calc_date: str) -> Optional[pd.DataFrame]:
        """
        加载股票风险指标
        
        Args:
            calc_date: 计算日期
            
        Returns:
            DataFrame或None（文件不存在）
        """
        filename = OUTPUT_CONFIG['stock_risk_filename'].format(date=calc_date.replace('-', ''))
        filepath = self.output_dir / filename
        
        if filepath.exists():
            return pd.read_csv(filepath, encoding=OUTPUT_CONFIG['encoding'])
        else:
            return None
    
    def load_factor_risk(self, calc_date: str) -> Optional[pd.DataFrame]:
        """
        加载因子风险指标
        
        Args:
            calc_date: 计算日期
            
        Returns:
            DataFrame或None（文件不存在）
        """
        filename = OUTPUT_CONFIG['factor_risk_filename'].format(date=calc_date.replace('-', ''))
        filepath = self.output_dir / filename
        
        if filepath.exists():
            return pd.read_csv(filepath, encoding=OUTPUT_CONFIG['encoding'])
        else:
            return None

    def save_data(self, data: pd.DataFrame, path: str, type: str = 'csv'):
        '''
        将数据保存到指定位置

        Parameters
        ----------
        data: 数据对象
        path: 保存路径
        type: 保存的数据类型. csv/parquet
        '''
        logger.info('Saving data to {}...'.format(path))
        path = Path(os.path.join(self.output_dir, path))
        os.makedirs(path.parent, exist_ok=True)
        if type == 'csv':
            data.to_csv(path)
        elif type == 'parquet':
            data.to_parquet(path)
        else:
            raise ValueError("Invalid type. Supported types: 'csv', 'parquet'")
        logger.info('Data saved success ')

    def load_data(self, path: str, type: str = 'csv'):
        '''
        从指定位置加载数据

        Parameters
        ----------
        path: 加载路径
        type: 数据类型. csv/parquet

        Returns
        -------
        DataFrame或None（文件不存在）
        '''
        logger.info('Loading data from {}...'.format(path))
        path = Path(os.path.join(self.output_dir, path))

        if not path.exists():
            logger.warning('File not found: {}'.format(path))
            return None

        if type == 'csv':
            data = pd.read_csv(path, encoding=OUTPUT_CONFIG['encoding'])
        elif type == 'parquet':
            data = pd.read_parquet(path)
        else:
            raise ValueError("Invalid type. Supported types: 'csv', 'parquet'")

        logger.info('Data loaded success, shape: {}'.format(data.shape))
        return data

    def save_to_mysql(self, results: dict, calc_date: str,
                      factor_types: pd.Series,
                      portfolio_name: str = 'default') -> None:
        """
        将风险指标写入MySQL（factor_risk + portfolio_risk）

        Args:
            results: analyze_risk 返回的字典，包含 mcar/rcar/fmcar/frcar
            calc_date: 计算日期
            factor_types: 因子类型映射 Series
            portfolio_name: 组合名称
        """
        precision = OUTPUT_CONFIG['float_precision']

        # 1. factor_risk 表
        fmcar = results['fmcar']
        frcar = results['frcar']
        factor_data = [
            {
                'day': calc_date,
                'name': name,
                'type': str(factor_types.get(name, '')),
                'FMCAR': round(float(fmcar[name]), precision),
                'FRCAR': round(float(frcar[name]), precision),
            }
            for name in fmcar.index
        ]
        factor_sql = (
            'INSERT INTO factor_risk (day, name, type, FMCAR, FRCAR) '
            'VALUES (%(day)s, %(name)s, %(type)s, %(FMCAR)s, %(FRCAR)s) '
            'ON DUPLICATE KEY UPDATE '
            'FMCAR=VALUES(FMCAR), FRCAR=VALUES(FRCAR)'
        )

        # 2. portfolio_risk 表
        mcar = results['mcar']
        rcar = results['rcar']
        stock_data = [
            {
                'day': calc_date,
                'qlib_code': instrument,
                'portfolio': portfolio_name,
                'MCAR': round(float(mcar[instrument]), precision),
                'RCAR': round(float(rcar[instrument]), precision),
            }
            for instrument in mcar.index
        ]
        stock_sql = (
            'INSERT INTO portfolio_risk (day, qlib_code, portfolio, MCAR, RCAR) '
            'VALUES (%(day)s, %(qlib_code)s, %(portfolio)s, %(MCAR)s, %(RCAR)s) '
            'ON DUPLICATE KEY UPDATE '
            'MCAR=VALUES(MCAR), RCAR=VALUES(RCAR)'
        )

        with MySQLDB() as db:
            db.executemany(factor_sql, factor_data)
            db.executemany(stock_sql, stock_data)

        logger.info(f'MySQL写入完成: factor_risk {len(factor_data)}条, '
                    f'portfolio_risk {len(stock_data)}条')