"""
输出管理模块 - Alpha预测结果保存与加载
"""
import os
from pathlib import Path
from typing import Optional

import pandas as pd

from utils import LoggerFactory

logger = LoggerFactory.get_logger(__name__)


class AlphaOutputManager:
    """Alpha预测输出管理器"""

    def __init__(self, output_dir: str = 'barra/alpha/output'):
        """初始化

        Args:
            output_dir: 输出目录路径
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_alpha(self, alpha: pd.DataFrame, calc_date: str) -> str:
        """保存每日Alpha预测到parquet

        Args:
            alpha: Alpha预测，index=instrument, column='alpha'
            calc_date: 计算日期

        Returns:
            保存的文件路径
        """
        date_str = calc_date.replace('-', '')
        filepath = self.output_dir / f'alpha_{date_str}.parquet'
        alpha.to_parquet(filepath)
        logger.info(f'Alpha保存完成: {filepath}')
        return str(filepath)

    def save_diagnostics(self, diagnostics: pd.DataFrame, calc_date: str) -> str:
        """保存诊断信息到parquet

        Args:
            diagnostics: 诊断信息（IC, case, omega等）
            calc_date: 计算日期

        Returns:
            保存的文件路径
        """
        date_str = calc_date.replace('-', '')
        diag_dir = self.output_dir / 'diagnostics'
        diag_dir.mkdir(parents=True, exist_ok=True)
        filepath = diag_dir / f'diag_{date_str}.parquet'
        diagnostics.to_parquet(filepath)
        logger.info(f'诊断信息保存完成: {filepath}')
        return str(filepath)

    def load_alpha(self, calc_date: str) -> Optional[pd.DataFrame]:
        """加载Alpha预测

        Args:
            calc_date: 计算日期

        Returns:
            DataFrame或None（文件不存在）
        """
        date_str = calc_date.replace('-', '')
        filepath = self.output_dir / f'alpha_{date_str}.parquet'
        if not filepath.exists():
            logger.warning(f'Alpha文件不存在: {filepath}')
            return None
        return pd.read_parquet(filepath)
