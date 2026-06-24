"""
因子计算模块

因子暴露矩阵构建 + 因子配置 + 行业映射。
"""

from barra.factor_com.conf import (
    CNE6_STYLE_FACTORS,
    CATEGORIES_MAP,
    STYLE_FACTOR_LIST,
    FACTOR_FUNCTIONS,
    INDUSTRY_MAPPING,
    INDUSTRY_NAMES,
    INDUSTRY_CODES,
)
from barra.factor_com.exposure import CNE6IndExposure, get_industry_dummies

__all__ = [
    # 配置
    'CNE6_STYLE_FACTORS',
    'CATEGORIES_MAP',
    'STYLE_FACTOR_LIST',
    'FACTOR_FUNCTIONS',
    'INDUSTRY_MAPPING',
    'INDUSTRY_NAMES',
    'INDUSTRY_CODES',
    # 因子暴露
    'CNE6IndExposure',
    'get_industry_dummies',
]
