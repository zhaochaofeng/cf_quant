"""
Barra CNE6 风险模型 - 每日运行脚本
加载月频模型数据(F, Delta, X)，计算日频风险指标(MCAR, RCAR, FMCAR, FRCAR)
"""
import sys
import argparse
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import qlib

from barra.risk_control.config import CNE6_STYLE_FACTORS, INDUSTRY_MAPPING
from barra.risk_control.output import RiskOutputManager
from barra.risk_control.portfolio import PortfolioManager
from barra.risk_control.risk_attribution import RiskAttributionAnalyzer
from utils import LoggerFactory, MySQLDB, send_email
import traceback

logger = LoggerFactory.get_logger(__name__)


def init_qlib():
    """初始化qlib，注册PTTM自定义操作符"""
    from utils.qlib_ops import PTTM
    qlib.init(
        provider_uri='~/.qlib/qlib_data/custom_data_hfq',
        custom_ops=[PTTM]
    )
    logger.info('Qlib初始化完成')


def get_factor_types() -> pd.Series:
    """获取因子类型映射（风格因子 + 行业因子）"""
    category_map = {
        'size': '规模', 'volatility': '波动率', 'liquidity': '流动性',
        'momentum': '动量', 'quality_leverage': '质量-杠杆',
        'quality_earn_vol': '质量-盈利波动', 'quality_earn_quality': '质量-盈利质量',
        'quality_profit': '质量-盈利能力', 'quality_invest': '质量-投资质量',
        'value': '价值', 'growth': '成长',
    }
    factor_types = {}
    for category, factors in CNE6_STYLE_FACTORS.items():
        for f in factors:
            factor_types[f] = category_map.get(category, category)
    for code, name in INDUSTRY_MAPPING.items():
        factor_types[name] = '行业'
    return pd.Series(factor_types)


def save_to_mysql(results: dict, calc_date: str,
                  factor_types: pd.Series,
                  portfolio_name: str = 'random') -> None:
    """将风险指标写入MySQL

    Args:
        results: 风险分析结果字典（含 mcar/rcar/fmcar/frcar）
        calc_date: 计算日期
        factor_types: 因子类型映射
        portfolio_name: 持仓组合名称
    """
    with MySQLDB() as db:
        # 因子风险指标
        factor_data = [
            {
                'day': calc_date,
                'name': name,
                'type': factor_types.get(name, ''),
                'FMCAR': round(float(results['fmcar'][name]), 6),
                'FRCAR': round(float(results['frcar'][name]), 6),
            }
            for name in results['fmcar'].index
        ]
        factor_sql = (
            'INSERT INTO factor_risk (day, name, type, FMCAR, FRCAR) '
            'VALUES (%(day)s, %(name)s, %(type)s, %(FMCAR)s, %(FRCAR)s) '
            'ON DUPLICATE KEY UPDATE '
            'type=VALUES(type), FMCAR=VALUES(FMCAR), FRCAR=VALUES(FRCAR)'
        )
        db.executemany(factor_sql, factor_data)

        # 股票风险指标
        stock_data = [
            {
                'day': calc_date,
                'qlib_code': instrument,
                'portfolio': portfolio_name,
                'MCAR': round(float(results['mcar'][instrument]), 6),
                'RCAR': round(float(results['rcar'][instrument]), 6),
            }
            for instrument in results['mcar'].index
        ]
        stock_sql = (
            'INSERT INTO portfolio_risk (day, qlib_code, portfolio, MCAR, RCAR) '
            'VALUES (%(day)s, %(qlib_code)s, %(portfolio)s, %(MCAR)s, %(RCAR)s) '
            'ON DUPLICATE KEY UPDATE '
            'MCAR=VALUES(MCAR), RCAR=VALUES(RCAR)'
        )
        db.executemany(stock_sql, stock_data)

    logger.info(f'MySQL写入完成: factor_risk {len(factor_data)}条, '
                f'portfolio_risk {len(stock_data)}条')


def run_daily_risk(output_dir: str = 'output',
                   portfolio_input: str = 'random') -> dict:
    """
    运行每日风险计算

    流程:
        1. 加载 F(因子协方差), Delta(特异风险), X(因子暴露)
        2. 提取 X 最新日期数据 X(t)
        3. 对齐 F, Delta, X(t) 的因子/股票索引
        4. 计算资产协方差矩阵 V = X(t)·F·X(t)^T + Delta
        5. 获取组合/基准权重，计算 MCAR/RCAR/FMCAR/FRCAR
        6. 输出风险指标 CSV + MySQL

    Args:
        output_dir: 输出目录（与 run_monthly 一致）
        portfolio_input: 投资组合输入 ('random' 或 CSV路径)

    Returns:
        dict: 风险分析结果
    """
    output_mgr = RiskOutputManager(output_dir=output_dir)

    # 1. 加载模型数据
    logger.info('=' * 60)
    logger.info('1. 加载模型数据...')
    F = output_mgr.load_data('model/factor_covariance.parquet', type='parquet')
    delta_df = output_mgr.load_data('model/specific_risk.parquet', type='parquet')
    X_all = output_mgr.load_data('debug/exposure_matrix.parquet', type='parquet')

    if F is None or delta_df is None or X_all is None:
        raise FileNotFoundError('模型数据文件缺失，请先运行 run_monthly.py')

    # 2. 提取 X 最新日期
    latest_date = X_all.index.get_level_values('datetime').max()
    calc_date = str(latest_date.date())
    X_t = X_all.xs(latest_date, level='datetime')

    logger.info(f'计算日期: {calc_date}')
    logger.info(f'F: {F.shape}, Delta: {delta_df.shape}, X(t): {X_t.shape}')

    # 3. 对齐因子和股票索引
    common_factors = X_t.columns.intersection(F.index)
    common_instruments = X_t.index.intersection(delta_df.index)

    F = F.loc[common_factors, common_factors]
    X_t = X_t.loc[common_instruments, common_factors]
    delta_diag = pd.Series(
        np.diag(delta_df.loc[common_instruments, common_instruments].values),
        index=common_instruments
    )

    logger.info(f'对齐后: {len(common_factors)}个因子, {len(common_instruments)}只股票')

    # 4. 计算资产协方差矩阵 V = X·F·X^T + Delta
    X_vals = X_t.values
    V_vals = X_vals @ F.values @ X_vals.T + np.diag(delta_diag.values)
    V = pd.DataFrame(V_vals, index=common_instruments, columns=common_instruments)

    logger.info(f'资产协方差矩阵 V: {V.shape}')

    # 5. 获取组合权重 & 基准权重
    portfolio_mgr = PortfolioManager(market='csi300')
    if portfolio_input == 'random':
        h_p = portfolio_mgr.generate_random_portfolio(
            calc_date, n_stocks=50, random_state=42
        )
    else:
        h_p = portfolio_mgr.load_portfolio(portfolio_input, calc_date)
    h_b = portfolio_mgr.get_benchmark_weights(calc_date)

    logger.info(f'组合: {len(h_p)}只股票, 基准: {len(h_b)}只股票')

    # 6. 风险归因分析
    analyzer = RiskAttributionAnalyzer()
    results = analyzer.analyze_risk(V, F, X_t, h_p, h_b)

    # 7. 保存结果到CSV
    logger.info('保存风险指标...')
    factor_types = get_factor_types()
    stock_file = output_mgr.save_stock_risk(
        results['mcar'], results['rcar'], calc_date
    )
    factor_file = output_mgr.save_factor_risk(
        results['fmcar'], results['frcar'], factor_types, calc_date
    )
    logger.info(f'股票风险: {stock_file}')
    logger.info(f'因子风险: {factor_file}')

    # 8. 写入MySQL
    save_to_mysql(results, calc_date, factor_types,
                  portfolio_name=portfolio_input)

    # 摘要
    logger.info('=' * 60)
    logger.info(f'计算日期: {calc_date}')
    logger.info(f'组合总风险: {results["total_risk"]:.6f}')
    logger.info(f'主动风险(跟踪误差): {results["active_risk"]:.6f}')
    logger.info(f'RCAR之和: {results["rcar_sum"]:.6f} '
                f'(应≈主动风险 {results["active_risk"]:.6f})')
    logger.info(f'FRCAR之和: {results["frcar_sum"]:.6f}')
    logger.info('=' * 60)

    return results


def main():
    try:
        parser = argparse.ArgumentParser(description='Barra CNE6 每日风险计算')
        parser.add_argument('--portfolio', type=str, default='random',
                            help='投资组合: random(随机) 或 CSV文件路径')
        parser.add_argument('--output_dir', type=str, default='output',
                            help='输出路径')
        args = parser.parse_args()

        init_qlib()
        run_daily_risk(output_dir=args.output_dir, portfolio_input=args.portfolio)
    except Exception as e:
        logger.error(f'运行出错: {e}')
        send_email(f'Barra CNE6 每日风险计算出错: {e}', traceback.format_exc())
        raise Exception('Barra CNE6 每日风险计算出错')

if __name__ == '__main__':
    main()
