"""
Barra CNE6 风险模型 - 日频统一运行脚本
流程：模型估计(因子暴露→回归→协方差→特异风险) → 风险归因 → CSV + MySQL 输出
"""
import os
import sys
import argparse
import traceback
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import qlib

from barra.risk_control.barra_engine import BarraRiskEngine
from barra.risk_control.config import CNE6_STYLE_FACTORS, INDUSTRY_MAPPING, PROVIDER_URI, BENCHMARK_CONFIG
from barra.risk_control.output import RiskOutputManager
from barra.risk_control.portfolio import PortfolioManager
from barra.risk_control.risk_attribution import RiskAttributionAnalyzer
from utils import LoggerFactory, dt, send_email

logger = LoggerFactory.get_logger(__name__)


def init_qlib():
    """初始化qlib，注册PTTM自定义操作符"""
    from utils.qlib_ops import PTTM
    qlib.init(
        provider_uri=PROVIDER_URI,
        custom_ops=[PTTM],
    )
    logger.info('Qlib初始化完成')


def get_factor_types() -> pd.Series:
    """获取因子类型映射（风格因子 + 行业因子）"""
    category_map = {
        'size': '规模', 'volatility': '波动率', 'liquidity': '流动性',
        'momentum': '动量', 'quality_leverage': '质量-杠杆',
        'quality_earn_vol': '质量-盈利波动',
        'quality_earn_quality': '质量-盈利质量',
        'quality_profit': '质量-盈利能力',
        'quality_invest': '质量-投资质量',
        'value': '价值', 'growth': '成长',
    }
    factor_types = {}
    for category, factors in CNE6_STYLE_FACTORS.items():
        for f in factors:
            factor_types[f] = category_map.get(category, category)
    for code, name in INDUSTRY_MAPPING.items():
        factor_types[name] = '行业'
    return pd.Series(factor_types)


def run(calc_date: str, history_months: int = 24,
        output_dir: str = 'output', n_jobs: int = 4,
        portfolio_input: str = 'random',
        use_cache: bool = False) -> dict:
    """
    日频统一流程

    Args:
        calc_date: 计算日期，格式'YYYY-MM-DD'
        history_months: 历史数据月数
        output_dir: 输出目录
        n_jobs: 并行进程数
        portfolio_input: 投资组合输入('random' 或 CSV 路径)
        use_cache: 是否使用缓存数据

    Returns:
        风险分析结果字典
    """
    start_date = dt.subtract_months(calc_date, history_months)

    logger.info('=' * 70)
    logger.info(f'Barra CNE6 日频风险计算: {calc_date}')
    logger.info(f'数据区间: {start_date} ~ {calc_date}')
    logger.info('=' * 70)

    # 1. 模型估计（因子暴露→横截面回归→协方差F→特异风险Delta）
    engine = BarraRiskEngine(
        calc_date, market=BENCHMARK_CONFIG['market'], output_dir=output_dir, n_jobs=n_jobs)
    engine.run(start_date, calc_date, use_cache)

    # 2. 加载模型输出
    output_mgr = RiskOutputManager(output_dir=output_dir)
    F = output_mgr.load_data(
        'model/factor_covariance.parquet', type='parquet')
    delta_df = output_mgr.load_data(
        'model/specific_risk.parquet', type='parquet')
    X_all = output_mgr.load_data(
        'debug/exposure_matrix.parquet', type='parquet')

    if F is None or delta_df is None or X_all is None:
        raise FileNotFoundError('模型数据文件缺失')

    # 3. 提取最新日期因子暴露，对齐索引
    latest_date = X_all.index.get_level_values('datetime').max()
    X_t = X_all.xs(latest_date, level='datetime')

    common_factors = X_t.columns.intersection(F.index)
    common_instruments = X_t.index.intersection(delta_df.index)

    F = F.loc[common_factors, common_factors]
    X_t = X_t.loc[common_instruments, common_factors]

    delta_diag = delta_df.loc[common_instruments, 'specific_var']

    logger.info(f'对齐后: {len(common_factors)}个因子, '
                f'{len(common_instruments)}只股票')

    # 4. 计算 V = X·F·X^T + Δ
    X_vals = X_t.values
    V_vals = X_vals @ F.values @ X_vals.T + np.diag(delta_diag.values)
    V = pd.DataFrame(
        V_vals, index=common_instruments, columns=common_instruments)
    logger.info(f'资产协方差矩阵 V: {V.shape}')

    # 5. 获取组合/基准权重
    portfolio_mgr = PortfolioManager(market='csi300')
    if portfolio_input == 'random':
        h_p = portfolio_mgr.generate_random_portfolio(
            calc_date, n_stocks=50, random_state=42)
    else:
        h_p = portfolio_mgr.load_portfolio(portfolio_input, calc_date)
    h_b = portfolio_mgr.get_benchmark_weights(calc_date)
    logger.info(f'组合: {len(h_p)}只股票, 基准: {len(h_b)}只股票')

    # 6. 风险归因
    analyzer = RiskAttributionAnalyzer()
    results = analyzer.analyze_risk(V, F, X_t, h_p, h_b)

    # 7. 输出 CSV + MySQL
    factor_types = get_factor_types()
    stock_file = output_mgr.save_stock_risk(
        results['mcar'], results['rcar'], calc_date)
    factor_file = output_mgr.save_factor_risk(
        results['fmcar'], results['frcar'], factor_types, calc_date)
    output_mgr.save_to_mysql(
        results, calc_date, factor_types,
        portfolio_name=portfolio_input)

    logger.info(f'股票风险: {stock_file}')
    logger.info(f'因子风险: {factor_file}')

    # 摘要
    logger.info('=' * 70)
    logger.info(f'组合总风险: {results["total_risk"]:.6f}')
    logger.info(f'主动风险(跟踪误差): {results["active_risk"]:.6f}')
    logger.info(f'RCAR之和: {results["rcar_sum"]:.6f} '
                f'(应≈主动风险 {results["active_risk"]:.6f})')
    logger.info(f'FRCAR之和: {results["frcar_sum"]:.6f}')
    logger.info('=' * 70)

    return results


def main():
    try:
        parser = argparse.ArgumentParser(
            description='Barra CNE6 日频风险计算')
        parser.add_argument(
            '--date', type=str, required=True,
            help='计算日期，格式YYYY-MM-DD')
        parser.add_argument(
            '--history-months', type=int, default=24,
            help='历史数据月数')
        parser.add_argument(
            '--output_dir', type=str, default='output',
            help='输出路径')
        parser.add_argument(
            '--n-jobs', type=int, default=os.cpu_count()-2,
            help='并行计算核心数')
        parser.add_argument(
            '--portfolio', type=str, default='random',
            help='投资组合: random(随机) 或 CSV文件路径')
        parser.add_argument(
            '--use-cache', action='store_true',
            help='是否使用缓存')
        args = parser.parse_args()

        init_qlib()
        run(calc_date=args.date,
            history_months=args.history_months,
            output_dir=args.output_dir + f'/{args.date}',
            n_jobs=args.n_jobs,
            portfolio_input=args.portfolio,
            use_cache=args.use_cache)
    except Exception as e:
        logger.error(f'运行出错: {e}')
        send_email(f'Barra CNE6 风险计算出错: {e}', traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
