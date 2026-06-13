"""
    因子评价体系-定时执行脚本
"""

import argparse
import sys
import traceback
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from prefect import flow
from prefect.logging import get_run_logger
from utils import email_send_message_flow, init_qlib, is_trade_day, dt, LoggerFactory
from qlib.data import D

from utils.io_utils import DataFrameIO
from barra.base import BaseDataLoader
from barra.factor_evaluation import FactorEvalEngine
from config import BENCHMARK_CONFIG
from barra.factor_evaluation.conf import DEFAULT_MAX_DECAY_LAG


logger = LoggerFactory.get_logger(__name__)

def run(
    calc_date: str,
    history_months: int = 24,
    output: str = "./data",
    logger: object = None,
):
    """Load data, run factor evaluation, and persist results.

    Args:
        calc_date: Calculation date (``YYYY-MM-DD``).
        history_months: Months of historical data to load.
        output: Directory path for intermediate PickleIO results.
    """

    output = Path(output)

    init_qlib()
    data_loader = BaseDataLoader(market=BENCHMARK_CONFIG['market'])

    start_date = dt.subtract_months(calc_date, history_months)
    end_date = calc_date
    logger.info('{} \nstart_date: {}, calc_date: {}'.format( '-'*50, start_date, end_date))

    # 加载 close 数据
    instruments = data_loader.load_instruments(start_date, calc_date)
    close_df = D.features(instruments, ["$close"], start_date, calc_date)
    close = close_df["$close"]  # Series
    close.name = 'close'
    close.sort_index(inplace=True)
    logger.info('{}\n {}'.format('-'*50, close))

    # 加载基准（沪深300）close 数据
    bench_close_df = D.features(
        [BENCHMARK_CONFIG['BENCHMARK']], ["$close"], start_date, calc_date,
    )
    benchmark_close = bench_close_df["$close"]

    # 加载风险因子数据。CNE6中包含了 LNCAP 因子，故需要单独再进行市值中性化
    exposure_path = project_root / "barra/factors/data" / calc_date / "exposure_matrix.parquet"
    risk_factors = DataFrameIO.read(str(exposure_path), "parquet")
    logger.info('{}\n {}'.format('-' * 50, risk_factors))

    # 加载 alpha 因子数据
    alpha_factors = data_loader.load_signal(start_date, calc_date)
    alpha_factors.columns = ['alpha1']
    logger.info('{}\n {}'.format('-' * 50, alpha_factors))

    com_index = close.index.intersection(alpha_factors.index).intersection(risk_factors.index)
    close = close.loc[com_index]
    risk_factors = risk_factors.loc[com_index]
    alpha_factors = alpha_factors.loc[com_index]
    DataFrameIO.write(close.to_frame(), output / 'close.parquet')
    DataFrameIO.write(risk_factors, output / 'risk_factors.parquet')
    DataFrameIO.write(alpha_factors, output / 'alpha_factors.parquet')
    DataFrameIO.write(bench_close_df, output / 'bench_close.parquet')
    logger.info('close shape: {}, risk_factors shape: {}, alpha_factors shape: {}, '
                'bench_close shape: {}'.format(
        close.shape, risk_factors.shape, alpha_factors.shape, benchmark_close.shape))

    # ---- Evaluate ----
    engine = FactorEvalEngine(
        close=close, risk_factors=risk_factors, alpha_factors=alpha_factors,
        ic_periods=(1, ),
        benchmark_close=benchmark_close,
    )
    result = engine.run(
        neutralize=True,
        n_groups=5,
        max_decay_lag=DEFAULT_MAX_DECAY_LAG,
        output=str(output),
    )

    # ---- Persist to MySQL ----
    engine.save_to_mysql(result, calc_date)
    logger.info(f"Factor evaluation completed for {calc_date}")


@flow(name="factor_evaluation", log_prints=True, retries=3, retry_delay_seconds=600, timeout_seconds=60 * 60 * 1)
def flow(now_date: str = ""):
    """Prefect flow: daily factor evaluation."""
    now_date = now_date or datetime.now().strftime("%Y-%m-%d")
    logger = get_run_logger()
    if not is_trade_day(now_date):
        logger.info(f"{now_date} 非交易日，跳过")
        return

    try:
        run(calc_date=now_date,
            output=f"./data/{now_date}",
            logger=logger)
    except Exception:
        err_msg = f"factor_evaluation flow({now_date}) 执行失败:\n{traceback.format_exc()}"
        logger.error(err_msg)
        email_send_message_flow(subject="Data: factor_evaluation", msg=err_msg)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="因子评价 — 每日运行脚本")
    parser.add_argument("--now-date", type=str, help="计算日期 (YYYY-MM-DD)，为空时默认当天")
    parser.add_argument("--history-months", type=int, default=12, help="历史数据月数")
    parser.add_argument("--output", type=str, default="./data", help="输出目录")
    parser.add_argument("--deploy", action="store_true", help="注册 Prefect 部署")
    args = parser.parse_args()

    if args.deploy:
        flow.from_source(
            source=str(Path(__file__).parent),
            entrypoint="run.py:flow",
        ).deploy(
            name="factor_evaluation",
            work_pool_name="cf_quant",
        )
    else:
        init_qlib()
        run(
            calc_date=args.now_date,
            history_months=args.history_months,
            output=f'{args.output}/{args.now_date}',
            logger=logger
        )
