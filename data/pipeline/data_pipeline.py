'''
    编排 flow: stock_info_ts → valuation_ts → (后续扩展)
'''
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_dir))

import argparse
from datetime import datetime

from prefect import flow
from prefect.deployments import run_deployment
from prefect.schedules import Schedule
from utils import is_trade_day


@flow(name='data_pipeline', log_prints=True, timeout_seconds=60 * 60 * 20)
def flow(start_date: str = '', end_date: str = '', now_date: str = ''):
    '''编排 flow: 依次执行各子 flow，UI 中显示完整血缘关系'''
    now_date = now_date or datetime.now().strftime('%Y-%m-%d')
    if start_date == '' or end_date == '':
        start_date = end_date = now_date
    if not is_trade_day(end_date):
        print(f'{end_date} 非交易日，跳过')
        return

    print(f'--- 步骤 1: stock_info_ts ({now_date}) ---')
    run_deployment(
        "stock_info_ts/stock_info_ts",
        parameters={"now_date": now_date},
        as_subflow=True,
    )

    print(f'--- 步骤 2: valuation_ts ({now_date}) ---')
    run_deployment(
        "valuation_ts/valuation_ts",
        parameters={"start_date": start_date, "end_date": end_date, "now_date": now_date},
        as_subflow=True,
    )

    print(f'--- 步骤 3: check_valuation_ts ({now_date}) ---')
    run_deployment(
        "check_valuation_ts/check_valuation_ts",
        parameters={"now_date": now_date},
        as_subflow=True,
    )

    print(f'--- 步骤 4: trade_daily_ts ({now_date}) ---')
    run_deployment(
        "trade_daily_ts/trade_daily_ts",
        parameters={"start_date": start_date, "end_date": end_date, "now_date": now_date},
        as_subflow=True,
    )

    print(f'--- 步骤 5: update_factor ({now_date}) ---')
    run_deployment(
        "update_factor/update_factor",
        parameters={"now_date": now_date},
        as_subflow=True,
    )

    print(f'--- 步骤 6: income_ts ({now_date}) ---')
    run_deployment(
        "income_ts_shell/income_ts_shell",
        as_subflow=True
    )

    print(f'--- 步骤 7: balance_ts ({now_date}) ---')
    run_deployment(
        "balance_ts_shell/balance_ts_shell",
        as_subflow=True
    )

    print(f'--- 步骤 8: cashflow_ts ({now_date}) ---')
    run_deployment(
        "cashflow_ts_shell/cashflow_ts_shell",
        as_subflow=True
    )

    print(f'--- 步骤 9: qlib_online ({now_date}) ---')
    run_deployment(
        "qlib_online_shell/qlib_online_shell",
        as_subflow=True
    )

    print(f'--- 步骤 10: qlib_online_check ({now_date}) ---')
    run_deployment(
        "qlib_online_check_shell/qlib_online_check_shell",
        as_subflow=True
    )







if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--now-date', type=str, default='',
                        help='日期 (YYYY-MM-DD)，为空时默认当天')
    parser.add_argument('--deploy', action='store_true',
                        help='注册 Prefect 部署')
    args = parser.parse_args()

    if args.deploy:
        schedule = Schedule(
            cron="1 18 * * *",
            timezone="Asia/Shanghai",
        )
        flow.from_source(
            source=str(Path(__file__).parent),
            entrypoint="data_pipeline.py:flow",
        ).deploy(
            name="data_pipeline",
            work_pool_name="cf_quant",
            schedule=schedule,
        )
    else:
        flow(now_date=args.now_date)



