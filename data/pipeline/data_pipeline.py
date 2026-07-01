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
from prefect.client.schemas.objects import FlowRun
from prefect.schedules import Schedule
from utils import is_trade_day


def _check_step(result: FlowRun, name: str):
    if result.state is None or result.state.is_failed():
        raise RuntimeError(f"上游 Deployment {name} 最终失败，终止")


@flow(name='data_pipeline', log_prints=True, timeout_seconds=60 * 60 * 20)
def flow(start_date: str = '', end_date: str = '', now_date: str = ''):
    '''编排 flow: 依次执行各子 flow，UI 中显示完整血缘关系'''
    now_date = now_date or datetime.now().strftime('%Y-%m-%d')
    if start_date == '' or end_date == '':
        start_date = end_date = now_date
    if not is_trade_day(end_date):
        print(f'{end_date} 非交易日，跳过')
        return

    print(f'--- shibor_rate ({now_date}) ---')
    _check_step(run_deployment(
        "shibor_rate/shibor_rate",
        parameters={"start_date": now_date, "end_date": now_date},
        as_subflow=True,
    ), "shibor_rate/shibor_rate")

    print(f'--- suspend_d ({now_date}) ---')
    _check_step(run_deployment(
        "suspend_d/suspend_d",
        parameters={"start_date": now_date, "end_date": now_date},
        as_subflow=True,
    ), "suspend_d/suspend_d")

    print(f'--- 步骤 1: stock_info_ts ({now_date}) ---')
    _check_step(run_deployment(
        "stock_info_ts/stock_info_ts",
        parameters={"now_date": now_date},
        as_subflow=True,
    ), "stock_info_ts/stock_info_ts")

    print(f'--- 步骤 2: valuation_ts ({now_date}) ---')
    _check_step(run_deployment(
        "valuation_ts/valuation_ts",
        parameters={"start_date": start_date, "end_date": end_date, "now_date": now_date},
        as_subflow=True,
    ), "valuation_ts/valuation_ts")

    print(f'--- 步骤 3: check_valuation_ts ({now_date}) ---')
    _check_step(run_deployment(
        "check_valuation_ts/check_valuation_ts",
        parameters={"now_date": now_date},
        as_subflow=True,
    ), "check_valuation_ts/check_valuation_ts")

    print(f'--- 步骤 4: trade_daily_ts ({now_date}) ---')
    _check_step(run_deployment(
        "trade_daily_ts/trade_daily_ts",
        parameters={"start_date": start_date, "end_date": end_date, "now_date": now_date},
        as_subflow=True,
    ), "trade_daily_ts/trade_daily_ts")

    print(f'--- 步骤 5: update_factor ({now_date}) ---')
    _check_step(run_deployment(
        "update_factor/update_factor",
        parameters={"now_date": now_date},
        as_subflow=True,
    ), "update_factor/update_factor")

    print(f'--- 步骤 6: income_ts ({now_date}) ---')
    _check_step(run_deployment(
        "income_ts_shell/income_ts_shell",
        as_subflow=True
    ), "income_ts_shell/income_ts_shell")

    print(f'--- 步骤 7: balance_ts ({now_date}) ---')
    _check_step(run_deployment(
        "balance_ts_shell/balance_ts_shell",
        as_subflow=True
    ), "balance_ts_shell/balance_ts_shell")

    print(f'--- 步骤 8: cashflow_ts ({now_date}) ---')
    _check_step(run_deployment(
        "cashflow_ts_shell/cashflow_ts_shell",
        as_subflow=True
    ), "cashflow_ts_shell/cashflow_ts_shell")

    print(f'--- 步骤 9: qlib_online ({now_date}) ---')
    _check_step(run_deployment(
        "qlib_online_shell/qlib_online_shell",
        as_subflow=True
    ), "qlib_online_shell/qlib_online_shell")

    print(f'--- 步骤 10: qlib_online_check ({now_date}) ---')
    _check_step(run_deployment(
        "qlib_online_check_shell/qlib_online_check_shell",
        as_subflow=True
    ), "qlib_online_check_shell/qlib_online_check_shell")

    print(f'--- 步骤 11: lightGBM_train ({now_date}) ---')
    _check_step(run_deployment(
        "lightGBM_train_shell/lightGBM_train_shell",
        as_subflow=True
    ), "lightGBM_train_shell/lightGBM_train_shell")

    print(f'--- 步骤 12: lightGBM_predict ({now_date}) ---')
    _check_step(run_deployment(
        "lightGBM_predict_shell/lightGBM_predict_shell",
        as_subflow=True
    ), "lightGBM_predict_shell/lightGBM_predict_shell")

    print(f'--- 步骤 13: factors_exposure ({now_date}) ---')
    _check_step(run_deployment(
        "factors_exposure/factors_exposure",
        as_subflow=True
    ), "factors_exposure/factors_exposure")

    print(f'--- 步骤 14: factor_evaluation ({now_date}) ---')
    _check_step(run_deployment(
        "factor_evaluation/factor_evaluation",
        as_subflow=True
    ), "factor_evaluation/factor_evaluation")

    """
    print(f'--- 步骤 15: barra_risk ({now_date}) ---')
    _check_step(run_deployment(
        "barra_risk/barra_risk",
        parameters={"now_date": now_date},
        as_subflow=True,
    ), "barra_risk/barra_risk")

    print(f'--- 步骤 16: barra_alpha ({now_date}) ---')
    _check_step(run_deployment(
        "barra_alpha/barra_alpha",
        parameters={"now_date": now_date},
        as_subflow=True,
    ), "barra_alpha/barra_alpha")

    print(f'--- 步骤 17: barra_portfolio ({now_date}) ---')
    _check_step(run_deployment(
        "barra_portfolio/barra_portfolio",
        parameters={"now_date": now_date},
        as_subflow=True,
    ), "barra_portfolio/barra_portfolio")
    """

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--now-date', type=str, default='',
                        help='日期 (YYYY-MM-DD)，为空时默认当天')
    parser.add_argument('--deploy', action='store_true',
                        help='注册 Prefect 部署')
    args = parser.parse_args()

    if args.deploy:
        schedule = Schedule(
            cron="1 20 * * *",
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



