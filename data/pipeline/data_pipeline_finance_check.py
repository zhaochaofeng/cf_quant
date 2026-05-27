'''
    编排 flow: 财务数据定期检查
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


def _check_step(result: FlowRun, name: str):
    if result.state is None or result.state.is_failed():
        raise RuntimeError(f"上游 Deployment {name} 最终失败，终止")


@flow(name='data_pipeline_finance_check', log_prints=True, timeout_seconds=60 * 60 * 20)
def flow():
    '''编排 flow: 依次执行各子 flow，UI 中显示完整血缘关系'''
    now_date  = datetime.now().strftime('%Y-%m-%d')

    print(f'--- 步骤 1: income_ts ({now_date}) ---')
    _check_step(run_deployment(
        "income_ts_check_shell/income_ts_check_shell",
        as_subflow=True,
    ), "income_ts_check_shell/income_ts_check_shell")

    print(f'--- 步骤 2: balance_ts ({now_date}) ---')
    _check_step(run_deployment(
        "balance_ts_check_shell/balance_ts_check_shell",
        as_subflow=True,
    ), "balance_ts_check_shell/balance_ts_check_shell")

    print(f'--- 步骤 3: cashflow_ts ({now_date}) ---')
    _check_step(run_deployment(
        "cashflow_ts_check_shell/cashflow_ts_check_shell",
        as_subflow=True,
    ), "cashflow_ts_check_shell/cashflow_ts_check_shell")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--deploy', action='store_true',
                        help='注册 Prefect 部署')
    args = parser.parse_args()

    if args.deploy:
        schedule = Schedule(
            cron="1 1 * * 7",
            timezone="Asia/Shanghai",
        )
        flow.from_source(
            source=str(Path(__file__).parent),
            entrypoint="data_pipeline_finance_check.py:flow",
        ).deploy(
            name="data_pipeline_finance_check",
            work_pool_name="cf_quant",
            schedule=schedule,
        )
    else:
        flow()
