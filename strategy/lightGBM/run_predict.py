'''
    指数成分股数据 — Prefect flow (shell 封装)
'''
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_dir))

import traceback
import subprocess
from datetime import datetime
from prefect import flow
from utils import is_trade_day, email_send_message_flow

NAME = 'lightGBM_predict'
SHELL = 'run_predict.sh'
PYTHON_SCRIPT = 'run_predict.py'


@flow(name=f'{NAME}_shell', log_prints=True, retries=3, retry_delay_seconds=60, timeout_seconds=60 * 60 * 2)
def flow(start_date: str = '', end_date: str = ''):
    '''Prefect flow: 通过 shell 执行指数成分股更新'''
    now_date = datetime.now().strftime('%Y-%m-%d')
    if start_date == '' or end_date == '':
        start_date = end_date = now_date
    if not is_trade_day(end_date):
        print(f'{end_date} 非交易日，跳过')
        return

    try:
        script = Path(__file__).parent / f'{SHELL}'
        result = subprocess.run(
            ['bash', str(script), start_date, end_date],
            capture_output=True, text=True,
        )
        print(result.stdout)
        if result.returncode != 0:
            raise RuntimeError(result.stderr)
        if result.stderr:
            print(result.stderr)
    except Exception as e:
        err_msg = f'{NAME}_shell_flow({now_date}) 执行失败:\n{traceback.format_exc()}'
        print(err_msg)
        email_send_message_flow(subject=f'Data: {NAME}', msg=err_msg)
        raise

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--start_date', type=str, help='开始日期')
    parser.add_argument('--end_date', type=str, help='结束日期')
    parser.add_argument('--deploy', action='store_true',
                        help='注册 Prefect 部署')
    args = parser.parse_args()

    if args.deploy:
        flow.from_source(
            source=str(Path(__file__).parent),
            entrypoint=f"{PYTHON_SCRIPT}:flow"
        ).deploy(
            name=f"{NAME}_shell",
            work_pool_name="cf_quant"
        )
    else:
        flow(start_date=args.start_date, end_date=args.end_date)
