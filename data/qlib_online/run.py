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


@flow(name='qlib_online_shell', log_prints=True, retries=3, retry_delay_seconds=60, timeout_seconds=60 * 60 * 2)
def flow():
    '''Prefect flow: 通过 shell 执行指数成分股更新'''
    now_date = datetime.now().strftime('%Y-%m-%d')
    if not is_trade_day(now_date):
        print(f'{now_date} 非交易日，跳过')
        return

    try:
        script = Path(__file__).parent / 'run.sh'
        result = subprocess.run(
            ['bash', str(script)],
            capture_output=True, text=True,
        )
        print(result.stdout)
        if result.returncode != 0:
            err_msg = f'index_shell({now_date}) 执行失败:\n{result.stderr}'
            print(err_msg)
            raise RuntimeError(err_msg)
        if result.stderr:
            print(result.stderr)
    except Exception as e:
        err_msg = 'qlib_online_shell_flow({}) 执行失败:\n{}'.format(now_date, traceback.format_exc())
        print(err_msg)
        email_send_message_flow(subject='Data: qlib_online', msg=err_msg)
        raise

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--deploy', action='store_true',
                        help='注册 Prefect 部署')
    args = parser.parse_args()

    if args.deploy:
        flow.from_source(
            source=str(Path(__file__).parent),
            entrypoint="run.py:flow"
        ).deploy(
            name="qlib_online_shell",
            work_pool_name="cf_quant"
        )
    else:
        flow()
