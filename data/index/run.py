'''
    指数成分股数据 — Prefect flow (shell 封装)
'''
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_dir))

import subprocess
from datetime import datetime

from prefect import flow
from prefect.schedules import Schedule

from utils import is_trade_day


@flow(name='index_shell', log_prints=True)
def flow(now_date: str = ''):
    '''Prefect flow: 通过 shell 执行指数成分股更新'''
    now_date = now_date or datetime.now().strftime('%Y-%m-%d')
    if not is_trade_day(now_date):
        print(f'{now_date} 非交易日，跳过')
        return

    script = Path(__file__).parent / 'run.sh'
    result = subprocess.run(
        ['bash', str(script), now_date],
        capture_output=True, text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        err_msg = f'index_shell({now_date}) 执行失败:\n{result.stderr}'
        print(err_msg)
        raise RuntimeError(err_msg)
    if result.stderr:
        print(result.stderr)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--now-date', type=str, default='',
                        help='日期 (YYYY-MM-DD)，为空时默认当天')
    parser.add_argument('--deploy', action='store_true',
                        help='注册 Prefect 部署')
    args = parser.parse_args()

    if args.deploy:
        schedule = Schedule(
            cron="1 2 * * *",
            timezone="Asia/Shanghai",
        )
        flow.from_source(
            source=str(Path(__file__).parent),
            entrypoint="run.py:flow",
        ).deploy(
            name="index_shell",
            work_pool_name="cf_quant",
            schedule=schedule,
        )
    else:
        flow(now_date=args.now_date)
