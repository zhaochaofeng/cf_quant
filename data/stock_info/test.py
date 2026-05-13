import time
from prefect import flow
from prefect.schedules import Schedule

@flow(log_prints=True)
def my_flow():
    print(f'hello flow {time.time()}')

if __name__ == '__main__':
    schedule = Schedule(
        cron="41 16 * * *",
        timezone="Asia/Shanghai",
    )
    my_flow.from_source(
    source="https://github.com/zhaochaofeng/cf_quant.git",
    entrypoint="data/stock_info/test.py:my_flow",
    ).deploy(
        name='my-flow-test',
        work_pool_name='cf_quant',   #  指定 work pool
        schedule=schedule,
    )






