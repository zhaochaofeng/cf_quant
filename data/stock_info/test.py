from prefect import flow

@flow(log_prints=True)
def my_flow(name: str = 'flow'):
    print(f'hello {name}')

if __name__ == '__main__':
    my_flow.from_source(   # 指定 flow 函数代码路径
    # source=".",
    # entrypoint="test.py:my_flow"
    source="https://github.com/zhaochaofeng/cf_quant.git",
    entrypoint="flows.py:my_flow",
    ).deploy(
        name='my-flow',
        work_pool_name='cf_quant',   #  指定 work pool
    )




