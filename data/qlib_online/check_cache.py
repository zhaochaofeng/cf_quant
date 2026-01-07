"""
    检查 缓存更新 是否准确
"""
import traceback
from datetime import datetime
from pprint import pprint

import time
import fire
import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config

from utils import get_config
from utils import send_email


def check(
        provider_uri: str = "~/.qlib/qlib_data/custom_data_hfq",
        start_date: str = None,
        end_date: str = None,
        epsilon: float = 1e-6
        ):
    try:
        t = time.time()
        if start_date is None:
            start_date = datetime.now().strftime("%Y-%m-%d")
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        # instruments = ['SH688012', 'SZ002049']
        data_handler_config = {
            'start_time': start_date,
            'end_time': end_date,
            'fit_start_time': start_date,
            'fit_end_time': end_date,
            'instruments': 'csi300',
            # 'instruments': instruments,
        }
        handler = {
            'class': 'Alpha158',
            'module_path': 'qlib.contrib.data.handler',
            'kwargs': data_handler_config,
        }

        pprint(handler)

        qlib.init(provider_uri=provider_uri)
        h1 = init_instance_by_config(handler)
        data1 = h1.fetch(col_set='__all', data_key='infer')
        data1.drop(columns=['LABEL0'], inplace=True)   # 非缓存数据有些标签为NaN
        print('\n{}\n{}'.format('-' * 50, data1))

        config = get_config()
        qlib.init(
            default_conf="server",
            region=REG_CN,
            redis_host=config['redis']['host'],
            redis_port=config['redis']['port'],
            redis_task_db=3,
            redis_password=config['redis']['password'],
            provider_uri=provider_uri
        )
        h2 = init_instance_by_config(handler)
        data2 = h2.fetch(col_set='__all', data_key='infer')
        data2.drop(columns=['LABEL0'], inplace=True)
        print('\n{}\n{}'.format('-' * 50, data1))

        # diff = (data1.eq(data2)) | ((data1.isna()) & (data2.isna()))
        diff = (abs(data1 - data2) < epsilon) | ((data1.isna()) & (data2.isna()))
        mask_ne = (diff.ne(True)).any(axis=1)
        index_ne = diff.index[mask_ne]
        res = []
        for index in index_ne:
            res.append('{}'.format(index))

        if len(res) > 0:
            print(res)
            send_email('Data:qlib_online:check_cache', '\n'.join(res))
        else:
            print('没有异常 ！！！')
        print('Consume time: {}s'.format(round(time.time() - t, 4)))
    except:
        err_mag = traceback.format_exc()
        send_email('Data:qlib_online:check_cache', err_mag)


if __name__ == '__main__':
    fire.Fire(check)


