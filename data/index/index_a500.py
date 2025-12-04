"""
    中证 A500 指数成分股
"""

import time
import traceback
from calendar import monthrange
from datetime import datetime, timedelta

import fire
import pandas as pd

from utils import send_email
from utils import tushare_pro


def get_last_month_range(date: str = None):
    """
    获取当前日期的上一个月的第1天和最后1天

    Args:
        date (str, optional): 指定日期，格式为 'YYYY-MM-DD' 或 'YYYYMMDD'。默认为None（使用当前日期）

    Returns:
        tuple: (上月第一天, 上月最后一天)，格式为 ('YYYYMMDD', 'YYYYMMDD')
    """
    # 处理日期参数
    if date is None:
        target_date = datetime.now()
    else:
        # 移除日期中的横线
        date = date.replace('-', '')
        target_date = datetime.strptime(date, '%Y%m%d')

    # 计算上一个月的年份和月份
    if target_date.month == 1:
        # 如果当前是1月，则上个月是去年12月
        last_month_year = target_date.year - 1
        last_month = 12
    else:
        last_month_year = target_date.year
        last_month = target_date.month - 1

    # 上个月第一天
    first_day = datetime(last_month_year, last_month, 1)

    # 上个月最后一天
    _, last_day_num = monthrange(last_month_year, last_month)
    last_day = datetime(last_month_year, last_month, last_day_num)

    return (
        first_day.strftime('%Y%m%d'),
        last_day.strftime('%Y%m%d')
    )


def main(
        date: str = None,
        instruments_path: str = '~/.qlib/qlib_data/custom_data_hfq/instruments/all.txt',
        output_path: str = '~/.qlib/qlib_data/index/instruments/csia500.txt'
    ):
    try:
        t = time.time()
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        date = date.replace('-', '')
        start_date, end_date = get_last_month_range()
        pro = tushare_pro()
        df = pro.index_weight(index_code='000510.SH', start_date=start_date, end_date=end_date)
        df['con_code'] = df['con_code'].apply(lambda x: '{}{}'.format(x[7:9], x[0:6]))
        df.set_index('con_code', inplace=True)

        all = pd.read_csv(instruments_path, sep='\t')
        all.columns = ['code', 'start_date', 'end_date']
        all.set_index('code', inplace=True)
        CSIA500 = all.reindex(df.index)
        CSIA500['end_date'] = date
        CSIA500.reset_index(inplace=True)
        CSIA500.to_csv(output_path, sep='\t', header=False, index=False)
        print('耗时：{}'.format(round(time.time() - t, 4)))
    except:
        error_msg = traceback.format_exc()
        send_email('Data:index:CSIA500', error_msg)


if __name__ == '__main__':
    fire.Fire(main)
