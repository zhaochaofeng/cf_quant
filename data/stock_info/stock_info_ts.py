'''
    功能：股票基本信息表（沪深京3个交易所股票）
    数据来源：Tushare
    表：stock_info_ts
'''

import time
import argparse
import pandas as pd
import traceback
from datetime import datetime
from utils.utils import tushare_pro
from utils.utils import mysql_connect
from utils.utils import is_trade_day
from utils.utils import send_email

pro = tushare_pro()

fea_1 = ['ts_code', 'name', 'area', 'industry', 'cnspell', 'market', 'list_date', 'act_name', 'act_ent_type', 'status']
fea_2 = ['qlib_code', 'day', 'exchange']  # 新增字段
exchange_map = {'BJ': 'BSE', 'SH': 'SSE', 'SZ': 'SZSE'}
exclude_codes = set(['T00018.SH'])  # 噪声数据

def main(args):
    try:
        # 上市股票
        df_l = pro.stock_basic(list_status='L')
        df_l['status'] = 1
        # 退市股票
        df_d = pro.stock_basic(list_status='D')
        df_d['status'] = 0
        df = pd.concat([df_l, df_d], ignore_index=True)
        print('-' * 100)
        print(df.iloc[0])
        print('股票数：{}'.format(len(df)))
    except Exception as e:
        raise Exception('error in get stock info: {}'.format(e))

    conn = mysql_connect()
    data = []
    for index, row in df.iterrows():
        if row['ts_code'] in exclude_codes:
            continue
        tmp = {}  # 单条数据
        tmp['day'] = args.date
        for f in fea_1:
            v = row[f]
            if f == 'ts_code':
                code, suffix = v.split('.')
                tmp['exchange'] = exchange_map[suffix]
                tmp['qlib_code'] = '{}{}'.format(suffix, code)
            if f in ['list_date']:
                v = datetime.strptime(v, '%Y%m%d').strftime('%Y-%m-%d')
            if pd.isna(v):
                v = None
            tmp[f] = v
        data.append(tmp)

    print('-' * 100)
    print("data len: {}".format(len(data)))
    fea_merge = fea_1 + fea_2
    fea_merge_format = ['%({})s'.format(f) for f in fea_merge]
    sql = '''
        INSERT INTO stock_info_ts({}) VALUES ({})
    '''.format(','.join(fea_merge), ','.join(fea_merge_format))
    with conn.cursor() as cursor:
        try:
            print(cursor.mogrify(sql, data[0]))
            cursor.executemany(sql, data)
            conn.commit()
        except Exception as e:
            conn.rollback()  # 回滚
            raise Exception('error in write to mysql: {}'.format(e))
        finally:
            conn.close()
    print('写入完成!!!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str, default='2025-09-09', help='取数日期')
    args = parser.parse_args()
    if not is_trade_day(args.date):
        print('不是交易日，退出！！！')
        exit(0)
    print(args)
    t = time.time()
    try:
        main(args)
    except:
        error_info = traceback.format_exc()
        print(error_info)
        send_email('Data: stock_info_ts', error_info)
    print('耗时：{}s'.format(round(time.time() - t, 4)))

