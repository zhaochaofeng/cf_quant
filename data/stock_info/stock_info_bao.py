'''
    功能：股票基本信息表（沪深京3个交易所股票）
    数据来源：Tushare
    表：stock_info_ts
'''

import time
import argparse
import pandas as pd
import traceback
from utils.utils import mysql_connect
from utils.utils import is_trade_day
from utils.utils import send_email
from utils.utils import bao_stock_connect

fea_1 = {
    'code': 'code',
    'name': 'code_name',
    'list_date': 'ipoDate',
    'out_date': 'outDate',
    'status': 'status'
}
fea_2 = ['qlib_code', 'day', 'exchange']
exchange_map = {'BJ': 'BSE', 'SH': 'SSE', 'SZ': 'SZSE'}
# sh.600849在2013年底退市，为了与tushare保持数据一致，做过滤处理
exclude_list = set(['sh.600849', 'sz.000022', 'sz.000043', 'sz.300114'])

bs = bao_stock_connect()

def main(args):
    try:
        rs = bs.query_stock_basic()
        data_list = []
        while (rs.error_code == '0') & rs.next():
            # 获取一条记录，将记录合并在一起
            data_list.append(rs.get_row_data())
        df = pd.DataFrame(data_list, columns=rs.fields)
        df = df[df['type'] == '1'][['code', 'code_name', 'ipoDate', 'outDate', 'status']]
        print('-' * 100)
        print(df.iloc[0])
        print('股票数：{}'.format(len(df)))
        bs.logout()
    except Exception as e:
        raise Exception('error in get stock info: {}'.format(e))

    conn = mysql_connect()
    data = []
    for index, row in df.iterrows():
        if row['code'] in exclude_list:
            continue
        tmp = {}    # 单条数据
        tmp['day'] = args.date
        for f in fea_1.keys():
            v = row[fea_1[f]]
            if f == 'code':
                suffix, code = v.split('.')
                tmp['exchange'] = exchange_map[suffix.upper()]
                tmp['qlib_code'] = '{}{}'.format(suffix.upper(), code)
            if pd.isna(v) or v == '':
                v = None
            tmp[f] = v
        data.append(tmp)

    print('-' * 100)
    print("data len: {}".format(len(data)))
    fea_merge = list(fea_1.keys()) + fea_2
    fea_merge_format = ['%({})s'.format(f) for f in fea_merge]
    sql = '''
        INSERT INTO stock_info_bao({}) VALUES ({})
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
        send_email('Data: stock_info_bao', error_info)
    print('耗时：{}s'.format(round(time.time() - t, 4)))
