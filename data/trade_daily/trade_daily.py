'''
    功能：tushare表导入mysql
    描述：日级交易数据（非复权）+ 复权因子
    刷数：长周期则设置[start_date, end_date]，短周期则按天刷
'''

import time
import argparse
import pandas as pd
from datetime import datetime
from utils.utils import get_config
from utils.utils import mysql_connect
from utils.utils import tushare_ts, tushare_pro, is_trade_day
from utils.utils import send_email
import traceback

import warnings
warnings.filterwarnings("ignore")

config = get_config()
ts = tushare_ts()
pro = tushare_pro()
batch_size = 10000    # 每次写入mysql的行数

fea_to_from = {
     'ts_code': 'ts_code',
     'day': 'trade_date',
     'open': 'open',
     'high': 'high',
     'low': 'low',
     'close': 'close',
     'pre_close': 'pre_close',
     'change': 'change',
     'pct_chg': 'pct_chg',
     'vol': 'vol',
     'amount': 'amount',
     'adj_factor': 'adj_factor'
}

def parse_line(row, fea_to_from):
    ''' 解析数据 '''
    tmp = {}
    for f in fea_to_from.keys():
        try:
            v = row[fea_to_from[f]]
            if f == 'day':
                # 日期格式转换
                v = datetime.strptime(v, '%Y%m%d').strftime('%Y-%m-%d')
            if f == 'pct_chg':
                # 处理pct_chg超出范围的情况
                if pd.isna(v):
                    v = None
                elif abs(v) > 9999.99:
                    print(f"警告: pct_chg值 {v} 超出范围，已截断为9999.99或-9999.99")
                    print(row)
                    v = 9999.99 if v > 0 else -9999.99
            if pd.isna(v):
                v = None
            tmp[f] = v
        except Exception as e:
            raise Exception('parse_line error: {}'.format(e))
    return tmp

def request_from_tushare(ts_codes):
    # 从tushare API获取数据
    print('-' * 100)
    print('从tushare API获取数据...')
    start_date = datetime.strptime(args.start_date, '%Y-%m-%d').strftime('%Y%m%d')
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d').strftime('%Y%m%d')
    print('start_date: {}, end_date: {}'.format(start_date, end_date))
    data = []

    if start_date == end_date:
        # 按天更新数据，则批量请求
        df = pd.DataFrame()
        k = len(ts_codes) // 1000 if len(ts_codes) % 1000 == 0 else len(ts_codes) // 1000 + 1
        # 交易数据
        for i in range(k):
            codes = ts_codes[i * 1000: min((i + 1) * 1000, len(ts_codes))]
            tmp = pro.daily(ts_code=','.join(codes),
                            start_date=start_date, end_date=end_date)
            df = pd.concat([df, tmp], axis=0, join='outer')
        df.set_index(keys=['ts_code', 'trade_date'], inplace=True)
        print(df.head())
        # 复权因子
        factor = pro.adj_factor(trade_date=start_date)
        factor.set_index(keys=['ts_code', 'trade_date'], inplace=True)

        print(factor.head())
        # 合并交易数据 和 复权因子
        merged = pd.concat([df, factor], axis=1, join='inner')

        print(merged.head())
        # 有些股票由于停盘等原因没有交易数据，所以df.shape[0] 可以小于 len(ts_codes)
        print('df shape: {}'.format(df.shape))
        print('factor shape: {}'.format(factor.shape))
        print('merged shape: {}'.format(merged.shape))
        if merged.shape[0] != df.shape[0]:
            raise Exception('merged.shape[0]({}) < df.shape[0]({})'.format(merged.shape[0], df.shape[0]))

        merged.reset_index(inplace=True)
        for index, row in merged.iterrows():
            tmp = parse_line(row, fea_to_from)
            data.append(tmp)
    else:
        # 回刷历史数据，按单个ts_code请求
        for i, code in enumerate(ts_codes):
            time.sleep(0.1)  # API调用频次：1min不超过1000次
            if (i + 1) % 100 == 0:
                print('requested num: {}'.format(i + 1))
            df = pro.daily(ts_code=code, start_date=start_date, end_date=end_date)
            factor = pro.adj_factor(ts_code=code, start_date=start_date, end_date=end_date)
            df.set_index(keys=['ts_code', 'trade_date'], inplace=True)
            factor.set_index(keys=['ts_code', 'trade_date'], inplace=True)
            merged = pd.concat([df, factor], axis=1, join='inner')
            merged.reset_index(inplace=True)

            print('df shape: {}'.format(df.shape))
            print('factor shape: {}'.format(factor.shape))
            print('merged shape: {}'.format(merged.shape))
            if merged.shape[0] != df.shape[0]:
                raise Exception('code: {}, merged.shape[0]({}) < df.shape[0]({})'.format(code, merged.shape[0], df.shape[0]))

            for index, row in merged.iterrows():
                tmp = parse_line(row, fea_to_from)
                data.append(tmp)
    return data

def write_to_mysql(data):
    print('-' * 100)
    print('导入msyql ...')
    conn = mysql_connect()

    with conn.cursor() as cursor:
        fea_format = ['%({})s'.format(f) for f in list(fea_to_from.keys())]
        fea_keys = ','.join(list(fea_to_from.keys()))
        fea_keys = fea_keys.replace('change', '`change`')

        sql = '''
                    INSERT INTO trade_daily2({}) VALUES ({})
                '''.format(fea_keys, ','.join(fea_format))

        k = len(data) // batch_size if len(data) % batch_size == 0 else len(data) // batch_size + 1
        for i in range(k):
            try:
                if i == 0:
                    print(cursor.mogrify(sql, data[0]))
                cursor.executemany(sql, data[i * batch_size: min((i + 1) * batch_size, len(data))])
                conn.commit()
            except Exception as e:
                conn.rollback()
                raise Exception('write to mysql error: {}'.format(e))
    conn.close()
    print('写入完成!!!')

def main(args):
    try:
        t = time.time()
        # 1、股票集合
        ts_codes = pro.stock_basic()['ts_code'].values.tolist()
        print('-' * 100)
        print('ts_codes len：{}'.format(len(ts_codes)))
        if len(ts_codes) == 0:
            raise Exception('没有股票数据 ！！！')

        # 2、获取交易数据
        data = request_from_tushare(ts_codes)
        print('数据请求耗时：{}s'.format(round(time.time()-t, 4)))
        t = time.time()
        print('data len: {}'.format(len(data)))
        if len(data) == 0:
            raise Exception('tushare 数据请求失败 ！！！')

        # 3、写入mysql
        write_to_mysql(data)
        print('数据写入耗时：{}s'.format(round(time.time() - t, 4)))
    except:
        error_info = traceback.format_exc()
        send_email('Data: trade_daily', error_info)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_date', type=str, default='2025-08-15')
    parser.add_argument('--end_date', type=str, default='2025-08-15')
    args = parser.parse_args()
    print(args)
    if not is_trade_day(args.end_date):
        print('不是交易日，退出！！！')
        exit(0)
    t = time.time()
    main(args)
    print('总耗时：{}s'.format(round(time.time() - t, 4)))
