'''
    功能：检查trade_daily_ts 与 trade_daily_bao数据是否一致
'''
import time
import fire
from datetime import datetime
import pandas as pd
from utils.utils import sql_engine
from utils.utils import is_trade_day
from utils.utils import send_email

engine = sql_engine()
fields = ['qlib_code', 'open', 'close', 'high', 'low', 'vol', 'amount']

def format_email_info(df_ts, df_bao):
    diff = df_ts.ne(df_bao)
    mask = diff.any(axis=1)
    if mask.sum() == 0:
        return []

    index_ne = diff.index[mask]

    res = []
    for index in index_ne:
        ts_f = []
        bao_f = []
        try:
            ts_row = df_ts.loc[index]
            for f in fields[1:]:
                ts_f.append('{}:{}'.format(f, ts_row[f]))
        except:
            ts_f = ['NaN']

        try:
            bao_row = df_bao.loc[index]
            for f in fields[1:]:
                bao_f.append('{}:{}'.format(f, bao_row[f]))
        except:
            bao_f = ['NaN']
        res.append('{}: [ts: {};  bao: {}]'.format(index, ', '.join(ts_f), ', '.join(bao_f)))
    return res

def main(date: str=None):
    today = datetime.now().strftime('%Y-%m-%d')
    if date == None:
        date = today
    if not is_trade_day(date):
        print('不是交易日！')
        exit(0)

    sql_ts = '''
        select {} from
        (select qlib_code as qlib from stock_info_ts where exchange in ('SSE', 'SZSE') and day='{}')a
        JOIN
        (select {} from trade_daily_ts where day='{}')b
        ON
        a.qlib = b.qlib_code;
    '''.format(','.join(fields), today, ','.join(fields), date)

    sql_bao = '''
        select {} from
        (select qlib_code as qlib from stock_info_bao where exchange in ('SSE', 'SZSE') and day='{}')a
        JOIN
        (select {} from trade_daily_bao where day='{}')b
        ON
        a.qlib = b.qlib_code;
    '''.format(','.join(fields), today, ','.join(fields), date)

    print('{}\n{}\n{}\n{}'.format('-' * 50, sql_ts, sql_bao, '-' * 50))
    df_ts = pd.read_sql(sql_ts, engine)
    df_bao = pd.read_sql(sql_bao, engine)

    if df_ts.empty or df_bao.empty:
        send_email('Check: trade_daily_ts & trade_daily_bao', 'df_ts or df_bao is Empty！')
        exit(1)

    df_ts = df_ts.set_index(keys=['qlib_code']).sort_index()
    df_bao = df_bao.set_index(keys=['qlib_code']).sort_index()
    print(df_ts.head())
    print(df_bao.head())
    print(df_ts.shape)
    print(df_bao.shape)

    res = format_email_info(df_ts, df_bao)
    if len(res) > 0:
        send_email("Data:trade_daily:check", '\n'.join(res))
    else:
        print('检查没有异常 ！！！')

if __name__ == '__main__':
    t = time.time()
    fire.Fire(main)
    print('耗时：{}s'.format(round(time.time()-t, 4)))
