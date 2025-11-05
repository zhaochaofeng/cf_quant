'''
    功能：检查stock_info_ts 与 stock_info_bao数据是否一致
'''
import time
import fire
from datetime import datetime
import pandas as pd
from utils.utils import sql_engine
from utils.utils import is_trade_day
from utils.utils import send_email

engine = sql_engine()
fields = ['qlib_code', 'list_date', 'status', 'exchange']

def format_email_info(df_ts, df_bao):
    diff = df_ts.eq(df_bao)
    mask_ne = (diff != True).any(axis=1)  # 不相等
    index_ne = diff.index[mask_ne]

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
    if date == None:
        date = datetime.now().strftime('%Y-%m-%d')
    if not is_trade_day(date):
        print('不是交易日！')
        exit(0)

    sql_ts = '''
        select {} from stock_info_ts where exchange in ('SSE', 'SZSE') and day='{}' and list_date<='{}';
    '''.format(', '.join(fields), date, date)
    sql_bao = '''
        select {} from stock_info_bao where exchange in ('SSE', 'SZSE') and day='{}' and list_date<='{}';
    '''.format(', '.join(fields), date, date)

    print('{}\n{}\n{}\n{}'.format('-' * 50, sql_ts, sql_bao, '-' * 50))
    df_ts = pd.read_sql(sql_ts, engine)
    df_bao = pd.read_sql(sql_bao, engine)

    if df_ts.empty or df_bao.empty:
        send_email('Check: stock_info_ts & stock_info_bao', 'df_ts or df_bao is Empty！')
        exit(1)

    df_ts = df_ts.set_index(keys=['qlib_code']).sort_index()
    df_bao = df_bao.set_index(keys=['qlib_code']).sort_index()
    print(df_ts.head())
    print(df_bao.head())
    print(df_ts.shape)
    print(df_bao.shape)

    res = format_email_info(df_ts, df_bao)
    if len(res) > 0:
        send_email("Check: stock_info_ts & stock_info_bao", '\n'.join(res))
    else:
        print('检查没有异常 ！！！')

if __name__ == '__main__':
    t = time.time()
    fire.Fire(main)
    print('耗时：{}s'.format(round(time.time()-t, 4)))
