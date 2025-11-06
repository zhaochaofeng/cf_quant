'''
    功能：BaoStock表导入mysql
    描述：日级交易数据（非复权）+ 复权因子
    刷数：设置[start_date, end_date] (单code请求)
'''

import time
import argparse
import pandas as pd
from datetime import datetime
from utils.utils import mysql_connect, sql_engine
from utils.utils import is_trade_day
from utils.utils import send_email
from utils.utils import bao_stock_connect
import traceback

import warnings
warnings.filterwarnings("ignore")
bs = bao_stock_connect()
batch_size = 10000    # 每次写入mysql的行数

fea_1 = {
    'code': 'code',
    'day': 'date',
    'open': 'open',
    'close': 'close',
    'high': 'high',
    'low': 'low',
    'pre_close': 'preclose',
    'pct_chg': 'pctChg',
    'vol': 'volume',
    'amount': 'amount',
    'is_st': 'isST',
    'adj_factor': 'adj_factor'
}
fea_2 = ['qlib_code']

round_dic = {
    'open': 2,
    'close': 2,
    'high': 2,
    'low': 2,
    'pre_close': 2,
    'pct_chg': 2,
    'vol': 2,
    'amount': 3,
    'is_st': 0,
    'adj_factor': 4
}

def parse_line(row, fea):
    ''' 解析数据 '''
    tmp = {}
    for f in fea.keys():
        try:
            v = row[fea[f]]
            if f == 'code':
                suffix, code = v.split('.')
                tmp['qlib_code'] = '{}{}'.format(suffix.upper(), code)
            elif f == 'pct_chg':
                # 处理pct_chg超出范围的情况
                if pd.isna(v):
                    v = None
                elif abs(v) > 9999.99:
                    print(f"警告: pct_chg值 {v} 超出范围，已截断为9999.99或-9999.99")
                    print(row)
                    v = 9999.99 if v > 0 else -9999.99
            elif f == 'vol':
                v = v / 100   # 股转化为手
            elif f == 'amount':
                v = v / 1000  # 元转为千元
            if pd.isna(v):
                v = None
            tmp[f] = v
        except Exception as e:
            raise Exception('parse_line error: {}'.format(e))
    return tmp

def get_factor(code, start_date, end_date):
    """
    获取指定股票在指定日期范围内每天的后复权因子

    参数:
    code (str): 股票代码，格式如"sh.600000"
    start_date (str): 起始日期，格式为"YYYY-MM-DD"
    end_date (str): 终止日期，格式为"YYYY-MM-DD"

    返回:
    pd.DataFrame: 包含code, date, adj_factor字段的DataFrame，包含日期范围内每一天的数据
    """

    try:
        # 查询指定股票的除权除息数据（包含后复权因子）
        rs = bs.query_adjust_factor(
            code=code,
            start_date="1990-01-01",  # 从较早日期开始查询
            end_date=end_date
        )

        if rs.error_code != '0':
            raise Exception(f"获取{code}复权因子失败: {rs.error_msg}")

        # 生成指定日期范围内的所有日期
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        full_dates = pd.DataFrame({'date': date_range})

        # 获取除权除息日数据
        df = rs.get_data()
        if df.empty:
            print(f"未获取到{code}的除权除息数据")
            result_df = full_dates.copy()
            result_df['code'] = code
            result_df['adj_factor'] = 1.0
            return result_df[['code', 'date', 'adj_factor']]

        # print(df[['code', 'dividOperateDate', 'backAdjustFactor']])

        # 筛选出需要的后复权因子列并重命名
        df = df[['code', 'dividOperateDate', 'backAdjustFactor']].rename(
            columns={'dividOperateDate': 'date', 'backAdjustFactor': 'adj_factor'}
        )

        # 转换日期为datetime类型，因子为数值类型
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
        df['adj_factor'] = pd.to_numeric(df['adj_factor'], errors='coerce')

        # 关键修复：找到起始日期前最近的除权除息日因子值
        start_dt = pd.to_datetime(start_date)
        # start_date这一天不是除权除息日
        if df[df['date'] == start_dt].empty:
            # 筛选出在起始日期之前的factor
            before_start = df[df['date'] < start_dt]
            if not before_start.empty:
                # 取起始日期前最近的一个因子值
                latest_before_start = before_start.sort_values('date', ascending=False).iloc[0]
                # 将该因子值添加到合并数据中
                pre_start_row = pd.DataFrame({
                    'date': [start_dt],
                    'adj_factor': [latest_before_start['adj_factor']]
                })
            else:
                pre_start_row = pd.DataFrame({
                    'date': [start_dt],
                    'adj_factor': [1.0]
                })
                # 合并到原始除权除息数据中
            df = pd.concat([df, pre_start_row], ignore_index=True)

        # 将除权除息日数据与完整日期序列合并
        merged_df = pd.merge(full_dates, df[['date', 'adj_factor']], on='date', how='left')

        # 调试：检查合并后的数据
        # print(f"合并后的数据样本:")
        # print(merged_df[merged_df['adj_factor'].notna()])

        # 向前填充因子值（关键修复点：确保填充逻辑正确）
        merged_df['adj_factor'] = merged_df['adj_factor'].ffill()

        # 添加股票代码列
        merged_df['code'] = code

        # 调整列顺序并转换日期为字符串格式
        result_df = merged_df[['code', 'date', 'adj_factor']]
        result_df['date'] = result_df['date'].dt.strftime('%Y-%m-%d')

        return result_df

    except Exception as e:
        raise Exception(f"发生异常: {str(e)}")

def get_stocks():
    print('-' * 100)
    print('get_stocks...')
    engine = sql_engine()
    today = datetime.now().strftime('%Y-%m-%d')
    sql = '''
        select code from stock_info_bao where day='{}';
    '''.format(today)
    print(sql)
    df = pd.read_sql(sql, engine)
    if df.empty:
        raise Exception('get_stocks failed: {}'.format(today))
    codes = df['code'].values.tolist()
    print('stocks len: {}'.format(len(codes)))
    return codes

def request_from_baostock(codes):
    # 从tushare API获取数据
    print('-' * 100)
    print('从BaoStock API获取数据...')
    data = []

    # feas = list(fea_1.values())
    fea_bao = list(fea_1.values())
    fea_bao.remove('adj_factor')

    for i, code in enumerate(codes):
        if (i + 1) % 100 == 0:
            print('requested num: {}'.format(i + 1))
        rs = bs.query_history_k_data_plus(code,
                                          "{}".format(','.join(fea_bao)),
                                          start_date=args.start_date, end_date=args.end_date,
                                          frequency="d", adjustflag="3")

        df = rs.get_data()
        if df.empty:
            continue
        df = df[~(df['amount'] == '')]   # BaoStock在停牌日也能请求到数据，volume/amount为'', 需要排除
        if df.empty:
            continue
        factor = get_factor(code, args.start_date, args.end_date)
        if factor.empty:
            continue

        df.set_index(keys=['code', 'date'], inplace=True)
        factor.set_index(keys=['code', 'date'], inplace=True)
        merged = pd.concat([df, factor], axis=1, join='inner')
        for f in merged.columns:
            merged[f] = pd.to_numeric(merged[f], errors='coerce')
        merged.reset_index(inplace=True)

        merged = merged.round(round_dic)
        for index, row in merged.iterrows():
            tmp = parse_line(row, fea_1)
            data.append(tmp)
    return data

def write_to_mysql(data):
    print('-' * 100)
    print('导入msyql ...')
    conn = mysql_connect()

    with conn.cursor() as cursor:
        fea_merge = list(fea_1.keys()) + fea_2
        fea_merge_format = ['%({})s'.format(f) for f in fea_merge]

        sql = '''
                    INSERT INTO trade_daily_bao({}) VALUES ({})
                '''.format(','.join(fea_merge), ','.join(fea_merge_format))

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

def main():
    try:
        t = time.time()
        # 1、股票集合
        codes = get_stocks()
        # codes = codes[0:10]
        # codes = ['sh.603418']
        # 2、获取交易数据
        data = request_from_baostock(codes)
        print('数据请求耗时：{}s'.format(round(time.time()-t, 4)))
        t = time.time()
        print('data len: {}'.format(len(data)))
        if len(data) == 0:
            raise Exception('BaoStack 数据请求失败 ！！！')

        # 3、写入mysql
        write_to_mysql(data)
        print('数据写入耗时：{}s'.format(round(time.time() - t, 4)))
    except:
        error_info = traceback.format_exc()
        send_email('Data: trade_daily_bao', error_info)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_date', type=str, default='2025-09-23')
    parser.add_argument('--end_date', type=str, default='2025-09-23')
    args = parser.parse_args()
    print(args)
    if not is_trade_day(args.end_date):
        print('不是交易日，退出！！！')
        exit(0)
    t = time.time()
    main()
    bs.logout()
    print('总耗时：{}s'.format(round(time.time() - t, 4)))