'''
    功能：将mysql中数据处理成qlib格式数据
    描述：
        3个月以上数据：先在服务器上导出成文件形式，然后通过脚本处理
        3个月以内数据：直接从mysql表中查询，也就是指定起止日期，执行脚本即可
'''

import os
import time
import fire
import shutil
import pandas as pd
import traceback
from datetime import datetime
from utils.utils import tushare_pro
from utils.utils import sql_engine
from utils.utils import is_trade_day
from utils.utils import send_email

pro = tushare_pro()

class Processor:
    def __init__(self,
                 provider_uri: str = '~/.qlib/qlib_data/custom_data_hfq',
                 path_in: str = None,            # 离线数据路径
                 start_date: str = None,         # 数据起始日期
                 end_date: str = None,           # 数据终止日期
                 is_offline: bool = False,       # 是否处理批量导出的离线数据
                 index_list: list = None,        # 指数code
                 fq: str = 'hfq',                # 复权方式。hfq: 后复权; qfq: 前复权; None: 不复权
                 columns: list = None
                 ):
        if columns is None:
            columns = ['ts_code', 'date', 'open', 'close', 'high', 'low', 'volume', 'amount', 'factor', 'change']
        if index_list is None:
            index_list = ['000300.SH', '000905.SH', '000903.SH']
        self.provider_uri = provider_uri
        self.path_in = path_in
        self.start_date = start_date
        self.end_date = end_date
        self.is_offline = is_offline
        self.index_list = index_list
        self.fq = fq
        self.columns = columns

        if self.start_date is None or self.end_date is None or self.start_date > self.end_date:
            raise ValueError('start_date and end_date cannot be None, start_date must be less then or equal to end_date')
        if self.is_offline and self.path_in is None:
            raise ValueError("When is_offline is True, path_in cannot be None")

    def load_data(self):
        ''' 加载股票数据 '''
        print('-' * 100)
        print('load_data ...')
        if self.is_offline:
            df = pd.read_csv(self.path_in, sep='\t')
        else:
            engine = sql_engine()
            sql = '''
            select ts_code, day as date, open, close, high, low, vol, amount, adj_factor 
            from trade_daily_ts where day>='{}' and day<='{}';
            '''.format(self.start_date, self.end_date)
            print('{}\n{}\n{}'.format('-' * 50, sql, '-' * 50))
            df = pd.read_sql(sql, engine)
            # 将date字段从datetime.date转化为str格式
            df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d').dt.strftime('%Y-%m-%d')
            df.to_csv(os.path.join(self.provider_uri, 'custom_{}_{}.csv'.format(self.start_date, self.end_date)), sep='\t', index=False)

        return df

    def load_data_index(self, codes):
        ''' 加载指数数据 '''
        print('-' * 100)
        print('load_data_index ...')
        start_date = datetime.strptime(self.start_date, '%Y-%m-%d').strftime('%Y%m%d')
        end_date = datetime.strptime(self.end_date, '%Y-%m-%d').strftime('%Y%m%d')

        index_df = pd.DataFrame()
        for code in codes:
            try:
                print('index code: {}'.format(code))
                tmp_df = pro.index_daily(ts_code=code, start_date=start_date, end_date=end_date)
                tmp_df.drop(columns=['pct_chg', 'pre_close', 'change'], inplace=True)
                tmp_df.columns = ['ts_code', 'date', 'close', 'open', 'high', 'low', 'vol', 'amount']
                tmp_df['date'] = pd.to_datetime(tmp_df['date'], format='%Y%m%d').dt.strftime('%Y-%m-%d')
                tmp_df['adj_factor'] = 1
                index_df = pd.concat([index_df, tmp_df], axis=0, ignore_index=True)
            except Exception as e:
                raise ValueError(f'从 tushare 中请求指数数据失败：{e}')
        return index_df

    def trans_fq(self, df):
        ''' 复权计算 '''
        print('trans_hfq ...')
        def hfq(group):
            ''' 后复权 '''
            # 对open, close, high, low, vol 复权
            group['open'] = group['open'] * group['adj_factor']
            group['close'] = group['close'] * group['adj_factor']
            group['high'] = group['high'] * group['adj_factor']
            group['low'] = group['low'] * group['adj_factor']
            group['vol'] = (group['vol'] / group['adj_factor'])
            return group

        def qfq(group):
            ''' 前复权 '''
            filtered_group = group[group['date'] == self.end_date]
            if filtered_group.empty:
                return None
            now_factor = filtered_group['adj_factor'].iloc[0]
            # 对open, close, high, low复权
            group['open'] = group['open'] * group['adj_factor'] / now_factor
            group['close'] = group['close'] * group['adj_factor'] / now_factor
            group['high'] = group['high'] * group['adj_factor'] / now_factor
            group['low'] = group['low'] * group['adj_factor'] / now_factor
            group['adj_factor'] = group['adj_factor'] / now_factor
            group['vol'] = group['vol'] * now_factor / group['adj_factor']
            return group

        if self.fq == 'hfq':
            df = df.groupby(by='ts_code', group_keys=False).apply(hfq)
        elif self.fq == 'qfq':
            df = df.groupby(by='ts_code', group_keys=False).apply(qfq)
        return df

    def split_stock_data(self, df):
        print('split_stock_data ...')
        """ 按股票代码分割成单独的CSV文件 """
        output_dir = os.path.join(self.provider_uri, 'out_{}_{}'.format(self.start_date, self.end_date))
        output_dir = os.path.expanduser(output_dir)
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        # 按股票代码分组并保存
        for ts_code, group in df.groupby('ts_code'):
            code = '{}{}'.format(ts_code.split('.')[1], ts_code.split('.')[0])
            output_file = os.path.join(output_dir, f"{code}.csv")
            group.to_csv(output_file, index=False, columns=self.columns[1:])
            print(f"生成: {output_file} ({len(group)}行)")

    def main(self):
        try:
            if not is_trade_day(self.end_date):
                print('end_date must be trade day: {}'.format(self.end_date))
                return

            df = self.load_data()
            index_df = self.load_data_index(self.index_list)
            if df.empty or index_df.empty:
                raise ValueError("DataFrames df and index_df cannot be empty")
            merged = pd.concat([df, index_df], axis=0, ignore_index=True)
            merged_fq = self.trans_fq(merged)
            # merged_fq.reset_index(drop=True, inplace=True)

            # 单位转化
            merged_fq['vol'] = merged_fq['vol'] * 100  # 单位股
            merged_fq['amount'] = merged_fq['amount'] * 1000  # 单位元

            # change字段
            def get_change(group):
                group.sort_values(by='date', inplace=True)
                group['change'] = (group['close'] - group['close'].shift(1)) / group['close'].shift(1)
                return group
            merged_fq = merged_fq.groupby(by='ts_code', group_keys=False).apply(get_change)
            merged_fq.columns = self.columns
            merged_fq = merged_fq.round({'open': 2, 'close': 2, 'high': 2, 'low': 2, 'volume': 2, 'amount': 3, 'change': 4, 'factor': 4})
            # 保存中间数据
            # merged_fq.to_csv(os.path.join(self.provider_uri, 'hfq_{}_{}.csv'.format(self.start_date, self.end_date)), index=False)
            print(merged_fq.head())
            self.split_stock_data(merged_fq)
        except Exception as e:
            error_info = traceback.format_exc()
            send_email('Data:qlib_online:process', error_info)

if __name__ == '__main__':
    '''
        批量离线: 
        python process.py main 
            --start_date 2025-01-01 \
            --end_date 2025-08-01 \
            --is_offline True \
            --path_in ~/.qlib/qlib_data/custom_data_hfq_tmp/custom_2025-01-01_2025-08-01.csv

        每日更新：
        python process.py main --start_date 2025-08-01 --end_date 2025-08-01
    '''
    dt = time.time()
    fire.Fire(Processor)
    print('耗时：{} s'.format(round(time.time() - dt, 4)))
