'''
    将mysql中数据处理成qlib格式数据
'''
import os
import time
import fire
import pandas as pd
from utils.utils import tushare_pro
from utils.utils import sql_engine
from utils.utils import is_trade_day
from datetime import datetime
pro = tushare_pro()

class Processor:
    def __init__(self,
                 provider_uri: str = '~/.qlib/qlib_data/custom_data_hfq_tmp',
                 path_in: str = None,            # 离线数据路径
                 start_date: str = None,         # 数据起始日期
                 end_date: str = None,           # 数据终止日期
                 is_offline: bool = False,       # 是否处理批量导出的离线数据
                 index_code: str = '000300.SH',  # 指数code
                 fq: str = 'hfq',                 # 复权方式。hfq: 后复权; qfq: 前复权; None: 不复权
                 columns = None
                 ):
        if columns is None:
            columns = ['ts_code', 'date', 'open', 'close', 'high', 'low', 'volume', 'amount', 'change', 'factor']
        self.provider_uri = provider_uri
        self.path_in = path_in
        self.start_date = start_date
        self.end_date = end_date
        self.is_offline = is_offline
        self.index_code = index_code
        self.fq = fq
        self.columns = columns

        if self.start_date is None or self.end_date is None or self.start_date > self.end_date:
            raise ValueError('start_date and end_date cannot be None, start_date must be less then or equal to end_date')
        if self.is_offline and self.path_in is None:
            raise ValueError("When is_offline is True, path_in cannot be None")

    def load_data(self):
        print('load_data ...')
        ''' 加载股票数据 '''
        if self.is_offline:
            df = pd.read_csv(self.path_in, sep='\t')
        else:
            engine = sql_engine()
            sql = '''
            select ts_code, day as date, open, close, high, low, vol, amount, `change`, adj_factor 
            from cf_quant.trade_daily2 where day>='{}' and day<='{}';
            '''.format(self.start_date, self.end_date)
            print('{}\n{}\n{}'.format('-' * 50, sql, '-' * 50))
            df = pd.read_sql(sql, engine)
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
            df.to_csv(os.path.join(self.provider_uri, 'custom_{}_{}.csv'.format(self.start_date, self.end_date)), sep='\t', index=False)
        return df

    def load_data_index(self):
        print('load_data_index ...')
        ''' 加载指数数据 '''
        start_date = datetime.strptime(self.start_date, '%Y-%m-%d').strftime('%Y%m%d')
        end_date = datetime.strptime(self.end_date, '%Y-%m-%d').strftime('%Y%m%d')
        index_df = pro.index_daily(ts_code=self.index_code, start_date=start_date, end_date=end_date)
        index_df.drop(columns=['pct_chg', 'pre_close'], inplace=True)
        index_df.columns = ['ts_code', 'date', 'close', 'open', 'high', 'low', 'change', 'vol', 'amount']
        index_df['date'] = pd.to_datetime(index_df['date'], format='%Y%m%d').dt.strftime('%Y-%m-%d')
        index_df['adj_factor'] = 1
        return index_df

    def trans_fq(self, df):
        print('trans_hfq ...')
        ''' 复权计算 '''
        def hfq(group):
            ''' 后复权 '''
            # 最近交易日期因子
            filtered_group = group[group['date'] == self.end_date]
            if filtered_group.empty:
                return None
            # 对open, close, high, low复权
            group['open'] = group['open'] * group['factor']
            group['close'] = group['close'] * group['factor']
            group['high'] = group['high'] * group['factor']
            group['low'] = group['low'] * group['factor']
            group['factor'] = group['factor']
            return group

        def qfq(group):
            ''' 前复权 '''
            # 最近交易日期因子
            filtered_group = group[group['date'] == self.end_date]
            if filtered_group.empty:
                return None
            now_factor = filtered_group['factor'].iloc[0]
            # 对open, close, high, low复权
            group['open'] = group['open'] * group['factor'] / now_factor
            group['close'] = group['close'] * group['factor'] / now_factor
            group['high'] = group['high'] * group['factor'] / now_factor
            group['low'] = group['low'] * group['factor'] / now_factor
            group['factor'] = group['factor'] / now_factor
            return group

        if self.fq == 'hfq':
            df = df.groupby(by='ts_code').apply(hfq)
        elif self.fq == 'qfq':
            df = df.groupby(by='ts_code').apply(qfq)
        return df

    def split_stock_data(self, df):
        print('split_stock_data ...')
        """ 按股票代码分割CSV数据 """
        output_dir = os.path.join(self.provider_uri, 'out_{}_{}'.format(self.start_date, self.end_date))
        output_dir = os.path.expanduser(output_dir)
        print(output_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # 按股票代码分组并保存
        for ts_code, group in df.groupby('ts_code'):
            ts_code = '{}{}'.format(ts_code.split('.')[1], ts_code.split('.')[0])
            output_file = os.path.join(output_dir, f"{ts_code}.csv")
            group.to_csv(output_file, index=False, columns=self.columns[1:])
            print(f"生成: {output_file} ({len(group)}行)")

    def main(self):
        if self.start_date == self.end_date and not is_trade_day(self.start_date):
            print('not trade day, exit !')
            exit(0)

        df = self.load_data()
        # index_df = self.load_data_index()
        # if df.empty or index_df.empty:
        #     raise ValueError("DataFrames df and index_df cannot be empty")
        # merged = pd.concat([df, index_df], axis=0, ignore_index=True)
        merged = df
        merged.columns = self.columns
        merged_fq = self.trans_fq(merged)
        merged_fq = merged_fq.round({'open': 2, 'close': 2, 'high': 2, 'low': 2, 'change': 2, 'factor': 4})
        merged_fq.to_csv(os.path.join(self.provider_uri, 'hfq_{}_{}.csv'.format(self.start_date, self.end_date)), index=False)
        merged_fq.reset_index(drop=True, inplace=True)
        self.split_stock_data(merged_fq)

if __name__ == '__main__':
    '''
        批量离线: python process.py main --start_date 2025-01-01 --end_date 2025-08-01 \
            --is_offline True \
            --path_in ~/.qlib/qlib_data/custom_data_hfq_tmp/custom_2025-01-01_2025-08-01.csv
        每日更新：python process.py main --start_date 2025-08-01 --end_date 2025-08-01
    '''
    dt = time.time()
    fire.Fire(Processor)
    print('耗时：{} s'.format(round(time.time() - dt, 4)))
