'''
    因子构建
'''
import time
import pandas as pd
import qlib
from qlib.constant import REG_CN
from qlib.data import D
from qlib.data.filter import NameDFilter
from utils import MySQLDB
import jqfactor_analyzer as ja

from data.factor.factor_func import (
    MACD, BOLL, RSI_Multi, KDJ, DMI, WR, BIAS_Multi, CCI, ROC
)
#  DMI
factor_class = {
    '技术指标': [MACD, BOLL, KDJ, WR, BIAS_Multi, CCI, ROC]
}

factor_exp = {
    'label': {'exp': 'Ref($close, -2)/Ref($close, -1) -1', 'name': 'label', 'class': 'Base'},
    'industry': {'exp': '$industry', 'name': '行业', 'class': 'Base'},
    'MA5': {'exp': 'Mean($close, 5)', 'name': '5日均线', 'class': '技术指标'}
}

for c, funs in factor_class.items():
    clazz = c
    for f in funs:
        for k, v in f().items():
            v['class'] = clazz
            factor_exp[k] = v

# for k, v in factor_exp.items():
#     print(k, v)


class Factor:
    # 设置一个 factor_list 变量，用来对新增因子评价指标补数
    def __init__(self,
                 start_date: str,
                 end_date: str,
                 provider_uri='~/.qlib/qlib_data/cn_data',
                 market='csi300',
                 ):
        ''' 初始化 '''
        self.start_date = start_date
        self.end_date = end_date
        self.market = market
        self.indicator = {}
        qlib.init(provider_uri=provider_uri, region=REG_CN)

        self.factor_list = list(factor_exp.keys())
        self.base_fields = ['label', 'industry']
        for c in self.base_fields:
            self.factor_list.remove(c)
        if market == 'zb':   # 主版
            # FIX: 按名称过滤很耗时
            nameFilter = NameDFilter(name_rule_re='^[A-Za-z]{2}(60|00)')
            self.instruments = D.instruments(market='all', filter_pipe=[nameFilter])
        else:
            self.instruments = D.instruments(market=self.market)

    def build_factor(self, is_winsorize=True, is_industry=True):
        '''
        构建因子
            Arg:
                is_industry: 是否进行行业中性化
        '''

        exps = [factor_exp[f]['exp'] for f in self.factor_list+self.base_fields]
        factor_df = D.features(instruments=self.instruments, fields=exps,
                               start_time=self.start_date, end_time=self.end_date)
        factor_df.columns = self.factor_list + self.base_fields

        # 去极值 (包括label)
        if is_winsorize:
            factor_df = ja.winsorize(factor_df, scale=3, axis=0, inclusive=True)
        # 行业中性化
        if is_industry:
            factor_df[self.factor_list] = factor_df[self.factor_list + ['industry']].\
                groupby('industry', group_keys=False).apply(lambda x: x-x.mean())[self.factor_list]
        return factor_df

    def calc_ic(self, pred: pd.Series, label: pd.Series, date_col="datetime", dropna=False) -> (pd.Series, pd.Series):
        """calc_ic.

        Parameters
        ----------
        pred :
            pred
        label :
            label
        date_col :
            date_col

        Returns
        -------
        (pd.Series, pd.Series)
            ic and rank ic
        """
        df = pd.DataFrame({"pred": pred, "label": label})
        ic = df.groupby(date_col, group_keys=False).apply(lambda df: df["pred"].corr(df["label"]))
        ric = df.groupby(date_col, group_keys=False).apply(lambda df: df["pred"].corr(df["label"], method="spearman"))

        if dropna:
            return ic.dropna(), ric.dropna()
        else:
            return ic, ric

    def calc_indicator(self, factor_df) -> None:
        ''' 计算指标 '''
        for factor in self.factor_list:
            ic, ric = self.calc_ic(pred=factor_df[factor], label=factor_df['label'], dropna=True)
            self.indicator[factor] = pd.DataFrame({'ic': ic, 'ric': ric})

    def write_indicator_to_mysql(self):
        ''' 写入指标 '''
        data = []
        for factor in self.factor_list:
            indicator = self.indicator[factor]
            for date, row in indicator.iterrows():
                ic, ric = row['ic'],  row['ric']
                if pd.isna(ic):
                    ic = None
                if pd.isna(ric):
                    ric = None
                tmp = {'name': factor_exp[factor]['name'],
                       'code': factor,
                       'day': date,
                       'class': factor_exp[factor]['class'],
                       'IC': ic,
                       'RIC': ric
                }
                data.append(tmp)
        sql = """ 
            INSERT INTO factor_eval (name, code, day, class, IC, RIC)
                  VALUES (%(name)s, %(code)s, %(day)s, %(class)s, %(IC)s, %(RIC)s) 
            """
        with MySQLDB() as db:
            db.executemany(sql, data)

if __name__ == '__main__':
    t = time.time()
    factor = Factor(
        provider_uri='~/.qlib/qlib_data/custom_data_hfq_tmp',
        market='all',  # zb
        start_date='2025-01-01', end_date='2025-10-30',
    )
    factor_df = factor.build_factor()
    factor.calc_indicator(factor_df)
    factor.write_indicator_to_mysql()

    print('耗时： {}s'.format(round(time.time() - t, 4)))
