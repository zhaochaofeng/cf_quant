''''
    量化策略：因子分析 + 线性回归
'''

import sys
import time
import argparse
import redis
import pandas as pd
import numpy as np
from jqdatasdk import *
from statsmodels.multivariate.factor import Factor
from jqfactor_analyzer import winsorize, standardlize

from utils.utils import get_config
from utils.utils import sql_engine
from utils.utils import tushare_pro
from utils.utils import get_n_pretrade_day, is_trade_day

def get_fea_data(start_date: str, end_date: str) -> pd.DataFrame:
    """
    从MySQL获取特征数据
    
    参数:
        start_date: 开始日期 (格式: 'YYYY-MM-DD')
        end_date: 结束日期 (格式: 'YYYY-MM-DD')

    返回:
        DataFrame with columns: ['date', 'code', '<特征>...']
    """
    fea_list = ['turnover_rate', 'turnover_rate_f', 'volume_ratio', 'pe', 'pe_ttm',
                  'pb', 'ps', 'ps_ttm', 'dv_ratio', 'dv_ttm', 'total_share', 'float_share',
                  'free_share', 'total_mv', 'circ_mv'
                  ]

    engine = sql_engine()

    sql = '''
        select day as date, ts_code as code, {} from valuation_tushare 
        where day>='{}' AND day<='{}' order by day, ts_code;
    '''.format(','.join(fea_list), start_date, end_date)
    print(sql)
    # 读取数据
    df = pd.read_sql(sql, engine)

    # 关闭连接
    engine.dispose()
    return df

def get_return_data(start_date, end_date):
    ''' 收益率 '''
    auth(config['joinqaunt']['username'], config['joinqaunt']['password'])

    stocks_info = get_all_securities(types=['stock'], date=start_date)
    stocks = stocks_info.index.tolist()
    print('股票数：{}'.format(len(stocks)))
    df = get_price(stocks, start_date=start_date, end_date=end_date, frequency='1d', fields=['close'])

    def calculate_return(group):
        # group 为DataFrame
        group = group.sort_values("time")
        initial_price = group["close"].iloc[0]
        final_price = group["close"].iloc[1]
        return (final_price - initial_price) / initial_price * 100
    s = df.groupby("code").apply(calculate_return)
    df = s.reset_index(name="return")
    df.set_index(['code'], inplace=True)
    return df

def get_close_data(start_date, end_date):
    ''' 收盘价数据 '''
    engine = sql_engine()
    sql = '''
            select day as date, ts_code as code, close from trade_daily 
            where day>='{}' AND day<='{}' order by day, ts_code;
        '''.format(start_date, end_date)
    print(sql)
    df = pd.read_sql(sql, engine)
    engine.dispose()
    return df

def fa_model(endog, n_factor, index):
    factor_model = Factor(endog=endog, n_factor=n_factor)
    result = factor_model.fit()

    # 因子载荷矩阵
    loadings = result.loadings
    loadings = pd.DataFrame(loadings, index=index)
    print('{}: {}'.format('因子载荷矩阵', '-'*100))
    print(loadings)
    # 特征值
    eigen = np.array(result.eigenvals)
    print('{}: {}'.format('特征值', '-'*100))
    print(eigen)

    # 因子旋转
    result.rotate(method='varimax')
    loadings = result.loadings
    loadings = pd.DataFrame(loadings, index=index)
    print('{}: {}'.format('旋转因子载荷矩阵', '-' * 100))
    print(loadings)

    return result

def fa_predict(result, endog, factor_name):
    # 原始特征值转化为因子值
    factor_score = result.factor_scoring(endog=endog, transform=True)

    factor_score = pd.DataFrame(factor_score)
    factor_score.columns = factor_name
    factor_score.set_index(endog.index, inplace=True)
    print('{}: {}'.format('因子得分', '-' * 100))
    print(factor_score.head(5))
    print(factor_score.shape)
    print(factor_score.describe())
    return factor_score

def load_to_redis(predict, day):
    # 入库redis
    r = redis.Redis(host=config['redis']['host'])
    dic = {}
    codes = predict.index.tolist()
    scores = predict.values.tolist()
    for i in range(len(codes)):
        dic[codes[i]] = scores[i]
    key = "{}_{}".format(args.alg_name, day)
    print('key: {}'.format(key))
    r.hset(key, mapping=dic)

    if args.is_print_case:
        p = r.hgetall(key)
        p = {k.decode(): float(v.decode()) for k, v in p.items()}
        p_sorted = sorted(p.items(), key=lambda x: x[1], reverse=True)
        for i, (code, score) in enumerate(p_sorted):
            if i >= 100:
                break

            name = pro.stock_basic(ts_code=code)['name'].iloc[0]
            print('{}, {}, {}, {}'.format(i+1, code, name, round(score, 6)))

def preprocess(data, axis=0):
    # 去极值
    data = winsorize(data, scale=3, axis=axis)
    # 归一化
    data = standardlize(data, axis=axis)
    # 处理缺失值。自变量用0填充，因变量过滤为Nan的数据
    data.fillna(0, inplace=True)
    return data

def calc_ic(data, periods=(1, 5, 10), method='spearman'):
    """
    实现的IC值计算函数

    参数:
        data: 必须包含二级索引(date, code)和['factor', 'close']列
        periods: 计算周期，支持单周期或多周期

    返回:
        DataFrame。index：日期；column: period_{n}。如下：
                    peroid_1  peroid_5  peroid_10
        date
        2025-06-03 -0.055255 -0.017521  -0.061794
        2025-06-04 -0.082961 -0.029947  -0.076773
    """
    # 步骤1：参数标准化处理
    if isinstance(periods, int):
        periods = (periods,)  # 将单周期转换为元组格式

    # 步骤2：数据索引验证与重建
    if not isinstance(data.index, pd.MultiIndex) or len(data.index.names) != 2:
        data = data.reset_index().set_index(['date', 'code'])  # 强制重建二级索引

    # 步骤3：因子数据预处理
    factor_series = data['factor'].copy()  # 提取因子序列
    price_matrix = data['close'].unstack()  # 价格转换为(code×date)矩阵

    # 步骤4：IC值计算主流程
    ic_results = {}
    for period in periods:
        # 4.1 计算未来收益率
        future_prices = price_matrix.shift(-period)  # 获取未来N日价格
        returns = (future_prices / price_matrix) - 1  # 收益率公式

        # 4.2 数据对齐处理
        aligned_data = pd.concat([
            factor_series.rename('factor'),
            returns.stack().rename(f'return_{period}d')
        ], axis=1, join='inner').dropna()  # 严格对齐并去除缺失值

        # 4.3 按日期分组计算相关系数
        def calculate_corr(group):
            return group['factor'].corr(group[f'return_{period}d'], method=method)

        # 4.4 按日期分组计算IC值
        daily_ic = aligned_data.groupby(level='date').apply(calculate_corr)

        # 4.5 结果存储
        ic_results['peroid_{}'.format(period)] = daily_ic
    return pd.DataFrame(ic_results) if len(ic_results) > 0 else None

def calc_factors_ic(merged, factor_name, period):
    ic_list = []
    for col in factor_name:
        ic_data = merged[[col, 'close']]
        ic_data.columns = ['factor', 'close']
        ic = calc_ic(ic_data, periods=period)
        if ic is None:
            continue
        print('{}: {}'.format(col, ic))
        ic = ic.mean(axis=0)
        ic_list.append(ic.iloc[0])
    return ic_list

def main():
    train_dt = get_n_pretrade_day(args.date, 1)
    start_date = get_n_pretrade_day(train_dt, args.period + args.period_num - 1)
    end_date = train_dt
    print('{}\ntrain_dt: {}, start_date: {}, end_date: {}'.format('-' * 50, train_dt, start_date, end_date))

    # 加载训练数据
    print('{} {}'.format('1、加载训练数据', '-' * 50))
    X = get_fea_data(train_dt, train_dt)
    # 数据预处理
    X.drop(labels=['date'], axis=1, inplace=True)
    X.set_index(keys=['code'], inplace=True)
    # X = preprocess(X, axis=0)
    X.fillna(0, axis=0, inplace=True)
    X = winsorize(X, scale=3, axis=0)
    print(X.head())
    print("X shape: {}".format(X.shape))

    # 样本分析
    # print('-' * 100)
    # print('{} {}'.format('样本特征', '-' * 50))
    # code_sample = ['603886.SH', '001323.SZ', '300616.SZ', '002088.SZ', '603801.SH', '002884.SZ', '002293.SZ', '002572.SZ', '600729.SH', '002327.SZ']
    # X_sample = X[X.index.isin(code_sample)]
    # # 打印X_sample中所有数据，每个字段用逗号分隔
    # print(','.join(X_sample.columns))
    # for index, row in X_sample.iterrows():
    #     print(index + ',' + ','.join(map(str, row.values)))
    # print('-' * 100)

    # 因子分析
    print('{} {}'.format('2、因子分析模型', '-' * 50))
    factor_name = []
    for i in range(args.n_factor):
        factor_name.append('{}{}'.format('f', i + 1))
    results_fa = fa_model(endog=X, n_factor=args.n_factor, index=X.columns)
    factor_score = fa_predict(results_fa, X, factor_name)

    # print('-' * 100)
    # print('样本因子得分:')
    # factor_score_sample = factor_score[factor_score.index.isin(code_sample)]
    # print(','.join(factor_name))
    # for index, row in factor_score_sample.iterrows():
    #     print(index + ',' + ','.join(map(str, row.values)))
    # print('-' * 100)
    # print('样本因子得分（手动）:')
    # X_sample = (X_sample - X.mean(axis=0)) / X.std(axis=0, ddof=0)
    # B = results_fa.factor_score_params(method='bartlett')
    # B = pd.DataFrame(data=B, index=X.columns, columns=factor_name)
    # print(B)
    # print('B shape: {}'.format(B.shape))
    # print('B type：{}'.format(type(B)))
    # print('X_sample shape: {}'.format(X_sample.shape))
    # print(X_sample)
    # factor_score_sample2 = X_sample.dot(B)
    # factor_score_sample2 = pd.DataFrame(data=factor_score_sample2, index=X_sample.index, columns=factor_name)
    # print(factor_score_sample2)
    # print('-' * 100)

    # 加载多天特征数据
    print('{} {}'.format('3、多天特征数据的因子得分', '-' * 50))
    X_new = get_fea_data(start_date, end_date)
    X_new.set_index(keys=['date', 'code'], inplace=True)
    X_new.fillna(0, axis=0, inplace=True)
    X_new = winsorize(X_new, scale=3, axis=0, inclusive=True)
    factor_score_new = fa_predict(results_fa, X_new, factor_name)

    # 加载收盘价数据
    print('{} {}'.format('4、加载收盘价数据', '-' * 50))
    close = get_close_data(start_date, end_date)
    close.set_index(keys=['date', 'code'], inplace=True)
    print(close.head())

    # 合并特征和收盘价
    print('{} {}'.format('5、合并特征和收盘价', '-' * 50))
    merged = pd.concat([factor_score_new, close], axis=1, join='inner')
    print('factor_score_new shape: {}'.format(factor_score_new.shape))
    print('close shape: {}'.format(close.shape))
    print('merged shape: {}'.format(merged.shape))
    print(merged.head())

    # 计算ic值
    print('{} {}'.format('6、计算ic值', '-' * 50))
    ic_list = calc_factors_ic(merged, factor_name, period=args.period)
    print(ic_list)

    # 计算股票得分
    print('{} {}'.format('7、计算股票得分', '-' * 50))
    choose_ic_idx = []
    ic_weight = {}
    target_ic = 0.03
    for i in range(len(ic_list)):
        if abs(ic_list[i]) > target_ic:
            choose_ic_idx.append(i)
            ic_weight[factor_name[i]] = ic_list[i]
    print(ic_weight)

    predict = 0
    for i in choose_ic_idx:
        predict += ic_weight[factor_name[i]] * factor_score[factor_name[i]]

    # 加载到redis
    print('{}{}'.format('8、预测结果加载到redis', '-' * 50))
    load_to_redis(predict, args.date)

    '''
    print('-' * 100)
    print('top10 因子得分分布：')
    print(factor_score.describe())
    print(factor_score_sample.describe())
    print('-' * 100)
    print('top10 特征取值分布：')

    for index, row in X.describe().iterrows():
        print(index+"\t", ' '.join(map(str, row.values)))
    for index, row in X_sample.describe().iterrows():
        print(index + "\t", ' '.join(map(str, row.values)))
    '''

if __name__ == '__main__':
    parser = argparse.ArgumentParser('ALG-Factor-IC')
    parser.add_argument('--date', type=str, default='2025-07-04', help='预测日期')
    parser.add_argument('--n_factor', type=int, default=4)
    parser.add_argument('--period', type=int, default=1, help='调仓周期')
    parser.add_argument('--period_num', type=int, default=1, help='计算ic平均值的周期数')
    parser.add_argument('--is_print_case', action='store_true', help='是否打印预测结果')
    parser.add_argument('--alg_name', type=str, default='factor_ic', help='算法名称')
    args = parser.parse_args()
    print(args)
    if not is_trade_day(args.date):
        print(f'{args.date} 为非交易日 !!!')
        sys.exit(0)

    t = time.time()
    config = get_config()
    pro = tushare_pro()
    main()
    print('耗时：{}s'.format(round(time.time() - t, 4)))

