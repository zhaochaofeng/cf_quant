''''
    量化策略：因子分析 + 线性回归
'''

import time
import argparse
from datetime import datetime, timedelta
import redis
import pandas as pd
import numpy as np
from jqdatasdk import *
from sqlalchemy import create_engine
from statsmodels.multivariate.factor import Factor
import statsmodels.api as sm
from jqfactor_analyzer import winsorize, standardlize

from utils.utils import get_config
features = ['capitalization', 'circulating_cap', 'market_cap',
             'circulating_market_cap', 'turnover_ratio', 'pe_ratio', 'pe_ratio_lyr',
             'pb_ratio', 'ps_ratio', 'pcf_ratio']
alg_name = 'linear_regression'

def get_fea_data(day):
    url = "mysql+pymysql://{}:{}@{}:3306/{}?charset=utf8mb4".format(
        config['mysql']['user'],
        config['mysql']['password'],
        config['mysql']['host'],
        config['mysql']['db']
    )
    engine = create_engine(url, echo=False)
    sql = "select * from valuation where day='{}';".format(day)
    df = pd.read_sql(sql, con=engine)
    df = df[['code'] + features]
    df.set_index(['code'], inplace=True)
    return df

def get_return_data():
    ''' 收益率 '''
    auth(config['joinqaunt']['username'], config['joinqaunt']['password'])

    stocks_info = get_all_securities(types=['stock'], date=args.day)
    stocks = stocks_info.index.tolist()
    print('股票数：{}'.format(len(stocks)))
    end_date = datetime.strptime(args.day, '%Y-%m-%d') + timedelta(days=1)
    df = get_price(stocks, start_date=args.day, end_date=end_date, frequency='1d', fields=['close'])

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

def fa_model(endog=None):
    factor_model = Factor(endog=endog, n_factor=args.n_factor)
    result = factor_model.fit()
    # 因子载荷矩阵
    loadings = result.loadings
    loadings = pd.DataFrame(loadings)
    print('{}: {}'.format('因子载荷矩阵', '-'*100))
    print(loadings)
    # 特征值
    eigen = np.array(result.eigenvals)
    print('{}: {}'.format('特征值', '-'*100))
    print(eigen)

    # 因子旋转
    result.rotate(method='varimax')
    loadings = result.loadings
    loadings = pd.DataFrame(loadings)
    print('{}: {}'.format('旋转因子载荷矩阵', '-' * 100))
    print(loadings)

    return result

def fa_predict(result, endog):
    # 原始特征值转化为因子值
    factor_score = result.factor_scoring(endog=endog)
    factor_name = []
    for i in range(args.n_factor):
        factor_name.append('{}{}'.format('f', i + 1))
    print('factor_name: {}'.format(factor_name))

    factor_score = pd.DataFrame(factor_score)
    factor_score.columns = factor_name
    factor_score.set_index(endog.index, inplace=True)
    print('{}: {}'.format('因子得分', '-' * 100))
    print(factor_score.head(5))
    print(factor_score.shape)
    return factor_score

def load_to_redis(predict, day):
    # 入库redis
    r = redis.Redis(host=config['redis']['host'])
    dic = {}
    codes = predict.index.tolist()
    scores = predict.values.tolist()
    for i in range(len(codes)):
        dic[codes[i]] = scores[i]
    key = "{}_{}".format(alg_name, day)
    print('key: {}'.format(key))
    r.hset(key, mapping=dic)

    p = r.hgetall(key)
    p = {k.decode(): float(v.decode()) for k, v in p.items()}
    p_sorted = sorted(p.items(), key=lambda x: x[1], reverse=True)
    for i, (code, score) in enumerate(p_sorted):
        if i >= 100:
            break
        name = get_security_info(code, date=args.day).display_name
        print('{}, {}, {}, {}'.format(i+1, code, name, score))

def preprocess(data):
    # 去极值
    data = winsorize(data, scale=3, axis=0)
    # 归一化
    data = standardlize(data, axis=0)
    # 处理缺失值。自变量用0填充，因变量过滤为Nan的数据
    data.fillna(0, inplace=True)
    return data

def main():
    # 加载特征数据
    X = get_fea_data(args.day)
    # print(X.head())
    # 加载收益率数据
    y = get_return_data()
    # print(y.head())
    # 数据预处理
    X = preprocess(X)
    # print(X.head())
    # print(X.describe())
    y = y[~y['return'].isna()]
    # 合并特征和收益率
    df_merge = pd.concat([X, y], axis=1, join='inner')
    # print(df_merge.isna().any(axis=0))
    X = df_merge[features]
    y = df_merge[['return']]

    # 因子分析
    results_fa = fa_model(endog=X)
    factor_score = fa_predict(results_fa, X)
    print('factor_score: {}'.format(type(factor_score)))

    # 线性回归
    X = factor_score
    X = sm.add_constant(X)
    model = sm.OLS(y, X)
    results_linear = model.fit()
    print('{}: {}'.format('回归参数', '-'*100))
    print(results_linear.params)  # Series
    print('Rsquared: {}'.format(results_linear.rsquared))
    print('adjust Rsquared: {}'.format(results_linear.rsquared_adj))

    # 预测后一天的数据
    day_one = (datetime.strptime(args.day, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
    day_two = (datetime.strptime(args.day, '%Y-%m-%d') + timedelta(days=2)).strftime('%Y-%m-%d')
    print('day_one: {}'.format(day_one))
    X_new = get_fea_data(day_one)
    X_new = preprocess(X_new)
    X_new = fa_predict(results_fa, X_new)
    X_new = sm.add_constant(X_new)
    print('X_new shape: '.format(X_new.shape))
    print(X_new.head())
    predict = results_linear.predict(X_new)  # Series
    print("predict type: {}".format(type(predict)))
    print("predict shape: {}".format(predict.shape))
    print(predict.head())
    # 加载到redis
    load_to_redis(predict, day_two)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('linear_regression')
    parser.add_argument('--day', type=str, default='2025-03-19')
    parser.add_argument('--n_factor', type=int, default=4)
    args = parser.parse_args()
    print(args)
    t = time.time()
    config = get_config()
    main()
    print('耗时：{}s'.format(round(time.time() - t, 4)))

