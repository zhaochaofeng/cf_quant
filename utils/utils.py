
import yaml
import tushare as ts
import numpy as np
import pandas as pd
import pymysql
import redis
from pathlib import Path
from sqlalchemy import create_engine
from datetime import datetime, timedelta
from jqdatasdk import *

def get_config(config_path: str = None):
    ''' 读取配置文件 '''
    if config_path is None:
        config_path = Path(__file__).parent.parent / 'config.yaml'
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        # print('配置文件加载完成：{}'.format(config_path))
        return config
    except Exception as e:
        print('配置文件加载失败：{}'.format(e))
    return None

def tushare_pro():
    config = get_config()
    pro = ts.pro_api(config['tushare']['token'])
    return pro

def tushare_ts():
    ''' 有些数据不能使用pro对象访问 '''
    config = get_config()
    ts.set_token(config['tushare']['token'])
    return ts

def jq_connect():
    config = get_config()
    auth(config['joinqaunt']['username'], config['joinqaunt']['password'])

def mysql_connect():
    ''' 创建mysql连接实例 '''
    config = get_config()
    conn = pymysql.connect(host=config['mysql']['host'],
                           user=config['mysql']['user'],
                           password=config['mysql']['password'],
                           db=config['mysql']['db'],
                           charset='utf8')
    return conn

def redis_connect():
    ''' 创建redis连接实例 '''
    config = get_config()
    conn = redis.Redis(host=config['redis']['host'],
                       password=config['redis']['password'])
    return conn

def sql_engine():
    ''' 创建sqlalchemy连接mysql的引擎 '''
    config = get_config()
    url = "mysql+pymysql://{}:{}@{}:3306/{}?charset=utf8mb4".format(
        config['mysql']['user'],
        config['mysql']['password'],
        config['mysql']['host'],
        config['mysql']['db']
    )
    engine = create_engine(url, echo=False)
    return engine

def get_feas(config_path: str = None):
    ''' 读取特征字段 '''
    if config_path is None:
        config_path = Path(__file__).parent.parent / 'features.yaml'
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        # print('配置文件加载完成：{}'.format(config_path))
        return config
    except Exception as e:
        print('配置文件加载失败：{}'.format(e))
    return None

def is_trade_day(date):
    ''' 判断是否为交易日 '''
    pro = tushare_pro()
    date = datetime.strptime(date, '%Y-%m-%d').strftime('%Y%m%d')
    return True if (pro.trade_cal(start_date=date, end_date=date)['is_open'][0] == 1) else False

def get_n_pretrade_day(date, n):
    ''' date 前n个交易日。格式 YYYY-MM-DD
        若date为非交易日，则会认为date为历史最近的一个交易日
        n=0时，返回原始日期date
    '''
    pro = tushare_pro()
    trade_days = pro.trade_cal(
        exchange='',
        start_date=(datetime.strptime(date, '%Y-%m-%d') - timedelta(days=(n+300))).strftime('%Y%m%d'),
        end_date=datetime.strptime(date, '%Y-%m-%d').strftime('%Y%m%d'),
        is_open='1')
    trade_days = trade_days['cal_date'].tolist()
    trade_days.sort()
    return datetime.strptime(trade_days[-(n+1)], '%Y%m%d').strftime('%Y-%m-%d')

def get_month_start_end(date_str):
    """
    根据输入日期字符串，返回该月的第一天和最后一天
    参数:
        date_str: 日期字符串，格式为 'YYYY-MM-DD'
    返回:
        (month_start, month_end): 元组，包含该月第一天和最后一天的日期字符串（格式为 'YYYY-MM-DD'）
    """
    # 将字符串转换为datetime对象
    date = datetime.strptime(date_str, '%Y-%m-%d')
    # 获取当月第一天
    month_start = date.replace(day=1)
    # 获取下个月的第一天，并减去一天得到本月的最后一天
    if month_start.month == 12:
        next_month = month_start.replace(year=month_start.year + 1, month=1)
    else:
        next_month = month_start.replace(month=month_start.month + 1)
    month_end = next_month - timedelta(days=1)
    # 转换为字符串格式输出
    return month_start.strftime('%Y-%m-%d'), month_end.strftime('%Y-%m-%d')

if __name__ == '__main__':
    jq_conn()
    print(get_account_info())



