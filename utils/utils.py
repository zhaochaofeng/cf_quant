
import yaml
import tushare as ts
import numpy as np
import pandas as pd
import pymysql
from pathlib import Path
from sqlalchemy import create_engine
from datetime import datetime

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
def mysql_connect():
    ''' 创建mysql连接实例 '''
    config = get_config()
    conn = pymysql.connect(host=config['mysql']['host'],
                           user=config['mysql']['user'],
                           password=config['mysql']['password'],
                           db=config['mysql']['db'],
                           charset='utf8')
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

def winsorize(data,  scale=3, axis=0, inclusive=True):
    '''
        功能：极值化
        data：Series,DataFrame，array。待处理数据
        scale: int。标准差倍数
        inclusive: bool。超过边界的数据是否保留。True:用边界值替换；False：用None替换
        axis: 0/1。指定坐标轴。0：按行处理；1：按列处理
    '''
    def _process_1d(arr, scale, inclusive):
        """处理一维数据"""
        non_nan = arr[~np.isnan(arr)]
        if len(non_nan) < 2:  # 数据不足无法计算标准差
            return arr.copy()

        mean_val = np.nanmean(arr)
        std_val = np.nanstd(arr, ddof=1)
        low_bound = mean_val - scale * std_val
        high_bound = mean_val + scale * std_val

        arr_copy = arr.copy()
        if inclusive:
            arr_copy[arr_copy < low_bound] = low_bound
            arr_copy[arr_copy > high_bound] = high_bound
        else:
            arr_copy[(arr_copy < low_bound) | (arr_copy > high_bound)] = np.nan
        return arr_copy

    # 分数据类型处理
    if isinstance(data, pd.Series):
        return pd.Series(_process_1d(data.values, scale, inclusive), index=data.index)

    elif isinstance(data, pd.DataFrame):
        if axis == 0:  # 按列处理
            return data.apply(lambda col: _process_1d(col.values, scale, inclusive), axis=0)
        else:  # 按行处理
            return data.apply(lambda row: _process_1d(row.values, scale, inclusive), axis=1)

    elif isinstance(data, np.ndarray):
        if data.ndim == 1:  # 一维数组
            return _process_1d(data, scale, inclusive)
        else:  # 多维数组
            return np.apply_along_axis(
                lambda x: _process_1d(x, scale, inclusive),
                axis,
                data
            )
    else:
        raise TypeError("不支持的数据类型，仅支持Series/DataFrame/ndarray")

def is_trade_day(date):
    ''' 判断是否为交易日 '''
    pro = tushare_pro()
    date = datetime.strptime(date, '%Y-%m-%d').strftime('%Y%m%d')
    return True if (pro.trade_cal(start_date=date, end_date=date)['is_open'][0] == 1) else False

if __name__ == '__main__':
    print(is_trade_day('2025-06-29'))




