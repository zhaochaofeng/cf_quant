'''
    规模因子
'''

import numpy as np
import pandas as pd

from data.factor.utils import factor_output
from utils import winsorize, standardize, neutralize
from utils.dt import time_decorator

@time_decorator
@factor_output
def LNCAP(df) -> pd.Series:
    """ 市值(大小盘) """
    total_mv = df['$total_mv']
    lncap = np.log(total_mv)
    return lncap


@time_decorator
@factor_output
def MIDCAP(df) -> pd.Series:
    """ 中等市值

    import qlib
    from qlib.data import D
    from utils import neutralize
    import pandas as pd
    import numpy as np

    qlib.init(provider_uri='~/.qlib/qlib_data/custom_data_hfq', kernels=1)
    instruments = D.instruments(market='all')
    start_time='2025-12-01'
    end_time='2025-12-31'
    instruments = D.list_instruments(
        instruments, start_time=start_time, end_time=end_time, as_list=True
    )
    total_mv = D.features(instruments, ['$total_mv'], start_time=start_time, end_time=end_time)
    total_mv.columns = ['total_mv']
    lncap = np.log(total_mv * 10000).dropna()
    y = lncap ** 3

    resid = neutralize(y, lncap, intercept=True, level='datetime')
    merge = pd.concat([lncap, resid], axis=1, join='inner')
    merge.columns = ['lncap', 'resid']

    labels = pd.qcut(merge['lncap'], 5, labels=['小盘股', '中小盘', '中盘股', '大中盘', '大盘股'], duplicates='drop')
    print(merge.groupby(labels)['resid'].describe().T)

    # 可以观察到。中盘股残差均值小于0，下盘股和大盘股残差均值都大于0
    lncap           小盘股           中小盘           中盘股           大中盘           大盘股
    count  23735.000000  23734.000000  23734.000000  23734.000000  23734.000000
    mean     130.578505     15.974275    -53.389554   -103.583552     10.414824
    std       68.157654     22.307908     18.372853      9.826758    241.394774
    min       48.833205    -26.709102    -86.267268   -115.987386   -115.613872
    25%       84.484272     -2.996741    -69.700754   -112.654295   -106.551545
    50%      116.784224     15.015561    -54.300882   -105.716003    -78.629203
    75%      159.795758     34.802384    -37.525170    -96.004140     12.282583
    max     1194.794017     62.499206    -16.560729    -80.045903   1864.057956

    """

    total_mv = df['$total_mv']
    lncap = np.log(total_mv).dropna()

    y = lncap ** 3
    midcap = neutralize(y, lncap, intercept=True, level='datetime')
    midcap = winsorize(midcap, method='median', lower=0.01, upper=0.99, level='datetime')
    midcap = standardize(midcap, level='datetime')

    return midcap

