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
    field = df['$circ_mv']
    lncap = np.log(field)
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
    circ_mv = D.features(instruments, ['$circ_mv'], start_time=start_time, end_time=end_time)
    circ_mv.columns = ['circ_mv']
    lncap = np.log(circ_mv * 10000).dropna()
    y = lncap ** 3

    resid = neutralize(y, lncap, intercept=True, level='datetime')
    merge = pd.concat([lncap, resid], axis=1, join='inner')
    merge.columns = ['lncap', 'resid']

    labels = pd.qcut(merge['lncap'], 5, labels=['小盘股', '中小盘', '中盘股', '大中盘', '大盘股'], duplicates='drop')
    print(merge.groupby(labels)['resid'].describe().T)

    # 可以观察到。中盘股残差均值小于0，下盘股和大盘股残差均值都大于0
    lncap           小盘股           中小盘           中盘股           大中盘           大盘股
    count  23735.000000  23734.000000  23734.000000  23734.000000  23734.000000
    mean     137.879820     -2.270755    -66.843147   -103.865496     35.093769
    std       91.857469     22.525693     15.585970      5.199230    236.510209
    min       32.801393    -42.817697    -94.268194   -110.015085   -105.848627
    25%       67.822948    -22.138199    -80.098711   -108.027798    -89.501895
    50%      110.278836     -3.437104    -67.897848   -106.017824    -51.170542
    75%      181.711688     16.928683    -53.762151   -100.298271     56.080001
    max     1009.902285     44.348441    -34.661370    -89.483353   2003.726138

    """

    field = df['$circ_mv']
    lncap = np.log(field).dropna()

    y = lncap ** 3
    midcap = neutralize(y, lncap, intercept=True, level='datetime')
    midcap = winsorize(midcap, method='median', lower=0.01, upper=0.99, level='datetime')
    midcap = standardize(midcap, level='datetime')

    return midcap

