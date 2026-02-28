'''
    规模因子
'''

import numpy as np


def LNCAP(df):
    """ 市值 """

    df = df.sort_index()
    circ_mv = df['$circ_mv']
    lncap = np.log(circ_mv)
    lncap.name = 'LNCAP'
    result_df = lncap.to_frame()

    return result_df


def MIDCAP(df):
    """ 中等市值 """

    lncap = LNCAP(df)

    return lncap

