'''
    因子库建设
'''


""" 表达式因子 """
from .factor_func import (
    MACD, BOLL, RSI_Multi, KDJ, DMI, WR, BIAS_Multi, CCI, ROC
)

""" 表达式因子-自动构建 """
from .factor_expr import EXPRS

""" 函数因子-CNE6 (当前38个) """
# 函数因子输入的df 索引为 <instrument, datetime>
# 1.规模因子（2个）
from .size import (
    LNCAP, MIDCAP
)

# 2.波动率因子（4个）
from .volatility import (
    HBETA, HSIGMA, DASTD, CMRA
)

# 3.流动性因子（4个）
from .liquidity import (
    STOM, STOQ, STOA, ATVR
)

# 4.动量因子（5个）
from .momentum import (
    STREV, SEASON, INDMOM, RSTR, HALPHA
)

# 5.质量因子（15个）
from .quility import (
    MLEV, BLEV, DTOA,   # 杠杆
    VSAL, VERN, VFLO,   # 盈利波动性
    ABS, ACF,           # 盈利质量
    ATO, GP, GPM, ROA,  # 盈利能力,
    AGRO, IGRO, CXGRO,  # 投资能力
)

# 6.价值因子（6个）
from .value import (
    BTOP, ETOP, CETOP, EM, LTRSTR, LTHALPHA
)

# 7.成长因子（2个）
from .growth import (
    EGRO, SGRO
)


""" 函数因子-自动构建 """
from .volatility import (
    VOLATILITY_20D
)

from .momentum import (
    MOM_10D, REVERSAL_5D, MOM_VOL_ADJ_10D
)

