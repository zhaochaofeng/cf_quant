
""" 表达式因子 """
from .factor_func import (
    MACD, BOLL, RSI_Multi, KDJ, DMI, WR, BIAS_Multi, CCI, ROC
)

""" 函数因子 """
# 函数因子输入的df 索引为 <instrument, datetime>
# 规模因子
from .size import (
    LNCAP, MIDCAP
)

# 波动率因子
from .volatility import (
    VOLATILITY_20D, BETA, HSIGMA, DASTD, CMRA
)

# 动量因子
from .momentum import (
    STREV, SEASON, INDMOM, RSTR, HALPHA,
    MOM_10D, REVERSAL_5D, MOM_VOL_ADJ_10D
)

# 流动性因子
from .liquidity import (
    STOM, STOQ, STOA, ATVR
)

