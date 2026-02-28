
""" 表达式因子 """
from .factor_func import (
    MACD, BOLL, RSI_Multi, KDJ, DMI, WR, BIAS_Multi, CCI, ROC
)

""" 函数因子 """
# 动量因子
from .momentum import (
    MOM_10D, REVERSAL_5D, MOM_VOL_ADJ_10D
)

# 波动率因子
from .volatility import (
    VOLATILITY_20D
)

# 规模因子
from .size import (
    LNCAP, MIDCAP
)




