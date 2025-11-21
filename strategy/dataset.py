'''
    数据提供框架
'''

from qlib.contrib.data.handler import Alpha158, Alpha360
from data.factor.factor_func import (
    MACD, BOLL, KDJ, WR, BIAS_Multi, CCI, ROC
)
from utils import (
    standardize, winsorize
)

class ExpAlpha158(Alpha158):
    def __init__(self,
                 use_expand_feas: bool = False,
                 is_win: bool = False,
                 is_std: bool = False,
                 **kwargs
                 ):
        """
        Args:
            use_expand_feas: 是否使用扩展特征
            is_win: 是否取极值
            is_std: 是否标准化
        """
        self.use_expand_feas = use_expand_feas
        self.is_win = is_win
        self.is_std = is_std
        super().__init__(**kwargs)

    def get_feature_config(self):
        # 获取原始Alpha158特征配置
        fields, names = super().get_feature_config()
        # 添加自定义特征
        if self.use_expand_feas:
            factors = [MACD(), BOLL(), KDJ(), WR(), BIAS_Multi(), CCI(), ROC()]
            for factor in factors:
                for name, info in factor.items():
                    fields.append(info["exp"])
                    names.append(name)
        if self.is_win:
            fields = [winsorize(f, 3) for f in fields]
        if self.is_std:
            fields = [standardize(f) for f in fields]
        # 删除 VWAP0 列
        # idx = names.index("VWAP0")
        # fields.pop(idx)
        # names.pop(idx)
        return fields, names




