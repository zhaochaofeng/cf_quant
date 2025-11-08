'''
    因子计算函数
'''

from qlib.data.ops import (
    EMA, Delta, Mean, Std,
    Min, Max, Abs, Ref, If,
    Sub, Add
)


def MACD(short=12, long=26, mid=9):
    DIF = EMA('$close', short) - EMA('$close', long)
    DEA = EMA(DIF, mid)
    MACD = (DIF - DEA) * 2
    return {
        "MACD": {"exp": MACD, 'name': '异同移动平均线'},
        "DIF": {"exp": DIF, 'name': '差离值'},
        "DEA": {"exp": DEA, 'name': '异同平均数'}
    }


def BOLL(n=20, k=2):
    MB = Mean('$close', n)
    std = Std('$close', n)
    UP = MB + std * k
    DN = MB - std * k

    return {
        "BOLL_MID": {"exp": MB, 'name': '布林中轨'},
        "BOLL_UP": {"exp": UP, 'name': '布林上轨'},
        "BOLL_LOW": {"exp": DN, 'name': '布林下轨'}
    }


def RSI(n=14):
    # 近似实现
    delta = Delta('$close', 1)
    U = Max(delta, 0)
    D = Max(delta * (-1), 0)

    AvgU = EMA(U, 2 * n - 1)
    AvgD = EMA(D, 2 * n - 1)

    RS = AvgU / (AvgD + 1e-8)
    RSI_val = 100 - 100 / (1 + RS)
    return {
        'RSI': {"exp": RSI_val, 'name': '相对强弱指标'}
    }

# 告警日志
def RSI_Multi():
    """ 多期RSI """
    rsi6 = RSI(6)['RSI']
    rsi12 = RSI(12)['RSI']
    rsi24 = RSI(24)['RSI']
    return {
        'RSI6': {"exp": rsi6['exp'], 'name': '6日相对强弱指标'},
        'RSI12': {"exp": rsi12['exp'], 'name': '12日相对强弱指标'},
        'RSI24': {"exp": rsi24['exp'], 'name': '24日相对强弱指标'}
    }


def KDJ(N=9, M1=3, M2=3):
    # 近似实现
    Ln = Min('$low', N)
    Hn = Max('$high', N)
    RSV = (('$close' - Ln) / (Hn - Ln + 1e-8)) * 100
    K = EMA(RSV, 2 * M1 - 1)
    D = EMA(K, 2 * M2 - 1)
    J = 3 * K - 2 * D
    return {
        "K": {"exp": K, 'name': 'K值'},
        "D": {"exp": D, 'name': 'D值'},
        "J": {"exp": J, 'name': 'J值'}
    }

# 耗时较长
def DMI(N=14):
    """
    DMI（Directional Movement Index, 动向指标）
    使用 Qlib 表达式 API 实现，采用 EMA 近似 Wilder 平滑。
    """
    high = '$high'
    low = '$low'
    close = '$close'

    # 计算 True Range (TR)
    prev_close = Ref(close, 1)
    tr1 = Sub(high, low)
    tr2 = Abs(Sub(high, prev_close))
    tr3 = Abs(Sub(low, prev_close))
    TR = If(tr1 >= tr2,
            If(tr1 >= tr3, tr1, tr3),
            If(tr2 >= tr3, tr2, tr3))

    # 计算 +DM 和 -DM
    prev_high = Ref(high, 1)
    prev_low = Ref(low, 1)

    up_move = high - prev_high
    down_move = prev_low - low

    plus_DM = If((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_DM = If((down_move > up_move) & (down_move > 0), down_move, 0)

    # 使用 EMA 近似 Wilder 平滑
    STR = EMA(TR, 2 * N - 1)
    plus_SDM = EMA(plus_DM, 2 * N - 1)
    minus_SDM = EMA(minus_DM, 2 * N - 1)

    # 计算 +DI 和 -DI
    plus_DI = (plus_SDM / (STR + 1e-8)) * 100
    minus_DI = (minus_SDM / (STR + 1e-8)) * 100

    # 计算 DX
    DX = (Abs(plus_DI - minus_DI) / (plus_DI + minus_DI + 1e-8)) * 100

    # 计算 ADX（DX 的平滑）
    ADX = EMA(DX, 2 * N - 1)

    # 计算 ADXR（平均趋向指标）
    ADXR = (ADX + Ref(ADX, N)) / 2

    return {
        "PDI": {"exp": plus_DI, 'name': '上升方向线'},
        "MDI": {"exp": minus_DI, 'name': '下降方向线'},
        "ADX": {"exp": ADX, 'name': '平均趋向指标'},
        "ADXR": {"exp": ADXR, 'name': '平均趋向指数'}
    }


def WR(N=10):
    """ 威廉指标 """
    max_n = Max('$high', N)
    min_n = Min('$low', N)
    WR = (max_n - '$close') / (max_n - min_n + 1e-8) * 100
    return {
        'WR': {"exp": WR, 'name': '威廉指标'}
    }


def BIAS(N=12):
    """ 乖离率 """
    close = '$close'
    ma_N = Mean(close, N)
    bias = (close - ma_N) / (ma_N + 1e-8) * 100  # 防止除零
    return {
        "BIAS": {"exp": bias, 'name': '乖离率'}
    }


def BIAS_Multi():
    """ 多期乖离率 """
    bias6 = BIAS(6)['BIAS']
    bias12 = BIAS(12)['BIAS']
    bias24 = BIAS(24)['BIAS']
    return {
        'BIAS6': {"exp": bias6['exp'], 'name': '6日乖离率'},
        'BIAS12': {"exp": bias12['exp'], 'name': '12日乖离率'},
        'BIAS24': {"exp": bias24['exp'], 'name': '24日乖离率'}
    }


def CCI(N=20):
    """
    CCI（Commodity Channel Index，商品通道指标）
    """
    high = '$high'
    low = '$low'
    close = '$close'

    TP = Add(high, Add(low, close)) / 3
    MATP = Mean(TP, N)
    # 计算平均偏差（Mean Deviation），用每时刻的偏差作为近似
    AD = Abs(TP - MATP)
    MD = Mean(AD, N)
    CCI = (TP - MATP) / (0.015 * MD + 1e-8)
    return {
        "CCI": {"exp": CCI, 'name': '商品通道指数'}
    }


def ROC(N=12):
    """
    ROC（Rate of Change，变化率）
    """
    close = '$close'
    ROC = (close - Ref(close, N)) / (Ref(close, N) + 1e-8) * 100
    return {
        "ROC": {"exp": ROC, 'name': '变化率'}
    }


if __name__ == '__main__':
    import qlib
    from qlib.constant import REG_CN
    from qlib.data import D

    qlib.init(provider_uri='~/.qlib/qlib_data/custom_data_hfq', region=REG_CN)
    feas = D.features(["SZ000001"], ['$close', ROC()['ROC']],
                      start_time='2025-08-01', end_time='2025-09-11', freq='day')
    feas.columns = ['close', 'BIAS6', 'BIAS12', 'BIAS24']
    print(feas)
