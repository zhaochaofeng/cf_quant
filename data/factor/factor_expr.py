'''
    因子表达式
'''

EXPRS = {
    "LOWPOS_SHARPE_COV": {
        "expr": "Cov($low/$close, EMA($close/Ref($close,1)-1, 20)/Std($close/Ref($close,1)-1, 20), 20)",
        "desc": "平滑趋势中的质量动量：low/close（日内下探幅度）与风险调整动量（EMA(ret)/Std(ret)）的20日协方差。值高=趋势走强的同时路径平滑、下探不深，区分趋势质量与噪声型上涨"
    },
    "CANDLE_SHADOW_P3": {
        "expr": "EMA(Power(($close-$open)/$open + ($high-Greater($open,$close))/$open, 3), 20)",
        "desc": "短线情绪过热反转：实体涨跌幅+上影线比例的三次方（极端K线非线性放大）做EMA平滑。值高=近期反复强上冲长上影，买盘情绪强但上方抛压同步释放，IC为负，捕捉短期过热后的回吐"
    },
    "RETCHG_MINUS_GAP": {
        "expr": "EMA(($close/Ref($close,1)-1)*($volume/Ref($volume,1)-1) - ($open-Ref($close, 1)) / (Ref($close, 1) + 1e-12), 20)",
        "desc": "剔除跳空后的量价拥挤：(收益率×成交量变化率)减去隔夜跳空幅度。第一项=盘中量价共振，第二项=隔夜信息冲击。剔除跳空后的盘中量价同步越强→资金拥挤→短期越易回落（IC为负）"
    },
    "PRICE_AMPLITUDE_EFF": {
        "expr": "Div(Div($close, Add(Max($high, 5), 1e-12)), Div(Sub($high, $low), Add($open, 1e-12)))",
        "desc": "单位振幅的价格效率：收盘占5日最高价的比例除以当日振幅率。值高=缩量高位（价格接近近期高点但振幅小，筹码锁定好）；值低=放量低位（振幅大但位置低，恐慌性抛售）。衡量上涨的'性价比'——同等价格位置下振幅越小越健康"
    },
    "AMPLITUDE_MAD17": {
        "expr": "Mad(Div(Sub($high, $low), Add($open, 1e-12)), 17)",
        "desc": "振幅不稳定性：17日振幅率的平均绝对偏差（Mad），衡量日内波动幅度的离散程度。值高=振幅忽大忽小，市场对标的的定价分歧剧烈波动；捕捉波动率聚集效应的二阶变化——波动率本身的稳定性"
    },
    "DUAL_RISK_RESONANCE": {
        "expr": "Add(Mul(Div($close, Add(WMA($close, 20), 1e-12)), Div(Std($close, 5), Add(Std($close, 20), 1e-12))), Mul(Div(Sub($high, $low), Add($open, 1e-12)),Std($volume, 5)))",
        "desc": "双维风险共振：第一项=价格偏离WMA(20) × 波动率加速（Std5/Std20），捕捉趋势方向与波动率抬升的共振；第二项=振幅率 × 量能波动（Std(volume,5)），捕捉放量震荡。Add将两个独立维度线性叠加，缺乏交互逻辑，值高=趋势偏离+波动放大+放量震荡三者共存，综合风险预警"
    },
    "VOLPRICE_COV_AMP": {
        "expr": "Mul(Cov(Div($close, Ref($close, 1)), Sub($volume, Ref($volume, 1)), 20), Div(Sub($high, $low), $close))",
        "desc": "量价协震×振幅：20日量价协方差（收益×成交量变化的同向程度）乘以当日振幅率。值高=量价正相关叠加高振幅——放量上涨或缩量下跌伴随剧烈波动，趋势确认但有泡沫风险；值低=量价背离或低振幅，趋势不牢"
    },
    "RET_VAR_11": {
        "expr": "Var(Div($close, Ref($close, 1)), 11)",
        "desc": "短期收益波动：11日收益方差。值高=近期涨跌幅剧烈，风险加大；值低=收益平稳。捕捉短期波动率聚集效应"
    },
    "AMP_HIGHVOL_RESONANCE": {
        "expr": "Mul(Div(Sub($high, $low), $close), Div(Std($high, 20), $close))",
        "desc": "振幅×高价波共振：当日振幅率乘以标准化20日高价波动（Std(high,20)/close）。值高=当日振幅大且近期高价波动剧烈——短期多空在历史高位附近剧烈博弈；双重确认高波动状态，区分偶发振幅与持续高波动环境下的振幅"
    },
    "LOWPOS_VAR_27": {
        "expr": "Var(Div($low, $close), 27)",
        "desc": "下探幅度不稳定性：27日low/close的方差，衡量日内最低价相对收盘位置的不稳定程度。值高=下探深度忽大忽小，盘中支撑位不稳定，空方试探性打压反复出现；值低=下探幅度稳定，支撑明确"
    },
    "RETVOL_VOLRATIO": {
        "expr": "Mul(Std(Div($close, Ref($close, 1)), 20), Div($volume, Mean($volume, 5)))",
        "desc": "波动率放量共振：20日收益波动率乘以量比（volume/5日均量）。值高=高波动伴随放量——市场情绪被充分激发，多空激烈交锋；值低=低波动缩量或高波动缩量——方向不明。捕捉风险事件中的量能确认"
    },
    "SURGE2_RETVOL": {
        "expr": "Mul(Power(Div($volume, EMA($volume, 20)), 2.0), Std(Div($close, Ref($close, 1)), 10))",
        "desc": "极端放量×波动率：量比平方（相对20日均量，幂次非线性放大极端值）乘以10日收益波动率。值高=极端放量叠加高波动——天量换手伴随剧烈涨跌，筹码集中转移或情绪极端宣泄；捕捉交易拥挤+价格剧烈波动的异常状态"
    },
    "INTRADAY_RETVAR_27": {
        "expr": "Var(Abs(Div(Sub($close, $open), $open)), 27)",
        "desc": "日内涨跌幅不稳定性：27日实体涨跌幅（绝对值）的方差。衡量开盘到收盘的实际涨跌（不含跳空和影线）的离散程度。值高=日内方向反复切换、涨跌幅极不稳定；值低=日内走势方向稳定"
    },
    "LOWPOS_CUMSUM": {
        "expr": "Sum(WMA(Div($low, $close), 9), 17)",
        "desc": "累计下探强度：low/close的9日WMA再做17日滚动求和。WMA放大近期下探权重，Sum累积信号。值高=近期持续深下探（收盘附近反复出现长下影或盘中深度回调），下方买盘反复承接；值低=下探浅，多头主导"
    },
    "VOLRATIO_LOWPOS_COV": {
        "expr": "Cov(Div($volume, EMA($volume, 20)), Div($low, $close), 20)",
        "desc": "量比与下探协方差：量比（volume/EMA(volume,20)）与low/close的20日协方差。值正=放量相伴下探加深——放量下跌特征，空方量能配合；值负=放量相伴下探收窄——放量时盘中支撑强劲，买盘承接有力。区分放量上涨vs放量下跌的量价配合模式"
    }
}
