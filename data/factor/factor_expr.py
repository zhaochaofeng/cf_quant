'''
    因子表达式
'''

factor_exprs = {
    "LOWPOS_SHARPE_COV": {
        "expr": "Cov($low/$close, EMA($close/Ref($close,1)-1, 20)/Std($close/Ref($close,1)-1, 20), 20)",
        "desc": "平滑趋势中的质量动量"
    },
    "CANDLE_SHADOW_P3": {
        "expr": "EMA(Power(($close-$open)/$open + ($high-Greater($open,$close))/$open, 3), 20)",
        "desc": "上冲情绪后的短期反转"
    },
    "RETCHG_MINUS_GAP": {
        "expr": "EMA(($close/Ref($close,1)-1)*($volume/Ref($volume,1)-1) - ($open-Ref($close, 1)) / (Ref($close, 1) + 1e-12), 20)",
        "desc": "剔除跳空后的量价拥挤"
    }
}










