CREATE TABLE IF NOT EXISTS tsfactor (
    id INT UNSIGNED NOT NULL AUTO_INCREMENT,
    ts_code VARCHAR(30) NOT NULL COMMENT 'TS股票代码，股票代码带BJ/SH/SZ后缀',
    `day` VARCHAR(10) NOT NULL COMMENT '取数日期，格式YYYY-MM-DD',
    lowdays DECIMAL(20,6) NULL COMMENT 'LOWRANGE(LOW)表示当前最低价是近多少周期内最低价的最小值',
    topdays DECIMAL(20,6) NULL COMMENT 'TOPRANGE(HIGH)表示当前最高价是近多少周期内最高价的最大值',
    downdays DECIMAL(20,6) NULL COMMENT '连跌天数',
    updays DECIMAL(20,6) NULL COMMENT '连涨天数',
    asi_qfq DECIMAL(20,6) NULL COMMENT '振动升降指标-OPEN, CLOSE, HIGH, LOW, M1=26, M2=10',
    asit_qfq DECIMAL(20,6) NULL COMMENT 'asi_qfq指标的均线',
    atr_qfq DECIMAL(20,6) NULL COMMENT '真实波动N日平均值-CLOSE, HIGH, LOW, N=20',
    bbi_qfq DECIMAL(20,6) NULL COMMENT 'BBI多空指标-CLOSE, M1=3, M2=6, M3=12, M4=20',
    bias1_qfq DECIMAL(20,6) NULL COMMENT 'BIAS乖离率-CLOSE, L1=6, L2=12, L3=24',
    bias2_qfq DECIMAL(20,6) NULL COMMENT 'BIAS乖离率-CLOSE, L1=6, L2=12, L3=24',
    bias3_qfq DECIMAL(20,6) NULL COMMENT 'BIAS乖离率-CLOSE, L1=6, L2=12, L3=24',
    boll_lower_qfq DECIMAL(20,6) NULL COMMENT 'BOLL指标，布林带-CLOSE, N=20, P=2',
    boll_mid_qfq DECIMAL(20,6) NULL COMMENT 'BOLL指标，布林带-CLOSE, N=20, P=2',
    boll_upper_qfq DECIMAL(20,6) NULL COMMENT 'BOLL指标，布林带-CLOSE, N=20, P=2',
    brar_ar_qfq DECIMAL(20,6) NULL COMMENT 'BRAR情绪指标-OPEN, CLOSE, HIGH, LOW, M1=26',
    brar_br_qfq DECIMAL(20,6) NULL COMMENT 'BRAR情绪指标-OPEN, CLOSE, HIGH, LOW, M1=26',
    cci_qfq DECIMAL(20,6) NULL COMMENT '顺势指标又叫CCI指标-CLOSE, HIGH, LOW, N=14',
    cr_qfq DECIMAL(20,6) NULL COMMENT 'CR价格动量指标-CLOSE, HIGH, LOW, N=20',
    dfma_dif_qfq DECIMAL(20,6) NULL COMMENT '平行线差指标-CLOSE, N1=10, N2=50, M=10',
    dfma_difma_qfq DECIMAL(20,6) NULL COMMENT '平行线差指标-CLOSE, N1=10, N2=50, M=10',
    dmi_adx_qfq DECIMAL(20,6) NULL COMMENT '动向指标-CLOSE, HIGH, LOW, M1=14, M2=6',
    dmi_adxr_qfq DECIMAL(20,6) NULL COMMENT '动向指标-CLOSE, HIGH, LOW, M1=14, M2=6',
    dmi_mdi_qfq DECIMAL(20,6) NULL COMMENT '动向指标-CLOSE, HIGH, LOW, M1=14, M2=6',
    dmi_pdi_qfq DECIMAL(20,6) NULL COMMENT '动向指标-CLOSE, HIGH, LOW, M1=14, M2=6',
    dpo_qfq DECIMAL(20,6) NULL COMMENT '区间震荡线-CLOSE, M1=20, M2=10, M3=6',
    madpo_qfq DECIMAL(20,6) NULL COMMENT '区间震荡线-CLOSE, M1=20, M2=10, M3=6',
    ema_qfq_10 DECIMAL(20,6) NULL COMMENT '指数移动平均-N=10',
    ema_qfq_20 DECIMAL(20,6) NULL COMMENT '指数移动平均-N=20',
    ema_qfq_250 DECIMAL(20,6) NULL COMMENT '指数移动平均-N=250',
    ema_qfq_30 DECIMAL(20,6) NULL COMMENT '指数移动平均-N=30',
    ema_qfq_5 DECIMAL(20,6) NULL COMMENT '指数移动平均-N=5',
    ema_qfq_60 DECIMAL(20,6) NULL COMMENT '指数移动平均-N=60',
    ema_qfq_90 DECIMAL(20,6) NULL COMMENT '指数移动平均-N=90',
    emv_qfq DECIMAL(20,6) NULL COMMENT '简易波动指标-HIGH, LOW, VOL, N=14, M=9',
    maemv_qfq DECIMAL(20,6) NULL COMMENT '简易波动指标-HIGH, LOW, VOL, N=14, M=9',
    expma_12_qfq DECIMAL(20,6) NULL COMMENT 'EMA指数平均数指标-CLOSE, N1=12, N2=50',
    expma_50_qfq DECIMAL(20,6) NULL COMMENT 'EMA指数平均数指标-CLOSE, N1=12, N2=50',
    kdj_qfq DECIMAL(20,6) NULL COMMENT 'KDJ指标-CLOSE, HIGH, LOW, N=9, M1=3, M2=3',
    kdj_d_qfq DECIMAL(20,6) NULL COMMENT 'KDJ指标-CLOSE, HIGH, LOW, N=9, M1=3, M2=3',
    kdj_k_qfq DECIMAL(20,6) NULL COMMENT 'KDJ指标-CLOSE, HIGH, LOW, N=9, M1=3, M2=3',
    ktn_down_qfq DECIMAL(20,6) NULL COMMENT '肯特纳交易通道, N选20日，ATR选10日-CLOSE, HIGH, LOW, N=20, M=10',
    ktn_mid_qfq DECIMAL(20,6) NULL COMMENT '肯特纳交易通道, N选20日，ATR选10日-CLOSE, HIGH, LOW, N=20, M=10',
    ktn_upper_qfq DECIMAL(20,6) NULL COMMENT '肯特纳交易通道, N选20日，ATR选10日-CLOSE, HIGH, LOW, N=20, M=10',
    ma_qfq_10 DECIMAL(20,6) NULL COMMENT '简单移动平均-N=10',
    ma_qfq_20 DECIMAL(20,6) NULL COMMENT '简单移动平均-N=20',
    ma_qfq_250 DECIMAL(20,6) NULL COMMENT '简单移动平均-N=250',
    ma_qfq_30 DECIMAL(20,6) NULL COMMENT '简单移动平均-N=30',
    ma_qfq_5 DECIMAL(20,6) NULL COMMENT '简单移动平均-N=5',
    ma_qfq_60 DECIMAL(20,6) NULL COMMENT '简单移动平均-N=60',
    ma_qfq_90 DECIMAL(20,6) NULL COMMENT '简单移动平均-N=90',
    macd_qfq DECIMAL(20,6) NULL COMMENT 'MACD指标-CLOSE, SHORT=12, LONG=26, M=9',
    macd_dea_qfq DECIMAL(20,6) NULL COMMENT 'MACD指标-CLOSE, SHORT=12, LONG=26, M=9',
    macd_dif_qfq DECIMAL(20,6) NULL COMMENT 'MACD指标-CLOSE, SHORT=12, LONG=26, M=9',
    mass_qfq DECIMAL(20,6) NULL COMMENT '梅斯线-HIGH, LOW, N1=9, N2=25, M=6',
    ma_mass_qfq DECIMAL(20,6) NULL COMMENT '梅斯线-HIGH, LOW, N1=9, N2=25, M=6',
    mfi_qfq DECIMAL(20,6) NULL COMMENT 'MFI指标是成交量的RSI指标-CLOSE, HIGH, LOW, VOL, N=14',
    mtm_qfq DECIMAL(20,6) NULL COMMENT '动量指标-CLOSE, N=12, M=6',
    mtmma_qfq DECIMAL(20,6) NULL COMMENT '动量指标-CLOSE, N=12, M=6',
    obv_qfq DECIMAL(20,6) NULL COMMENT '能量潮指标-CLOSE, VOL',
    psy_qfq DECIMAL(20,6) NULL COMMENT '投资者对股市涨跌产生心理波动的情绪指标-CLOSE, N=12, M=6',
    psyma_qfq DECIMAL(20,6) NULL COMMENT '投资者对股市涨跌产生心理波动的情绪指标-CLOSE, N=12, M=6',
    roc_qfq DECIMAL(20,6) NULL COMMENT '变动率指标-CLOSE, N=12, M=6',
    maroc_qfq DECIMAL(20,6) NULL COMMENT '变动率指标-CLOSE, N=12, M=6',
    rsi_qfq_12 DECIMAL(20,6) NULL COMMENT 'RSI指标-CLOSE, N=12',
    rsi_qfq_24 DECIMAL(20,6) NULL COMMENT 'RSI指标-CLOSE, N=24',
    rsi_qfq_6 DECIMAL(20,6) NULL COMMENT 'RSI指标-CLOSE, N=6',
    taq_down_qfq DECIMAL(20,6) NULL COMMENT '唐安奇通道(海龟)交易指标-HIGH, LOW, 20',
    taq_mid_qfq DECIMAL(20,6) NULL COMMENT '唐安奇通道(海龟)交易指标-HIGH, LOW, 20',
    taq_up_qfq DECIMAL(20,6) NULL COMMENT '唐安奇通道(海龟)交易指标-HIGH, LOW, 20',
    trix_qfq DECIMAL(20,6) NULL COMMENT '三重指数平滑平均线-CLOSE, M1=12, M2=20',
    trma_qfq DECIMAL(20,6) NULL COMMENT '三重指数平滑平均线-CLOSE, M1=12, M2=20',
    vr_qfq DECIMAL(20,6) NULL COMMENT 'VR容量比率-CLOSE, VOL, M1=26',
    wr_qfq DECIMAL(20,6) NULL COMMENT 'W&R 威廉指标-CLOSE, HIGH, LOW, N=10, N1=6',
    wr1_qfq DECIMAL(20,6) NULL COMMENT 'W&R 威廉指标-CLOSE, HIGH, LOW, N=10, N1=6',
    xsii_td1_qfq DECIMAL(20,6) NULL COMMENT '薛斯通道II-CLOSE, HIGH, LOW, N=102, M=7',
    xsii_td2_qfq DECIMAL(20,6) NULL COMMENT '薛斯通道II-CLOSE, HIGH, LOW, N=102, M=7',
    xsii_td3_qfq DECIMAL(20,6) NULL COMMENT '薛斯通道II-CLOSE, HIGH, LOW, N=102, M=7',
    xsii_td4_qfq DECIMAL(20,6) NULL COMMENT '薛斯通道II-CLOSE, HIGH, LOW, N=102, M=7',
    PRIMARY KEY (id),
    UNIQUE KEY uk_code_day (`ts_code`, `day`)
) ENGINE=InnoDB
  DEFAULT CHARSET=utf8mb4
  COLLATE=utf8mb4_unicode_ci
  COMMENT='tushare技术因子'
  AUTO_INCREMENT=1;


'''
# 缺失值数量。day='2025-07-10'
id	0
ts_code	0
day	0
lowdays	0
topdays	0
downdays	0
updays	0
asi_qfq	10
asit_qfq	11
atr_qfq	5
bbi_qfq	5
bias1_qfq	1
bias2_qfq	3
bias3_qfq	8
boll_lower_qfq	5
boll_mid_qfq	5
boll_upper_qfq	5
brar_ar_qfq	8
brar_br_qfq	10
cci_qfq	3
cr_qfq	5
dfma_dif_qfq	16
dfma_difma_qfq	21
dmi_adx_qfq	5
dmi_adxr_qfq	8
dmi_mdi_qfq	3
dmi_pdi_qfq	3
dpo_qfq	10
madpo_qfq	11
ema_qfq_10	0
ema_qfq_20	0
ema_qfq_250	0
ema_qfq_30	0
ema_qfq_5	0
ema_qfq_60	0
ema_qfq_90	0
emv_qfq	10
maemv_qfq	11
expma_12_qfq	0
expma_50_qfq	0
kdj_qfq	2
kdj_d_qfq	2
kdj_k_qfq	2
ktn_down_qfq	2
ktn_mid_qfq	0
ktn_upper_qfq	2
ma_qfq_10	2
ma_qfq_20	5
ma_qfq_250	109
ma_qfq_30	10
ma_qfq_5	1
ma_qfq_60	22
ma_qfq_90	40
macd_qfq	0
macd_dea_qfq	0
macd_dif_qfq	0
mass_qfq	15
ma_mass_qfq	16
mfi_qfq	3
mtm_qfq	3
mtmma_qfq	5
obv_qfq	0
psy_qfq	3
psyma_qfq	5
roc_qfq	3
maroc_qfq	5
rsi_qfq_12	0
rsi_qfq_24	0
rsi_qfq_6	0
taq_down_qfq	5
taq_mid_qfq	5
taq_up_qfq	5
trix_qfq	0
trma_qfq	5
vr_qfq	8
wr_qfq	2
wr1_qfq	1
xsii_td1_qfq	1
xsii_td2_qfq	1
xsii_td3_qfq	0
xsii_td4_qfq	0
'''




