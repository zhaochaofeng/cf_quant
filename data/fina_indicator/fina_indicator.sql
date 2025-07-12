CREATE TABLE IF NOT EXISTS fina_indicator(
    id INT NOT NULL AUTO_INCREMENT,
    ts_code VARCHAR(30) NOT NULL COMMENT 'TS股票代码，股票代码带BJ/SH/SZ后缀',
    day VARCHAR(10) NOT NULL COMMENT '交易日期，格式YYYY-MM-DD',
    ann_date VARCHAR(10) COMMENT '公告日期(财报发布日期)',
    end_date VARCHAR(10) COMMENT '报告期（财报周期0331/0630/0930/1231）',
    eps DECIMAL(20, 4) COMMENT '基本每股收益',
    dt_eps DECIMAL(20, 4) COMMENT '稀释每股收益',
    total_revenue_ps DECIMAL(20, 4) COMMENT '每股营业总收入',
    revenue_ps DECIMAL(20, 4) COMMENT '每股营业收入',
    capital_rese_ps DECIMAL(20, 4) COMMENT '每股资本公积',
    surplus_rese_ps DECIMAL(20, 4) COMMENT '每股盈余公积',
    undist_profit_ps DECIMAL(20, 4) COMMENT '每股未分配利润',
    extra_item DECIMAL(20, 4) COMMENT '非经常性损益',
    profit_dedt DECIMAL(20, 4) COMMENT '扣除非经常性损益后的净利润（扣非净利润）',
    gross_margin DECIMAL(20, 4) COMMENT '毛利',
    current_ratio DECIMAL(20, 4) COMMENT '流动比率',
    quick_ratio DECIMAL(20, 4) COMMENT '速动比率',
    cash_ratio DECIMAL(20, 4) COMMENT '保守速动比率',
    ar_turn DECIMAL(20, 4) COMMENT '应收账款周转率',
    ca_turn DECIMAL(20, 4) COMMENT '流动资产周转率',
    fa_turn DECIMAL(20, 4) COMMENT '固定资产周转率',
    assets_turn DECIMAL(20, 4) COMMENT '总资产周转率',
    op_income DECIMAL(20, 4) COMMENT '经营活动净收益',
    ebit DECIMAL(20, 4) COMMENT '息税前利润',
    ebitda DECIMAL(20, 4) COMMENT '息税折旧摊销前利润',
    fcff DECIMAL(20, 4) COMMENT '企业自由现金流量',
    fcfe DECIMAL(20, 4) COMMENT '股权自由现金流量',
    current_exint DECIMAL(20, 4) COMMENT '无息流动负债',
    noncurrent_exint DECIMAL(20, 4) COMMENT '无息非流动负债',
    interestdebt DECIMAL(20, 4) COMMENT '带息债务',
    netdebt DECIMAL(20, 4) COMMENT '净债务',
    tangible_asset DECIMAL(20, 4) COMMENT '有形资产',
    working_capital DECIMAL(20, 4) COMMENT '营运资金',
    networking_capital DECIMAL(20, 4) COMMENT '营运流动资本',
    invest_capital DECIMAL(20, 4) COMMENT '全部投入资本',
    retained_earnings DECIMAL(20, 4) COMMENT '留存收益',
    diluted2_eps DECIMAL(20, 4) COMMENT '期末摊薄每股收益',
    bps DECIMAL(20, 4) COMMENT '每股净资产',
    ocfps DECIMAL(20, 4) COMMENT '每股经营活动产生的现金流量净额',
    retainedps DECIMAL(20, 4) COMMENT '每股留存收益',
    cfps DECIMAL(20, 4) COMMENT '每股现金流量净额',
    ebit_ps DECIMAL(20, 4) COMMENT '每股息税前利润',
    fcff_ps DECIMAL(20, 4) COMMENT '每股企业自由现金流量',
    fcfe_ps DECIMAL(20, 4) COMMENT '每股股东自由现金流量',
    netprofit_margin DECIMAL(20, 4) COMMENT '销售净利率',
    grossprofit_margin DECIMAL(20, 4) COMMENT '销售毛利率',
    cogs_of_sales DECIMAL(20, 4) COMMENT '销售成本率',
    expense_of_sales DECIMAL(20, 4) COMMENT '销售期间费用率',
    profit_to_gr DECIMAL(20, 4) COMMENT '净利润/营业总收入',
    saleexp_to_gr DECIMAL(20, 4) COMMENT '销售费用/营业总收入',
    adminexp_of_gr DECIMAL(20, 4) COMMENT '管理费用/营业总收入',
    finaexp_of_gr DECIMAL(20, 4) COMMENT '财务费用/营业总收入',
    impai_ttm DECIMAL(20, 4) COMMENT '资产减值损失/营业总收入',
    gc_of_gr DECIMAL(20, 4) COMMENT '营业总成本/营业总收入',
    op_of_gr DECIMAL(20, 4) COMMENT '营业利润/营业总收入',
    ebit_of_gr DECIMAL(20, 4) COMMENT '息税前利润/营业总收入',
    roe DECIMAL(20, 4) COMMENT '净资产收益率',
    roe_waa DECIMAL(20, 4) COMMENT '加权平均净资产收益率',
    roe_dt DECIMAL(20, 4) COMMENT '净资产收益率(扣除非经常损益)',
    roa DECIMAL(20, 4) COMMENT '总资产报酬率',
    npta DECIMAL(20, 4) COMMENT '总资产净利润',
    roic DECIMAL(20, 4) COMMENT '投入资本回报率',
    roe_yearly DECIMAL(20, 4) COMMENT '年化净资产收益率',
    roa2_yearly DECIMAL(20, 4) COMMENT '年化总资产报酬率',
    debt_to_assets DECIMAL(20, 4) COMMENT '资产负债率',
    assets_to_eqt DECIMAL(20, 4) COMMENT '权益乘数',
    dp_assets_to_eqt DECIMAL(20, 4) COMMENT '权益乘数(杜邦分析)',
    ca_to_assets DECIMAL(20, 4) COMMENT '流动资产/总资产',
    nca_to_assets DECIMAL(20, 4) COMMENT '非流动资产/总资产',
    tbassets_to_totalassets DECIMAL(20, 4) COMMENT '有形资产/总资产',
    int_to_talcap DECIMAL(20, 4) COMMENT '带息债务/全部投入资本',
    eqt_to_talcapital DECIMAL(20, 4) COMMENT '归属于母公司的股东权益/全部投入资本',
    currentdebt_to_debt DECIMAL(20, 4) COMMENT '流动负债/负债合计',
    longdeb_to_debt DECIMAL(20, 4) COMMENT '非流动负债/负债合计',
    ocf_to_shortdebt DECIMAL(20, 4) COMMENT '经营活动产生的现金流量净额/流动负债',
    debt_to_eqt DECIMAL(20, 4) COMMENT '产权比率',
    eqt_to_debt DECIMAL(20, 4) COMMENT '归属于母公司的股东权益/负债合计',
    eqt_to_interestdebt DECIMAL(20, 4) COMMENT '归属于母公司的股东权益/带息债务',
    tangibleasset_to_debt DECIMAL(20, 4) COMMENT '有形资产/负债合计',
    tangasset_to_intdebt DECIMAL(20, 4) COMMENT '有形资产/带息债务',
    tangibleasset_to_netdebt DECIMAL(20, 4) COMMENT '有形资产/净债务',
    ocf_to_debt DECIMAL(20, 4) COMMENT '经营活动产生的现金流量净额/负债合计',
    turn_days DECIMAL(20, 4) COMMENT '营业周期',
    roa_yearly DECIMAL(20, 4) COMMENT '年化总资产净利率',
    roa_dp DECIMAL(20, 4) COMMENT '总资产净利率(杜邦分析)',
    fixed_assets DECIMAL(20, 4) COMMENT '固定资产合计',
    profit_to_op DECIMAL(20, 4) COMMENT '利润总额／营业收入',
    q_saleexp_to_gr DECIMAL(20, 4) COMMENT '销售费用／营业总收入 (单季度)',
    q_gc_to_gr DECIMAL(20, 4) COMMENT '营业总成本／营业总收入 (单季度)',
    q_roe DECIMAL(20, 4) COMMENT '净资产收益率(单季度)',
    q_dt_roe DECIMAL(20, 4) COMMENT '净资产单季度收益率(扣除非经常损益)',
    q_npta DECIMAL(20, 4) COMMENT '总资产净利润(单季度)',
    q_ocf_to_sales DECIMAL(20, 4) COMMENT '经营活动产生的现金流量净额／营业收入(单季度)',
    basic_eps_yoy DECIMAL(20, 4) COMMENT '基本每股收益同比增长率(%)',
    dt_eps_yoy DECIMAL(20, 4) COMMENT '稀释每股收益同比增长率(%)',
    cfps_yoy DECIMAL(20, 4) COMMENT '每股经营活动产生的现金流量净额同比增长率(%)',
    op_yoy DECIMAL(20, 4) COMMENT '营业利润同比增长率(%)',
    ebt_yoy DECIMAL(20, 4) COMMENT '利润总额同比增长率(%)',
    netprofit_yoy DECIMAL(20, 4) COMMENT '归属母公司股东的净利润同比增长率(%)',
    dt_netprofit_yoy DECIMAL(20, 4) COMMENT '归属母公司股东的净利润-扣除非经常损益同比增长率(%)',
    ocf_yoy DECIMAL(20, 4) COMMENT '经营活动产生的现金流量净额同比增长率(%)',
    roe_yoy DECIMAL(20, 4) COMMENT '净资产收益率(摊薄)同比增长率(%)',
    bps_yoy DECIMAL(20, 4) COMMENT '每股净资产相对年初增长率(%)',
    assets_yoy DECIMAL(20, 4) COMMENT '资产总计相对年初增长率(%)',
    eqt_yoy DECIMAL(20, 4) COMMENT '归属母公司的股东权益相对年初增长率(%)',
    tr_yoy DECIMAL(20, 4) COMMENT '营业总收入同比增长率(%)',
    or_yoy DECIMAL(20, 4) COMMENT '营业收入同比增长率(%)',
    q_sales_yoy DECIMAL(20, 4) COMMENT '营业收入同比增长率(%)(单季度)',
    q_op_qoq DECIMAL(20, 4) COMMENT '营业利润环比增长率(%)(单季度)',
    equity_yoy DECIMAL(20, 4) COMMENT '净资产同比增长率',
    PRIMARY KEY (id)
    UNIQUE KEY uk_code_endday (ts_code, end_date)
) ENGINE=InnoDB
  DEFAULT CHARSET=utf8mb4
  COLLATE=utf8mb4_unicode_ci
  COMMENT='tushare-财务指标'
  AUTO_INCREMENT=1;

'''
# 缺失值统计：day='2025-07-11'
id	0
ts_code	0
day	0
ann_date	0
end_date	0
eps	10
dt_eps	187
total_revenue_ps	3
revenue_ps	4
capital_rese_ps	29
surplus_rese_ps	113
undist_profit_ps	1
extra_item	9
profit_dedt	9
gross_margin	143
current_ratio	119
quick_ratio	119
cash_ratio	135
ar_turn	171
ca_turn	137
fa_turn	136
assets_turn	2
op_income	0
ebit	135
ebitda	5539
fcff	150
fcfe	150
current_exint	135
noncurrent_exint	169
interestdebt	134
netdebt	134
tangible_asset	134
working_capital	135
networking_capital	135
invest_capital	134
retained_earnings	1
diluted2_eps	1
bps	1
ocfps	1
retainedps	1
cfps	1
ebit_ps	135
fcff_ps	150
fcfe_ps	150
netprofit_margin	9
grossprofit_margin	149
cogs_of_sales	149
expense_of_sales	144
profit_to_gr	8
saleexp_to_gr	273
adminexp_of_gr	9
finaexp_of_gr	146
impai_ttm	1627
gc_of_gr	143
op_of_gr	9
ebit_of_gr	143
roe	113
roe_waa	139
roe_dt	119
roa	135
npta	0
roic	184
roe_yearly	113
roa2_yearly	135
debt_to_assets	0
assets_to_eqt	111
dp_assets_to_eqt	113
ca_to_assets	135
nca_to_assets	135
tbassets_to_totalassets	134
int_to_talcap	189
eqt_to_talcapital	189
currentdebt_to_debt	135
longdeb_to_debt	169
ocf_to_shortdebt	135
debt_to_eqt	117
eqt_to_debt	0
eqt_to_interestdebt	410
tangibleasset_to_debt	134
tangasset_to_intdebt	410
tangibleasset_to_netdebt	4951
ocf_to_debt	1
turn_days	147
roa_yearly	0
roa_dp	0
fixed_assets	2
profit_to_op	9
q_saleexp_to_gr	665
q_gc_to_gr	555
q_roe	484
q_dt_roe	480
q_npta	409
q_ocf_to_sales	424
basic_eps_yoy	27
dt_eps_yoy	207
cfps_yoy	183
op_yoy	2
ebt_yoy	2
netprofit_yoy	1
dt_netprofit_yoy	100
ocf_yoy	2
roe_yoy	315
bps_yoy	1
assets_yoy	0
eqt_yoy	0
tr_yoy	9
or_yoy	10
q_sales_yoy	414
q_op_qoq	412
equity_yoy	172
'''








