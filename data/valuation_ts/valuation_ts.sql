CREATE TABLE IF NOT EXISTS valuation_tushare(
    id INT NOT NULL AUTO_INCREMENT,
    ts_code VARCHAR(30) NOT NULL COMMENT 'TS股票代码，股票代码带BJ/SH/SZ后缀',
    day VARCHAR(10) NOT NULL COMMENT '交易日期，格式YYYY-MM-DD',
    close DECIMAL(20, 4) COMMENT '当日收盘价',
    turnover_rate DECIMAL(20, 4) COMMENT '换手率（%）',
    turnover_rate_f DECIMAL(20, 4) COMMENT '换手率（自由流通股）(%)',
    volume_ratio DECIMAL(20, 4) COMMENT '量比',
    pe DECIMAL(20, 4) COMMENT '市盈率（总市值/净利润， 亏损的PE为空）',
    pe_ttm DECIMAL(20, 4) COMMENT '市盈率（TTM，亏损的PE为空）',
    pb DECIMAL(20, 4) COMMENT '市净率（总市值/净资产）',
    ps DECIMAL(20, 4) COMMENT '市销率',
    ps_ttm DECIMAL(20, 4) COMMENT '市销率（TTM）',
    dv_ratio DECIMAL(20, 4) COMMENT '股息率 （%）',
    dv_ttm DECIMAL(20, 4) COMMENT '股息率（TTM）（%）',
    total_share DECIMAL(20, 4) COMMENT '总股本 （万股）',
    float_share DECIMAL(20, 4) COMMENT '流通股本 （万股）',
    free_share DECIMAL(20, 4) COMMENT '自由流通股本 （万）',
    total_mv DECIMAL(20, 4) COMMENT '总市值 （万元）',
    circ_mv DECIMAL(20, 4) COMMENT '流通市值（万元）',
    PRIMARY KEY (id),
    UNIQUE KEY uk_code_day (ts_code, day)
) ENGINE=InnoDB
  DEFAULT CHARSET=utf8mb4
  COLLATE=utf8mb4_unicode_ci
  COMMENT='tushare-市值数据'
  AUTO_INCREMENT=1;


'''
# Nan值分布。date='2025-06-30'
df.isna().sum(axis=0)
Out[110]:
id                    0
ts_code               0
day                   0
close                 0
turnover_rate         0
turnover_rate_f       0
volume_ratio          1
pe                 1368
pe_ttm             1445
pb                   30
ps                    0
ps_ttm                1
dv_ratio            917
dv_ttm             2365
total_share           0
float_share           0
free_share            0
total_mv              0
circ_mv               0
'''




