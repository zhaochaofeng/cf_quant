CREATE TABLE IF NOT EXISTS valuation_ts(
    id INT NOT NULL AUTO_INCREMENT,
    ts_code VARCHAR(10) NOT NULL COMMENT 'TS股票代码，股票代码带BJ/SH/SZ后缀',
    qlib_code VARCHAR(10) NOT NULL COMMENT 'Qlib股票代码, 如 SZ000001',
    day DATE NOT NULL COMMENT '交易日期，格式YYYY-MM-DD',
    close DECIMAL(9, 2) COMMENT '当日收盘价',
    turnover_rate DECIMAL(9, 4) COMMENT '换手率（%）',
    turnover_rate_f DECIMAL(9, 4) COMMENT '换手率（自由流通股）(%)',
    volume_ratio DECIMAL(9, 2) COMMENT '量比',
    pe DECIMAL(20, 4) COMMENT '市盈率（总市值/净利润， 亏损的PE为空）',
    pe_ttm DECIMAL(20, 4) COMMENT '市盈率（TTM，亏损的PE为空）',
    pb DECIMAL(20, 4) COMMENT '市净率（总市值/净资产）',
    ps DECIMAL(20, 4) COMMENT '市销率',
    ps_ttm DECIMAL(20, 4) COMMENT '市销率（TTM）',
    dv_ratio DECIMAL(12, 4) COMMENT '股息率 （%）',
    dv_ttm DECIMAL(12, 4) COMMENT '股息率（TTM）（%）',
    total_share DECIMAL(20, 4) COMMENT '总股本（万股）',
    float_share DECIMAL(20, 4) COMMENT '流通股本（万股）',
    free_share DECIMAL(20, 4) COMMENT '自由流通股本（万股）',
    total_mv DECIMAL(20, 4) COMMENT '总市值 （万元）',
    circ_mv DECIMAL(20, 4) COMMENT '流通市值（万元）',
    PRIMARY KEY (id),
    UNIQUE KEY uk_day_code(day, ts_code),
    INDEX qlib_index(qlib_code)
) ENGINE=InnoDB
  DEFAULT CHARSET=utf8mb4
  COLLATE=utf8mb4_unicode_ci
  COMMENT='tushare-市值数据'
  AUTO_INCREMENT=1;
