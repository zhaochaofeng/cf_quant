#################### Tushare数据 ###########################
CREATE TABLE IF NOT EXISTS trade_daily_ts(
    id INT NOT NULL AUTO_INCREMENT,
    ts_code VARCHAR(10) NOT NULL COMMENT 'TS股票代码，股票代码带BJ/SH/SZ后缀',
    qlib_code VARCHAR(10) NOT NULL COMMENT 'Qlib股票代码, 如 SZ000001',
    day DATE NOT NULL COMMENT '取数日期，格式YYYY-MM-DD',
    open DECIMAL(9, 2) COMMENT '开盘价',
    close DECIMAL(9, 2) COMMENT '收盘价',
    high DECIMAL(9, 2) COMMENT '最高价',
    low DECIMAL(9, 2) COMMENT '最低价',
    pre_close DECIMAL(9, 2) COMMENT '昨日收盘价',
    `change` DECIMAL(9, 2) COMMENT '收盘价涨跌额',
    pct_chg DECIMAL(6, 2) COMMENT '收盘价涨跌幅（%）',
    vol DECIMAL(15, 2) COMMENT '成交量（手）',
    amount DECIMAL(15, 3) COMMENT '成交额（千元）',
    adj_factor DECIMAL(9, 4) COMMENT '复权因子',
    PRIMARY KEY (id),
    UNIQUE KEY uk_code_day(day, ts_code),
    INDEX qlib_index(qlib_code)
) ENGINE=InnoDB
  DEFAULT CHARSET=utf8mb4
  COLLATE=utf8mb4_unicode_ci
  COMMENT='日线行情数据(原始数据)'
  AUTO_INCREMENT=1;


#################### BaoStock数据 ###########################
CREATE TABLE IF NOT EXISTS trade_daily_bao(
    id INT NOT NULL AUTO_INCREMENT,
    code VARCHAR(30) NOT NULL COMMENT '股票代码，股票代码带sh/sz前缀，如 sz.000001',
    qlib_code VARCHAR(10) NOT NULL COMMENT 'Qlib股票代码, 如 SZ000001',
    day DATE NOT NULL COMMENT '取数日期，格式YYYY-MM-DD',
    open DECIMAL(9, 2) COMMENT '开盘价',
    close DECIMAL(9, 2) COMMENT '收盘价',
    high DECIMAL(9, 2) COMMENT '最高价',
    low DECIMAL(9, 2) COMMENT '最低价',
    pre_close DECIMAL(9, 2) COMMENT '昨日收盘价',
    pct_chg DECIMAL(6, 2) COMMENT '收盘价涨跌幅(%)',
    vol DECIMAL(15, 2) COMMENT '成交量 (手）',
    amount DECIMAL(15, 3) COMMENT '成交额 （千元）',
    adj_factor DECIMAL(9, 4) COMMENT '复权因子',
    is_st TINYINT COMMENT '是否ST股票. 1:是;0:否',
    PRIMARY KEY (id),
    UNIQUE KEY uk_code_day(day, code),
    INDEX qlib_index(qlib_code)
) ENGINE=InnoDB
  DEFAULT CHARSET=utf8mb4
  COLLATE=utf8mb4_unicode_ci
  COMMENT='日线行情数据(原始数据)'
  AUTO_INCREMENT=1;
