CREATE TABLE IF NOT EXISTS trade_daily(
    id INT NOT NULL AUTO_INCREMENT,
    ts_code VARCHAR(30) NOT NULL COMMENT 'TS股票代码，股票代码带BJ/SH/SZ后缀',
    day VARCHAR(10) NOT NULL COMMENT '取数日期，格式YYYY-MM-DD',
    open DECIMAL(20, 4) COMMENT '开盘价',
    high DECIMAL(20, 4) COMMENT '最高价',
    low DECIMAL(20, 4) COMMENT '最低价',
    close DECIMAL(20, 4) COMMENT '收盘价',
    pre_close DECIMAL(20, 4) COMMENT '昨收价【除权价，前复权】',
    `change` DECIMAL(20, 4) COMMENT '涨跌额',
    pct_chg DECIMAL(20, 4) COMMENT '涨跌幅 【基于除权后的昨收计算的涨跌幅：（今收-除权昨收）/除权昨收 】',
    vol DECIMAL(20, 4) COMMENT '成交量 （手）',
    amount DECIMAL(20, 4) COMMENT '成交额 （千元）',
    PRIMARY KEY (id),
    UNIQUE KEY uk_code_day (ts_code, day)
) ENGINE=InnoDB
  DEFAULT CHARSET=utf8mb4
  COLLATE=utf8mb4_unicode_ci
  COMMENT='日线行情数据'
  AUTO_INCREMENT=1;



