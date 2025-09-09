CREATE TABLE IF NOT EXISTS stock_info(
    id INT NOT NULL AUTO_INCREMENT,
    ts_code VARCHAR(30) NOT NULL COMMENT 'TS股票代码，股票代码带BJ/SH/SZ后缀',
    code VARCHAR(30) NOT NULL COMMENT '股票代码',
    day VARCHAR(10) NOT NULL COMMENT '取数日期，格式YYYY-MM-DD',
    name VARCHAR(30) NOT NULL COMMENT '股票名称(包含ST标识)',
    area VARCHAR(30) COMMENT '所属地域',
    industry VARCHAR(30) COMMENT '所属行业',
    cnspell VARCHAR(30) COMMENT '拼音缩写',
    market VARCHAR(30) COMMENT '市场类型（主板/创业板/科创板/北交所 ）',
    list_date VARCHAR(10) COMMENT '上市日期，格式YYYY-MM-DD',
    act_name VARCHAR(50) COMMENT '实控人名称',
    act_ent_type VARCHAR(50) COMMENT '实控人企业性质',
    exchange VARCHAR(30) COMMENT '交易所代码。SSE:上交所; SZSE:深交所; BSE:北交所',
    PRIMARY KEY (id),
    UNIQUE KEY un_code_day (code, day)
) ENGINE=InnoDB
    DEFAULT CHARSET=utf8mb4
    COLLATE=utf8mb4_unicode_ci
    COMMENT='股票信息表'
    AUTO_INCREMENT=1;


################ Tushare 数据源 ####################
CREATE TABLE IF NOT EXISTS stock_info_ts(
    id INT NOT NULL AUTO_INCREMENT,
    ts_code VARCHAR(10) NOT NULL COMMENT 'TS股票代码，股票代码带BJ/SH/SZ后缀, 如 000001.SZ',
    qlib_code VARCHAR(10) NOT NULL COMMENT 'Qlib股票代码, 如 SZ000001',
    day DATE NOT NULL COMMENT '取数日期，格式YYYY-MM-DD',
    name VARCHAR(30) NOT NULL COMMENT '股票名称(包含ST标识)',
    area VARCHAR(20) COMMENT '所属地域',
    industry VARCHAR(20) COMMENT '所属行业',
    cnspell VARCHAR(10) COMMENT '拼音缩写',
    market VARCHAR(10) COMMENT '市场类型（主板/创业板/科创板/北交所 ）',
    list_date DATE COMMENT '上市日期，格式YYYY-MM-DD',
    act_name VARCHAR(50) COMMENT '实控人名称',
    act_ent_type VARCHAR(50) COMMENT '实控人企业性质',
    exchange VARCHAR(10) COMMENT '交易所代码。SSE:上交所; SZSE:深交所; BSE:北交所',
    status TINYINT COMMENT '股票状态。1:上市; 0:退市',
    PRIMARY KEY (id),
    UNIQUE KEY un_code_day (day, ts_code),
    INDEX qlib_idx (qlib_code)
) ENGINE=InnoDB
    DEFAULT CHARSET=utf8mb4
    COLLATE=utf8mb4_unicode_ci
    COMMENT='Tushare股票信息表'
    AUTO_INCREMENT=1;



################ BaoStock 数据源 ####################
CREATE TABLE IF NOT EXISTS stock_info_bao(
    id INT NOT NULL AUTO_INCREMENT,
    code VARCHAR(30) NOT NULL COMMENT '股票代码，股票代码带sh/sz前缀，如 sz.000001',
    qlib_code VARCHAR(10) NOT NULL COMMENT 'Qlib股票代码, 如 SZ000001',
    day DATE NOT NULL COMMENT '取数日期，格式YYYY-MM-DD',
    name VARCHAR(30) NOT NULL COMMENT '股票名称(包含ST标识)',
    list_date DATE COMMENT '上市日期，格式YYYY-MM-DD',
    out_date DATE COMMENT '退市日期，格式YYYY-MM-DD',
    exchange VARCHAR(30) COMMENT '交易所代码。SSE:上交所; SZSE:深交所',
    status TINYINT COMMENT '股票状态。1:上市; 0:退市',
    PRIMARY KEY (id),
    UNIQUE KEY un_code_day (day, code),
    INDEX qlib_idx (qlib_code)
) ENGINE=InnoDB
    DEFAULT CHARSET=utf8mb4
    COLLATE=utf8mb4_unicode_ci
    COMMENT='BaoStock股票信息表'
    AUTO_INCREMENT=1;


