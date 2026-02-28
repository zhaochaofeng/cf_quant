################ Tushare 数据源 ####################
CREATE TABLE IF NOT EXISTS stock_info_ts(
    id INT NOT NULL AUTO_INCREMENT,
    ts_code VARCHAR(10) NOT NULL COMMENT 'TS股票代码，股票代码带BJ/SH/SZ后缀, 如 000001.SZ',
    qlib_code VARCHAR(10) NOT NULL COMMENT 'Qlib股票代码, 如 SZ000001',
    day DATE NOT NULL COMMENT '取数日期，格式YYYY-MM-DD',
    name VARCHAR(30) NOT NULL COMMENT '股票名称(包含ST标识)',
    area VARCHAR(20) COMMENT '所属地域',
    industry VARCHAR(20) COMMENT '所属行业',
    fullname VARCHAR(100) COMMENT '股票全称',
    enname VARCHAR(100) COMMENT '英文全称',
    cnspell VARCHAR(10) COMMENT '拼音缩写',
    market VARCHAR(10) COMMENT '市场类型（主板/创业板/科创板/北交所）',
    exchange VARCHAR(10) COMMENT '交易所代码。SSE:上交所; SZSE:深交所; BSE:北交所',
    curr_type VARCHAR(10) COMMENT '交易货币',
    status TINYINT COMMENT '股票状态。1:上市; 0:退市; 2:暂停上市; 3:过会未交易',
    list_date DATE COMMENT '上市日期，格式YYYY-MM-DD',
    delist_date DATE COMMENT '退市日期，格式YYYY-MM-DD',
    is_hs VARCHAR(10) COMMENT '是否沪深港通标的. N否; H沪股通; S深股通',
    act_name VARCHAR(50) COMMENT '实控人名称',
    act_ent_type VARCHAR(50) COMMENT '实控人企业性质',
    l1_code VARCHAR(30) COMMENT '申万一级行业代码',
    l1_name VARCHAR(30) COMMENT '申万二级行业名称',
    l2_code VARCHAR(30) COMMENT '申万二级行业代码',
    l2_name VARCHAR(30) COMMENT '申万二级行业名称',
    l3_code VARCHAR(30) COMMENT '申万三级行业代码',
    l3_name VARCHAR(30) COMMENT '申万三级行业名称',
    in_date VARCHAR(30) COMMENT '纳入行业日期',
    out_date VARCHAR(30) COMMENT '剔除行业日期',
    is_new VARCHAR(1) COMMENT '是否最新. Y是;N否',
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


