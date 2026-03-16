-- 风控指标表 --

CREATE TABLE IF NOT EXISTS factor_risk(
    id INT NOT NULL AUTO_INCREMENT,
    day DATE NOT NULL COMMENT '计算日期，格式YYYY-MM-DD',
    name VARCHAR(50) NOT NULL COMMENT '因子名称',
    type VARCHAR(50) NOT NULL COMMENT '因子类型',
    FMCAR Decimal(10, 6) NOT NULL COMMENT '因子主动风险边际贡献',
    FRCAR Decimal(10, 6) NOT NULL COMMENT '因子主动风险贡献',
    PRIMARY KEY (id),
    UNIQUE KEY un_day (day)
) ENGINE=InnoDB
    DEFAULT CHARSET=utf8mb4
    COLLATE=utf8mb4_unicode_ci
    COMMENT='因子风险指标'
    AUTO_INCREMENT=1;

CREATE TABLE IF NOT EXISTS portfolio_risk(
    id INT NOT NULL AUTO_INCREMENT,
    day DATE NOT NULL COMMENT '计算日期，格式YYYY-MM-DD',
    qlib_code VARCHAR(10) NOT NULL COMMENT 'Qlib股票代码, 如 SZ000001',
    portfolio VARCHAR(50) NOT NULL COMMENT '持仓组合名称',
    MCAR Decimal(10, 6) NOT NULL COMMENT '股票主动风险边际贡献',
    RCAR Decimal(10, 6) NOT NULL COMMENT '股票主动风险贡献',
    PRIMARY KEY (id),
    UNIQUE KEY un_day (day)
) ENGINE=InnoDB
    DEFAULT CHARSET=utf8mb4
    COLLATE=utf8mb4_unicode_ci
    COMMENT='持仓组合风险指标'
    AUTO_INCREMENT=1;



