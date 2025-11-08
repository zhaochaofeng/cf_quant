###################### 因子评价指标 #####################
CREATE TABLE IF NOT EXISTS factor_eval(
    id INT NOT NULL AUTO_INCREMENT,
    class VARCHAR(50) NOT NULL COMMENT '因子类别, 如 动量因子',
    code VARCHAR(50) NOT NULL COMMENT '因子, 如MA5',
    name VARCHAR(50) NOT NULL COMMENT '因子中文名称',
    day DATE NOT NULL COMMENT '日期，格式YYYY-MM-DD',
    IC DECIMAL(9, 4) COMMENT 'IC值',
    RIC DECIMAL(9, 4) COMMENT 'RIC值',
    PRIMARY KEY (id),
    UNIQUE KEY uk_code_day(code, day)
) ENGINE=InnoDB
  DEFAULT CHARSET=utf8mb4
  COLLATE=utf8mb4_unicode_ci
  COMMENT='因子评估指标'
  AUTO_INCREMENT=1;

###################### 因子库 #####################
CREATE TABLE IF NOT EXISTS factor(
    id INT NOT NULL AUTO_INCREMENT,
    qlib_code VARCHAR(10) NOT NULL COMMENT 'Qlib股票代码, 如 SZ000001',
    class VARCHAR(50) NOT NULL COMMENT '因子类别, 如 动量因子',
    code VARCHAR(50) NOT NULL COMMENT '因子, 如MA5',
    name VARCHAR(50) NOT NULL COMMENT '因子中文名称',
    day DATE NOT NULL COMMENT '日期，格式YYYY-MM-DD',
    value DECIMAL(9, 4) COMMENT '因子值',
    PRIMARY KEY (id),
    UNIQUE KEY uk_code_day(code, day, qlib_code),
    INDEX qlib_index(qlib_code)
) ENGINE=InnoDB
  DEFAULT CHARSET=utf8mb4
  COLLATE=utf8mb4_unicode_ci
  COMMENT='因子库'
  AUTO_INCREMENT=1;



