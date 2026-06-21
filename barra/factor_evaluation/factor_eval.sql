-- 因子评价指标

CREATE TABLE IF NOT EXISTS factor_evaluation(
    id INT NOT NULL AUTO_INCREMENT,
    day DATE NOT NULL COMMENT '计算日期，格式YYYY-MM-DD',
    name VARCHAR(50) NOT NULL COMMENT '因子名称',
    type VARCHAR(50) NOT NULL COMMENT '因子类型. risk: 风险因子; alpha: alpha因子',
    IC Decimal(10, 6) COMMENT 'Normal IC',
    ICIR Decimal(10, 6) COMMENT 'Noram IC 的信息比率',
    RIC Decimal(10, 6) COMMENT 'Rank IC',
    RICIR Decimal(10, 6) COMMENT 'Rank IC 的信息比率',
    long_short Decimal(10, 6) COMMENT '多空组合超额收益率',
    avg_return Decimal(10, 6) COMMENT '市场平均超额收益率(接近0)',
    half_life DECIMAL(10, 2) COMMENT '半衰期（天）',
    monotonic_tstat DECIMAL(10, 6) COMMENT '分组单调性t统计量',
    ls_alpha DECIMAL(10, 8) COMMENT '多空Jensen alpha',
    ls_alpha_tstat DECIMAL(10, 6) COMMENT '多空alpha t统计量',
    ls_beta DECIMAL(10, 6) COMMENT '多空市场beta',
    ls_ir DECIMAL(10, 6) COMMENT '多空收益率信息比率',
    ls_cum_return_1y DECIMAL(10, 6) COMMENT '多空1年累计超额收益率',
    PRIMARY KEY (id),
    UNIQUE KEY un_day (day, name, type)
) ENGINE=InnoDB
    DEFAULT CHARSET=utf8mb4
    COLLATE=utf8mb4_unicode_ci
    COMMENT='因子评价指标'
    AUTO_INCREMENT=1;


