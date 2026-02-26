#!/bin/bash

# 功能：权量更新 财务数据（PIT）
source ~/.bashrc

# 引入全局配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../config.sh"

cur_path=`pwd`
python_path="${PYTHON_PATH}"
qlib_path="${QLIB_PATH}"
cf_quant_path="${CF_QUANT_PATH}"
data_path="${QLIB_DATA_PATH}"
provider_uri="${PROVIDER_URI}"
# 读取mysql配置信息
config_file="${SCRIPT_DIR}/../../config.yaml"
mysql_user=$(sed -n '/^mysql:/,/^[^[:space:]]/p' "$config_file" | grep "^  user:" | sed 's/.*user:[[:space:]]*\(.*\)$/\1/' | sed 's/['\''\"]//g' | tr -d '[:space:]')
mysql_password=$(sed -n '/^mysql:/,/^[^[:space:]]/p' "$config_file" | grep "^  password:" | sed 's/.*password:[[:space:]]*\(.*\)$/\1/' | sed 's/[\'\''\"]//g' | tr -d '[:space:]')

echo "cur_path: ${cur_path}"

if [ $# -eq 0 ]; then
    dt1='2008-01-02'
    dt2=`date +%Y-%m-%d`
  elif [ $# -eq 1 ]; then
    dt1=$1
    dt2=`date +%Y-%m-%d`
  elif [ $# -eq 2 ]; then
    dt1=$1
    dt2=$2
  elif [ $# -eq 3 ]; then
    dt1=$1
    dt2=$2
    provider_uri=$3
  else
    echo "参数错误！！！"
    exit 1
fi

echo "时间区间为：[$dt1 - $dt2]"
echo "provider_uri: ${provider_uri}"

${python_path} ${cf_quant_path}/utils/is_trade_day.py ${dt2}
if [ $? -eq 5 ];then
  echo '非交易日！！！'
  exit 0
fi


check_success(){
  if [ $? -eq 0 ]; then
    echo "$1 执行成功！！！"
  else
    echo "$1 执行失败！！！"
    exit 1
  fi
}

get_data_from_mysql(){
  echo "get_data_from_mysql ..."
  # 提取字段 date、period、symbol、field、value
  # +------------+------------+----------+-----------------+-----------------+
  # | date       | period     | symbol   | field           | value           |
  # +------------+------------+----------+-----------------+-----------------+
  # | 2025-07-22 | 2020-12-31 | sz300500 | n_income        | 44590059.0900   |
  # | 2025-07-22 | 2020-12-31 | sz300500 | n_income_attr_p | 35645240.9500   |
  sql=$(cat <<-EOF
    SELECT
      t.f_ann_date AS date,
      t.end_date AS period,
      lower(t.qlib_code) AS symbol,
      jt.field,
      jt.value
    FROM
      cf_quant.income_ts t,   -- 利润表
      LATERAL (
        SELECT 'revenue' AS field, t.revenue AS value
        UNION ALL
        SELECT 'oper_cost', t.oper_cost
        UNION ALL
        SELECT 'n_income_attr_p', t.n_income_attr_p
        UNION ALL
        SELECT 'basic_eps', t.basic_eps
      ) AS jt
    WHERE
      t.f_ann_date >= '${dt1}' AND t.f_ann_date <= '${dt2}'

    UNION ALL

    SELECT
      c.f_ann_date AS date,
      c.end_date AS period,
      lower(c.qlib_code) AS symbol,
      jc.field,
      jc.value
    FROM
      cf_quant.cashflow_ts c,   -- 现金流量表
      LATERAL (
        SELECT 'n_incr_cash_cash_equ' AS field, c.n_incr_cash_cash_equ AS value
        UNION ALL
        SELECT 'n_cashflow_act', c.n_cashflow_act
        UNION ALL
        SELECT 'c_pay_acq_const_fiolta', c.c_pay_acq_const_fiolta
      ) AS jc
    WHERE
      c.f_ann_date >= '${dt1}' AND c.f_ann_date <= '${dt2}'
EOF
)
  echo "${sql}"
  mysql -u"${mysql_user}" -p"${mysql_password}" -e "${sql}" > "${provider_uri}/pit_${dt1}_${dt2}.csv"
  check_success "从mysql中导出数据"
}


split_stock(){
  echo "split_stock ..."
  ${python_path} "${cur_path}"/split_stock.py \
  --path_in "${provider_uri}/pit_${dt1}_${dt2}.csv" \
  --path_out "${provider_uri}" \
  --start_date "${dt1}" \
  --end_date "${dt2}"
  check_success "切分股票数据"
}


format(){
  echo "format ..."
  ${python_path} ${qlib_path}/scripts/data_collector/pit/collector.py normalize_data \
  --interval quarterly \
  --source_dir "${provider_uri}/pit_${dt1}_${dt2}" \
  --normalize_dir "${provider_uri}/pit_normalized_${dt1}_${dt2}"
  check_success "格式化"
}

trans_to_qlib(){
  echo "trans_to_qlib..."
  ${python_path} ${qlib_path}/scripts/dump_pit.py dump \
  --csv_path "${provider_uri}/pit_normalized_${dt1}_${dt2}" \
  --qlib_dir "${provider_uri}" \
  --interval quarterly
  check_success "转化为qlib格式"
}


function_set(){
  get_data_from_mysql
  split_stock
  format
  trans_to_qlib
}

main(){
  function_set
  if [ $? -eq 0 ];then
      echo "PIT 执行完成 ！！！"
    else
      echo "PIT 执行失败！！！"
  fi
}

main

