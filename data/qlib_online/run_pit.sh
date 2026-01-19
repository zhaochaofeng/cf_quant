#!/bin/bash

# 功能：权量更新 财务数据（PIT）
source ~/.bashrc
cur_path=`pwd`
python_path="/root/anaconda3/envs/python3/bin/python"
qlib_path="/root/qlib"
cf_quant_path="/root/cf_quant"
data_path="/root/.qlib/qlib_data"
provider_uri="${data_path}/custom_data_hfq"

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
      cf_quant.income_ts t,
      LATERAL (
        SELECT 'basic_eps' AS field, t.basic_eps AS value
        UNION ALL
        SELECT 'total_revenue', t.total_revenue
        UNION ALL
        SELECT 'revenue', t.revenue
        UNION ALL
        SELECT 'non_oper_income', t.non_oper_income
        UNION ALL
        SELECT 'non_oper_exp', t.non_oper_exp
        UNION ALL
        SELECT 'n_income_attr_p', t.n_income_attr_p
        UNION ALL
        SELECT 'operate_profit', t.operate_profit
        UNION ALL
        SELECT 'total_profit', t.total_profit
        UNION ALL
        SELECT 'ebit', t.ebit
        UNION ALL
        SELECT 'ebitda', t.ebitda
        UNION ALL
        SELECT 'continued_net_profit', t.continued_net_profit
        UNION ALL
        SELECT 'oper_cost', t.oper_cost
        UNION ALL
        SELECT 'total_cogs', t.total_cogs
        UNION ALL
        SELECT 'admin_exp', t.admin_exp
        UNION ALL
        SELECT 'rd_exp', t.rd_exp
        UNION ALL
        SELECT 'sell_exp', t.sell_exp
        UNION ALL
        SELECT 'fin_exp', t.fin_exp
        UNION ALL
        SELECT 'fin_exp_int_exp', t.fin_exp_int_exp
        UNION ALL
        SELECT 'fin_exp_int_inc', t.fin_exp_int_inc
        UNION ALL
        SELECT 'assets_impair_loss', t.assets_impair_loss
        UNION ALL
        SELECT 'invest_income', t.invest_income
        UNION ALL
        SELECT 'compr_inc_attr_p', t.compr_inc_attr_p
        UNION ALL
        SELECT 'diluted_eps', t.diluted_eps
        UNION ALL
        SELECT 'income_tax', t.income_tax
        UNION ALL
        SELECT 'int_income', t.int_income
        UNION ALL
        SELECT 'n_commis_income', t.n_commis_income
        UNION ALL
        SELECT 'prem_earned', t.prem_earned
        UNION ALL
        SELECT 'fv_value_chg_gain', t.fv_value_chg_gain
      ) AS jt
    WHERE
      t.f_ann_date >= '${dt1}' AND t.f_ann_date <= '${dt2}';
EOF
)
  echo "${sql}"
  mysql -uchaofeng -pZhao_123 -e "${sql}" > "${provider_uri}/pit_${dt1}_${dt2}.csv"
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

