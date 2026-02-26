#!/bin/bash

# 功能：每日全量更新qlib线上数据
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
provider_uri_tmp="${data_path}/custom_data_hfq_tmp"
provider_uri_bak="${data_path}/custom_data_hfq_bak"
# 读取mysql配置信息
config_file="${SCRIPT_DIR}/../../config.yaml"
mysql_user=$(sed -n '/^mysql:/,/^[^[:space:]]/p' "$config_file" | grep "^  user:" | sed 's/.*user:[[:space:]]*\(.*\)$/\1/' | sed 's/['\''\"]//g' | tr -d '[:space:]')
mysql_password=$(sed -n '/^mysql:/,/^[^[:space:]]/p' "$config_file" | grep "^  password:" | sed 's/.*password:[[:space:]]*\(.*\)$/\1/' | sed 's/[\'\''\"]//g' | tr -d '[:space:]')

echo 'cur_path: '${cur_path}

if [ $# -eq 0 ]; then
    dt1='2008-01-02'
    dt2=`date +%Y-%m-%d`
  elif [ $# -eq 1 ]; then
    dt1=$1
    dt2=`date +%Y-%m-%d`
  elif [ $# -eq 2 ]; then
    dt1=$1
    dt2=$2
  else
    echo "参数错误！！！"
    exit 1
fi
echo "时间区间为：["$dt1" - "$dt2"]"

${python_path} ${cf_quant_path}/utils/is_trade_day.py ${dt2}
if [ $? -eq 5 ];then
  echo '非交易日！！！'
  exit 0
fi

create_tmp_dir(){
  echo "create_tmp_dir ..."
  if [ -d "${provider_uri_tmp}" ]; then
    rm -rf "${provider_uri_tmp}"
  fi
  mkdir -p "${provider_uri_tmp}"
}

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

  sql=$(cat <<-EOF
    SELECT
      a.ts_code, a.date, a.open, a.close, a.high, a.low, a.vol, a.amount, a.adj_factor,
      b.ind_one, b.ind_two, b.ind_three,
      c.total_share, c.float_share, c.total_mv, c.circ_mv
    FROM
      (select ts_code, day as date, open, close, high, low, vol, amount, adj_factor
      from cf_quant.trade_daily_ts
      where day >= '${dt1}' and day <= '${dt2}'
      )a

    JOIN
      (select ts_code, left(l1_code, 6) as ind_one, left(l2_code, 6) as ind_two, left(l3_code, 6) as ind_three
      from cf_quant.stock_info_ts where day='${dt2}'
      and exchange in ('SSE', 'SZSE')
      )b
    ON
      a.ts_code=b.ts_code

    LEFT JOIN
      (select ts_code, day as date, total_share, float_share, total_mv, circ_mv from cf_quant.valuation_ts
      where day >= '${dt1}' and day <= '${dt2}'
      )c
    ON
      a.ts_code = c.ts_code and a.date = c.date
EOF
)
  echo "${sql}"
  mysql -u"${mysql_user}" -p"${mysql_password}" -e "${sql}" > ${provider_uri_tmp}/custom_${dt1}_${dt2}.csv
  check_success "从mysql中导出数据"
}

process_data(){
  echo "process_data..."
  ${python_path} ${cur_path}/process.py main \
  --provider_uri ${provider_uri_tmp} \
  --start_date ${dt1} \
  --end_date ${dt2} \
  --is_offline True \
  --path_in ${provider_uri_tmp}/custom_${dt1}_${dt2}.csv \
  --columns "['ts_code', 'date', 'open', 'close', 'high', 'low', 'volume', 'amount', 'factor', 'change', \
  'ind_one', 'ind_two', 'ind_three']" \
  --index_list "['000300.SH', '000905.SH', '000903.SH', '000906.SH']"
  check_success "复权等数据处理"
}

trans_to_qlib(){
  echo "trans_to_qlib..."
  ${python_path} ${qlib_path}/scripts/dump_bin.py dump_all \
  --date_field_name date \
  --data_path ${provider_uri_tmp}/out_${dt1}_${dt2} \
  --qlib_dir ${provider_uri_tmp} \
  --include_fields open,close,high,low,volume,amount,factor,change,ind_one,ind_two,ind_three
  check_success "转化为qlib格式"
}

update(){
  echo "update..."
    if [ -d "${provider_uri_bak}" ]; then
    rm -rf "${provider_uri_bak}"
  fi
  rm -rf ${provider_uri_tmp}/custom_${dt1}_${dt2}.csv
  rm -rf ${provider_uri_tmp}/out_${dt1}_${dt2}
  rm -rf ${provider_uri_tmp}/pit_${dt1}_${dt2}.csv
  rm -rf ${provider_uri_tmp}/pit_${dt1}_${dt2}
  rm -rf ${provider_uri_tmp}/pit_normalized_${dt1}_${dt2}
  cp ${data_path}/index/instruments/csi*.txt ${provider_uri_tmp}/instruments/
  if [ -d "${provider_uri}" ];then
    mv ${provider_uri} ${provider_uri_bak}
  fi
  mv ${provider_uri_tmp} ${provider_uri}
  check_success "替换历史数据"
}

process_pit(){
  echo "process_pit ..."
  sh "${cur_path}/run_pit.sh" "${dt1}" "${dt2}" "${provider_uri_tmp}"
  check_success "处理 PIT 数据"
}

function_set(){
  create_tmp_dir
  get_data_from_mysql
  process_data
  trans_to_qlib
}

main(){
  function_set
  process_pit
  update
  if [ $? -eq 0 ];then
      echo "执行完成 ！！！"
    else
      msg="执行失败！！！"
      echo ${msg}
      ${python_path} ${cf_quant_path}/utils/send_email.py "Data: qlib_online" "${msg}"
      exit 1
  fi
}

main

