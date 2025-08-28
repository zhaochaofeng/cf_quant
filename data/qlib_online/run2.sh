#!/bin/bash

# 功能：每日全量更新qlib线上数据
source ~/.bashrc
cur_path=`pwd`
python_path="/root/anaconda3/envs/python3/bin/python"
qlib_path="/root/qlib"
cf_quant_path="/root/cf_quant"
data_path="/root/.qlib/qlib_data"
provider_uri="${data_path}/custom_data_hfq"
provider_uri_tmp="${data_path}/custom_data_hfq_tmp"
provider_uri_bak="${data_path}/custom_data_hfq_bak"

echo 'cur_path: '${cur_path}

dt1='2015-01-05'
dt2=`date +%Y-%m-%d`
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

check_success() {
  if [ $? -eq 0 ]; then
    echo "$1 执行成功！！！"
  else
    echo "$1 执行失败！！！"
    exit 1
  fi
}

get_data_from_mysql(){
  echo "get_data_from_mysql ..."
  mysql -uchaofeng -pZhao_123 -e "select ts_code, day as date, open, close, high, low, vol, amount, adj_factor \
  from cf_quant.trade_daily2 \
  where day>='${dt1}' and day<='${dt2}';" > ${provider_uri_tmp}/custom_${dt1}_${dt2}.csv
  check_success "从mysql中导出数据"
}

process_data(){
  echo "process_data..."
  ${python_path} ${cur_path}/process.py main \
  --provider_uri ${provider_uri_tmp} \
  --start_date ${dt1} \
  --end_date ${dt2} \
  --is_offline True \
  --path_in ${provider_uri_tmp}/custom_${dt1}_${dt2}.csv

  return $?
}

trans_to_qlib(){
  echo "trans_to_qlib..."
  ${python_path} ${qlib_path}/scripts/dump_bin.py dump_all \
  --date_field_name date \
  --data_path ${provider_uri_tmp}/out_${dt1}_${dt2} \
  --qlib_dir ${provider_uri_tmp} \
  --include_fields open,close,high,low,volume,amount,factor
  check_success "转化为qlib格式"
}

update(){
  echo "update..."
    if [ -d "${provider_uri_bak}" ]; then
    rm -rf "${provider_uri_bak}"
  fi

  mv ${provider_uri} ${provider_uri_bak}
  mv ${provider_uri_tmp} ${provider_uri}
  check_success "替换历史数据"
}

function_set(){
  create_tmp_dir
  get_data_from_mysql
  process_data
  trans_to_qlib
  update
}

main(){
#  function_set
  ${python_path} ${cur_path}/check.py
#  if [ $? -eq 10 ];then
#    echo "mysql 复权因子更新，重新处理数据..."
#    function_set
#  fi
#  echo "执行完成 ！！！"
}

main
