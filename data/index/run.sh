#!/bin/bash
source ~/.bashrc

## 构建指数成分股集合
cur_path=`pwd`
python_path="/root/anaconda3/envs/python3/bin/python"
qlib_path="/root/qlib"
cf_quant_path="/root/cf_quant"
provider_uri='/root/.qlib/qlib_data/index'


if [ $# -eq 0 ]; then
    dt=`date +%Y-%m-%d`
  elif [ $# -eq 1 ]; then
    dt=$1
  else
    echo "参数错误！！！"
    exit 1
fi
echo "cur_path: ${cur_path}"
echo "dt: "${dt}


${python_path} ${cf_quant_path}/utils/is_trade_day.py ${dt}
if [ $? -eq 5 ];then
  echo '非交易日！！！'
  exit 0
fi


if [ ! -d ${provider_uri} ]; then
  mkdir -p ${provider_uri}
fi


retry_process() {
  # 重试函数
  local cmd="$1"
  local retry_count=0
  local status=1
  echo "$cmd"
  while (( retry_count < 5 )); do
    echo "retry_count: ${retry_count}"
    eval "$cmd"
    status=$?
    (( status == 0 )) && return 0
    (( retry_count++ ))
    sleep 1h
  done
  return $status
}


check_success(){
  # 执行结果检查函数
  if [ $? -eq 0 ]; then
    echo "$1 执行成功！！！"
  else
    err_msg="$1 执行失败！！！"
    echo "${err_msg}"
    ${python_path} ${cf_quant_path}/utils/send_email.py "Data: index" "${err_msg}"
    exit 1
  fi
}


# CSI300
retry_process "
  ${python_path} ${qlib_path}/scripts/data_collector/cn_index/collector.py \
  --index_name CSI300 \
  --qlib_dir ${provider_uri} \
  --method parse_instruments
"
check_success "CSI300"


# CSI500
retry_process "
  ${python_path} ${qlib_path}/scripts/data_collector/cn_index/collector.py \
  --index_name CSI500 \
  --qlib_dir ${provider_uri} \
  --method parse_instruments \
  --request_retry 5 \
  --retry_sleep 10
"
check_success "CSI500"


# CSIA500
retry_process "${python_path} ${cur_path}/index_a500.py"
check_success "CSIA500"


${python_path} ${cur_path}/change_end_date.py
check_success "change_end_date"

