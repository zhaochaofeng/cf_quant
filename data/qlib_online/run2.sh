#!/bin/bash

# 功能：每日权量更新
source ~/.bashrc
cur_path=`pwd`
python_path="/root/anaconda3/envs/python3/bin/python"
qlib_path="/root/qlib"
provider_uri="~/.qlib/qlib_data/custom_data_hfq"
provider_uri_tmp="~/.qlib/qlib_data/custom_data_hfq_tmp"
echo 'cur_path: '${cur_path}

dt1='2015-01-05'
dt2=`date +%Y-%m-%d`
echo "时间区间为：["$dt1" - "$dt2"]"

if [ -d "${provider_uri_tmp}" ]; then
  rm -rf "${provider_uri_tmp}"
fi
mkdir -p ${provider_uri_tmp}

check_success() {
  if [ $? -eq 0 ]; then
    echo "$1 执行成功！！！"
  else
    echo "$1 执行失败！！！"
    exit 1
  fi
}

echo "1、从mysql中导出数据..."
mysql -uchaofeng -pZhao_123 -e "select ts_code, day as date, open, close, high, low, vol, amount, adj_factor \
from cf_quant.trade_daily2 \
where day>='${dt1}' and day<='${dt2}';" > ${provider_uri_tmp}/custom_${dt1}_${dt2}.csv

check_success "从mysql中导出数据"

echo "2、处理数据..."
${python_path} ${cur_path}/process.py main \
--provider_uri ${provider_uri_tmp} \
--start_date ${dt1} \
--end_date ${dt2} \
--is_offline True \
--path_in ${provider_uri_tmp}/custom_${dt1}_${dt2}.csv

check_success "处理数据"

echo "3、转化为qlib格式..."
${python_path} ${qlib_path}/scripts/dump_bin.py dump_all \
--date_field_name date \
--csv_path ${provider_uri_tmp}/out_${dt1}_${dt2} \
--qlib_dir ${provider_uri_tmp} \
--include_fields open,close,high,low,volume,amount,factor

check_success "转化为qlib格式"

#rm -rf ${provider_uri}
#mv ${provider_uri_tmp} ${provider_uri}
