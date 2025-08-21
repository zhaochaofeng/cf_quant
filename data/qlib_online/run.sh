#!/bin/bash
source ~/.bashrc
cur_path=`pwd`
python_path="/root/anaconda3/envs/python3/bin/python"
qlib_path="/root/qlib"
provider_uri="~/.qlib/qlib_data/custom_data_hfq"

echo 'cur_path: '${cur_path}

if [ $# -eq 0 ]; then
    dt1=`date +%Y-%m-%d`
    dt2=${dt1}
  elif [ $# -eq 1 ]; then
    dt1=$1
    dt2=${dt1}
  elif [ $# -eq 2 ]; then
    dt1=$1
    dt2=$2
  else
    echo "参数错误"
    exit 1
fi

echo "时间区间为：["$dt1" - "$dt2"]"

${python_path} ${cur_path}/process.py main --start_date ${dt1} --end_date ${dt2}
if [ $? -eq 0 ];then
  echo "preprocess is completed"
else
  echo "preprocess is failed"
  exit 1
fi

# 更新数据
python ${qlib_path}/scripts/dump_bin.py dump_update \
--date_field_name date \
--csv_path ${provider_uri}/out_${dt1}_${dt2} \
--qlib_dir ${provider_uri} \
--include_fields open,close,high,low,volume,amount,factor


# 全量导入
#python scripts/dump_bin.py dump_all \
#--date_field_name date \
#--csv_path ~/.qlib/qlib_data/custom_data_hfq/out_2015-01-05_2025-08-18 \
#--qlib_dir ~/.qlib/qlib_data/custom_data_hfq \
#--include_fields open,close,high,low,volume,amount,factor


# 判断是否执行成功
if [ $? -eq 0 ]; then
  echo "dump_bin 执行成功！！！"
  exit 0
 else
  echo "dump_bin 执行失败！！！"
  exit 1
fi




