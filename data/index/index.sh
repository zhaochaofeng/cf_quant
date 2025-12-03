#!/bin/bash
source ~/.bashrc

## 构建指数成分股集合
cur_path=`pwd`
python_path="/root/anaconda3/envs/python3/bin/python"
qlib_path="/root/qlib"
provider_uri='/root/.qlib/qlib_data/index'
echo "cur_path: ${cur_path}"

if [ ! -d ${provider_uri} ]; then
  mkdir -p ${provider_uri}
fi

retry_process() {
  local cmd="$1"
  local retry_count=5
  local status=1
  while (( retry_count > 0 )); do
    eval "$cmd"
    status=$?
    (( status == 0 )) && return 0
    (( retry_count-- ))
    sleep 10m
  done
  return $status
}

# CSI300
retry_process "
  ${python_path} ${qlib_path}/scripts/data_collector/cn_index/collector.py \
  --index_name CSI300 \
  --qlib_dir ${provider_uri} \
  --method parse_instruments
"









