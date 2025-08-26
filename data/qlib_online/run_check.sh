#!/bin/bash
source ~/.bashrc
cur_path=`pwd`
python_path="/root/anaconda3/envs/python3/bin/python"

${python_path} ${cur_path}/check.py

if [ $? -eq 0 ];then
  echo "check is completed"
else
  echo "check is failed"
  exit 1
fi
