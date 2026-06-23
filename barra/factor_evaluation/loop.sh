#!/bin/bash
# loop.sh — 因子评价逐日刷数脚本
#
# 用法:
#   ./loop.sh                              # 当天
#   ./loop.sh 2025-01-01                   # 单日
#   ./loop.sh 2025-01-01 2025-12-31        # 日期范围
#
# 环境变量:
#   TRADE_DAY_ONLY=false   # 跳过交易日检查（默认检查）
#   ON_ERROR=stop          # 失败时中止（默认继续）

source ~/.bashrc

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../config.sh"
source "${SCRIPT_DIR}/../../utils/helper.sh"

HISTORY_MONTHS=12

run_factor_eval() {
    local dt=$1
    echo "[loop] ${dt} — 开始..."

    local output_dir="${SCRIPT_DIR}/data"
    if [ ! -d "$output_dir" ]
    then
        mkdir -p "$output_dir"
    fi

    ${PYTHON_PATH} "${SCRIPT_DIR}/run.py" \
        --now-date "$dt" \
        --history-months "$HISTORY_MONTHS" \
        --output "$output_dir"

    if [ $? -eq 0 ]; then
        echo "[loop] ${dt} — 成功"
    else
        echo "[loop] ${dt} — 失败" >&2
        return 1
    fi
}

case $# in
    0)
        dt=$(date +%Y-%m-%d)
        echo "[loop] 单日模式: ${dt}"
        run_factor_eval "$dt"
        ;;
    1)
        dt=$1
        echo "[loop] 单日模式: ${dt}"
        run_factor_eval "$dt"
        ;;
    2)
        start_date=$1
        end_date=$2
        echo "[loop] 范围模式: ${start_date} → ${end_date}"
        echo "[loop] TRADE_DAY_ONLY=${TRADE_DAY_ONLY:-true}, ON_ERROR=${ON_ERROR:-continue}"
        iterate_days "$start_date" "$end_date" "run_factor_eval"
        ;;
    *)
        echo "用法: $0 [start_date] [end_date]" >&2
        exit 2
        ;;
esac
