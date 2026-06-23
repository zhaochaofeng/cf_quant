#!/bin/bash
#
# helper.sh — 通用刷数工具函数
#
# 环境变量配置（默认可覆盖）：
#   TRADE_DAY_ONLY=true|false   — 是否启用交易日检查，默认 true
#   ON_ERROR="continue"|"stop"  — 回调失败时继续还是中止，默认 "continue"
#   PYTHON_PATH                  — Python 解释器（从 config.sh 或 fallback 到 python3）
#   PROJECT_PATH                 — 项目根目录（从 config.sh 或 fallback）
#
# 使用方式：被 run.sh 通过 source 加载

# 检查日期格式 YYYY-MM-DD
_validate_date() {
    local d=$1
    if [[ ! "$d" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
        echo "[helper] 无效日期格式: $d (expected YYYY-MM-DD)" >&2
        return 1
    fi
    return 0
}

# 日期递增：+1天，兼容 macOS 和 Linux
_date_add_one() {
    local d=$1
    if date -j -v+1d -f "%Y-%m-%d" "$d" "+%Y-%m-%d" 2>/dev/null; then
        return 0  # macOS 分支
    fi
    date -d "$d +1 day" "+%Y-%m-%d"  # Linux 分支
}

# 比较日期：d1 <= d2 返回 0，否则返回 1
_date_le() {
    local d1=$1 d2=$2
    # 转成 YYYYMMDD 数字比较（无分隔符）
    local n1="${d1//-/}" n2="${d2//-/}"
    [[ "$n1" -le "$n2" ]]
}

# 设置默认环境变量
_set_defaults() {
    : "${TRADE_DAY_ONLY:=true}"
    : "${ON_ERROR:=continue}"

    # 尝试从 config.sh 获取路径
    # 所有 调用此函数的脚本必须执行 source config.sh
    if [ -z "$PYTHON_PATH" ] && [ -f "${CF_QUANT_PATH}/config.sh" ]; then
        source "${CF_QUANT_PATH}/config.sh"
    fi
    : "${PROJECT_PATH:=${CF_QUANT_PATH:-.}}"
}

# ---- 主函数 ----
# iterate_days <start_date> <end_date> <callback_name> [callback_args...]
#
#   对 [start_date, end_date] 范围内每一天：
#     1. 如果 TRADE_DAY_ONLY=true，调用 is_trade_day.py 检查
#     2. 如果是交易日（或开关关闭），调用 callback_name <date> [callback_args...]
#     3. 如果回调返回非零且 ON_ERROR=stop，中止
#
#   返回值：累计失败天数（0 表示全部成功）
iterate_days() {
    if [ $# -lt 3 ]; then
        echo "用法: iterate_days <start_date> <end_date> <callback_name> [callback_args...]" >&2
        return 2
    fi

    local start_date=$1
    local end_date=$2
    local callback=$3        # 执行函数
    shift 3

    _set_defaults

    # 验证日期格式。如果验证失败则返回 2
    _validate_date "$start_date" || return 2
    _validate_date "$end_date" || return 2

    # 验证回调函数存在
    if ! declare -F "$callback" >/dev/null 2>&1; then
        echo "[helper] 回调函数 $callback 未定义" >&2
        return 2
    fi

    local cur_date="$start_date"
    local fail_count=0
    local ret

    while _date_le "$cur_date" "$end_date"; do
        echo "----------------------------------------"
        echo "[helper] 处理日期: $cur_date"

        # 交易日检查
        if [ "$TRADE_DAY_ONLY" = "true" ]; then
            ${PYTHON_PATH} "${PROJECT_PATH}/utils/is_trade_day.py" "$cur_date"
            ret=$?
            if [ "$ret" -eq 5 ]; then
                echo "[helper] 非交易日, 跳过: $cur_date"
                cur_date=$(_date_add_one "$cur_date")
                continue
            elif [ "$ret" -ne 0 ]; then
                echo "[helper] is_trade_day.py 执行异常, exit=$ret" >&2
            fi
        fi

        # 执行回调
        echo "[helper] 执行 $callback $cur_date $*"
        "$callback" "$cur_date" "$@"
        ret=$?

        if [ "$ret" -ne 0 ]; then
            echo "[helper] 回调执行失败 (exit=$ret): $cur_date" >&2
            fail_count=$((fail_count + 1))
            if [ "$ON_ERROR" = "stop" ]; then
                echo "[helper] ON_ERROR=stop, 中止" >&2
                return "$fail_count"
            fi
        fi

        cur_date=$(_date_add_one "$cur_date")
    done

    echo "----------------------------------------"
    echo "[helper] 完成! 总失败天数: $fail_count"
    return "$fail_count"
}
