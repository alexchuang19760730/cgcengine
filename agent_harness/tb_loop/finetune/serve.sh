#!/usr/bin/env bash
set -euo pipefail
# ============================================================
# mlx_lm.server 启停（M4 上跑）：base（未微调）与 ft（带 LoRA 适配器）。
# 对比时两个服务都要在跑。
#
# 用法:
#   ./serve.sh start base|ft|all
#   ./serve.sh stop  base|ft|all
#   ./serve.sh status
# ============================================================
TB_LOOP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=config.env
source "$TB_LOOP_DIR/config.env"

ADAPTER_DIR="$TB_LOOP_DIR/finetune/adapters/gemma4-tb-lora"
PID_DIR="$TB_LOOP_DIR/finetune/.pids"
mkdir -p "$PID_DIR"

start_one() {
    local name="$1" port="$2" extra="$3" log="$4"
    if [[ -f "$PID_DIR/$name.pid" ]] && kill -0 "$(cat "$PID_DIR/$name.pid")" 2>/dev/null; then
        echo "[$name] 已在跑 (PID $(cat "$PID_DIR/$name.pid"), :$port)"
        return
    fi
    # shellcheck disable=SC2086
    nohup mlx_lm.server --model "$MLX_MODEL" --port "$port" $extra \
        > "$log" 2>&1 &
    echo $! > "$PID_DIR/$name.pid"
    echo "[$name] 启动中 (PID $!, :$port, 日志 $log)"
    echo "   等待就绪: curl -sf http://127.0.0.1:$port/v1/models"
}

case "${1:-}" in
    start)
        case "${2:-all}" in
            base) start_one base "$MLX_BASE_PORT" "" "$TB_LOOP_DIR/finetune/server_base.log" ;;
            ft)   start_one ft "$MLX_FT_PORT" "--adapter-path $ADAPTER_DIR" "$TB_LOOP_DIR/finetune/server_ft.log" ;;
            all)  start_one base "$MLX_BASE_PORT" "" "$TB_LOOP_DIR/finetune/server_base.log"
                  start_one ft "$MLX_FT_PORT" "--adapter-path $ADAPTER_DIR" "$TB_LOOP_DIR/finetune/server_ft.log" ;;
        esac
        ;;
    stop)
        for n in "${2:-all}"; do
            [[ "$n" == "all" ]] && n="base ft"
            for name in $n; do
                if [[ -f "$PID_DIR/$name.pid" ]]; then
                    kill "$(cat "$PID_DIR/$name.pid")" 2>/dev/null && echo "[$name] 已停止" || echo "[$name] 未在跑"
                    rm -f "$PID_DIR/$name.pid"
                fi
            done
        done
        ;;
    status)
        for name in base ft; do
            port=$([ "$name" = base ] && echo "$MLX_BASE_PORT" || echo "$MLX_FT_PORT")
            if [[ -f "$PID_DIR/$name.pid" ]] && kill -0 "$(cat "$PID_DIR/$name.pid")" 2>/dev/null; then
                echo "[$name] 运行中 (:${port})"
            else
                echo "[$name] 停止 (:${port})"
            fi
        done
        ;;
    *) echo "用法: $0 start|stop|status [base|ft|all]" >&2; exit 1 ;;
esac
