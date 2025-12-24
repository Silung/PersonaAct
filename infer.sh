#!/bin/bash

# 解析命令行参数
NAME=""
MODEL_PATH=""
DATA_DIR="data"
PERSONA=false
EXPORT=false

# 显示使用说明
usage() {
    echo "用法: $0 --name NAME --model_path MODEL_PATH [--data_dir DATA_DIR] [--persona] [--export]"
    echo ""
    echo "参数:"
    echo "  --name NAME          用户名称 (yqg, zsl, 等)"
    echo "  --model_path PATH    模型路径"
    echo "  --data_dir DIR       数据目录 (默认: data)"
    echo "  --persona            使用persona模式"
    echo "  --export              先执行 swift export 合并 LoRA，然后使用合并后的模型"
    echo ""
    exit 1
}

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --name)
            NAME="$2"
            shift 2
            ;;
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --persona)
            PERSONA=true
            shift
            ;;
        --export)
            EXPORT=true
            shift
            ;;
        *)
            echo "未知参数: $1"
            usage
            ;;
    esac
done

# 检查必需参数
if [[ -z "$NAME" ]] || [[ -z "$MODEL_PATH" ]]; then
    echo "错误: 必须指定 --name 和 --model_path"
    usage
fi

# 如果指定了 --export，先执行 export
if [[ "$EXPORT" == true ]]; then
    echo "=========================================="
    echo "执行 swift export 合并 LoRA"
    echo "=========================================="
    echo "适配器路径: $MODEL_PATH"
    
    # 执行 swift export 命令
    swift export \
        --adapters "$MODEL_PATH" \
        --merge_lora true
    
    EXPORT_EXIT_CODE=$?
    if [[ $EXPORT_EXIT_CODE -ne 0 ]]; then
        echo "错误: swift export 失败，退出码: $EXPORT_EXIT_CODE"
        exit $EXPORT_EXIT_CODE
    fi
    
    # 更新模型路径为合并后的路径
    MODEL_PATH="${MODEL_PATH}-merged"
    echo "合并后的模型路径: $MODEL_PATH"
    echo ""
fi

# 检查模型路径是否存在
if [[ ! -d "$MODEL_PATH" ]]; then
    echo "错误: 模型路径不存在: $MODEL_PATH"
    exit 1
fi

# 查找未占用的端口（从8012开始）
find_free_port() {
    local start_port=8012
    local port=$start_port
    while lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; do
        port=$((port + 1))
        if [[ $port -gt 65535 ]]; then
            echo "错误: 无法找到可用端口"
            exit 1
        fi
    done
    echo $port
}

# 自动分配可用端口
PORT=$(find_free_port)
API_BASE="http://127.0.0.1:${PORT}/v1"

echo "=========================================="
echo "推理脚本配置"
echo "=========================================="
echo "用户名称: $NAME"
echo "模型路径: $MODEL_PATH"
echo "数据目录: $DATA_DIR"
echo "Persona模式: $PERSONA"
echo "Export模式: $EXPORT"
echo "端口: $PORT"
echo "API Base: $API_BASE"
echo "=========================================="

# 启动 vllm 服务（后台运行）
echo ""
echo "正在启动 vLLM 服务..."
VLLM_PID=""
VLLM_LOG="vllm_server_${PORT}.log"

# 启动 vllm 服务
CUDA_VISIBLE_DEVICES=0 vllm serve "$MODEL_PATH" \
    --port $PORT \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --served-model-name qwen \
    --allowed-local-media-path / \
    --data-parallel-size 1 \
    --mm_processor_cache_gb 0 > "$VLLM_LOG" 2>&1 &

VLLM_PID=$!
echo "vLLM 服务已启动，PID: $VLLM_PID"
echo "日志文件: $VLLM_LOG"

# 等待服务启动（最多等待300秒，因为模型加载可能需要较长时间）
echo "等待 vLLM 服务启动（每2秒检查一次 /health 端点）..."
MAX_WAIT=300
WAITED=0
SERVICE_READY=false

# 检查进程是否还在运行
check_process_alive() {
    if kill -0 $VLLM_PID 2>/dev/null; then
        return 0
    fi
    return 1
}

while [[ $WAITED -lt $MAX_WAIT ]]; do
    # 首先检查进程是否还在运行
    if ! check_process_alive; then
        echo "错误: vLLM 进程已退出，请检查日志: $VLLM_LOG"
        echo "最后20行日志："
        tail -20 "$VLLM_LOG" 2>/dev/null || true
        exit 1
    fi
    
    # 使用 /health 端点检查服务是否就绪
    if curl -s -f "http://127.0.0.1:${PORT}/health" > /dev/null 2>&1; then
        echo "vLLM 服务已就绪！"
        SERVICE_READY=true
        break
    fi
    
    sleep 2
    WAITED=$((WAITED + 2))
    echo "等待中... (${WAITED}/${MAX_WAIT}秒)"
done

if [[ "$SERVICE_READY" != "true" ]]; then
    echo "错误: vLLM 服务启动超时（${MAX_WAIT}秒）！"
    echo "请检查日志文件: $VLLM_LOG"
    echo "最后20行日志："
    tail -20 "$VLLM_LOG" 2>/dev/null || true
    # 清理进程
    kill $VLLM_PID 2>/dev/null || true
    exit 1
fi

# 运行推理
echo ""
echo "=========================================="
echo "开始推理"
echo "=========================================="

# 构建 infer.py 命令
INFER_CMD="python infer.py --name $NAME --data_dir $DATA_DIR --api_base $API_BASE"
if [[ "$PERSONA" == true ]]; then
    INFER_CMD="$INFER_CMD --persona"
fi

echo "执行命令: $INFER_CMD"
echo ""

# 执行推理
$INFER_CMD
INFER_EXIT_CODE=$?

# 停止 vllm 服务
echo ""
echo "=========================================="
echo "正在停止 vLLM 服务..."
echo "=========================================="

if [[ -n "$VLLM_PID" ]]; then
    # 尝试优雅关闭
    kill $VLLM_PID 2>/dev/null
    sleep 5
    
    # 如果还在运行，强制关闭
    if kill -0 $VLLM_PID 2>/dev/null; then
        echo "强制关闭 vLLM 服务..."
        kill -9 $VLLM_PID 2>/dev/null
    fi
    
    # 确保端口被释放（查找并关闭占用该端口的进程）
    LSOF_PID=$(lsof -ti:$PORT 2>/dev/null)
    if [[ -n "$LSOF_PID" ]]; then
        echo "关闭占用端口 $PORT 的进程: $LSOF_PID"
        kill -9 $LSOF_PID 2>/dev/null
    fi
    
    echo "vLLM 服务已停止"
else
    echo "未找到 vLLM 进程"
fi

# 退出时使用推理的退出码
if [[ $INFER_EXIT_CODE -ne 0 ]]; then
    echo ""
    echo "推理过程出现错误，退出码: $INFER_EXIT_CODE"
    exit $INFER_EXIT_CODE
fi

echo ""
echo "=========================================="
echo "推理完成！"
echo "=========================================="

