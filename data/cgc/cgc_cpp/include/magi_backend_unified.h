#ifndef MAGI_BACKEND_UNIFIED_H
#define MAGI_BACKEND_UNIFIED_H

#include <string>
#include <vector>
#include <unordered_map>
#include <iostream>
#include <memory>

// -------------------------- 后端类型定义 --------------------------
enum class MagiBackendType {
    LLAMA_CPP = 0,
    VLLM = 1,
    MEGATRAIN_2026_4 = 2,
    MLX_TUNE = 3,
    UNKNOWN = 99
};

// -------------------------- 执行模式定义 --------------------------
enum class MagiExecuteMode {
    INFER_PREFILL = 0,
    INFER_DECODE = 1,
    TRAIN_FWD = 2,
    TRAIN_BWD = 3,
    LAYER_EXEC = 4,
    TRAIN_GLOBAL = 5,
    TUNE_LORA = 6
};

// -------------------------- IO 路径定义 --------------------------
enum class MagiIOPath {
    NONE = 0,
    CPU_MEM = 1,
    GPU_MEM = 2,
    H2D = 3,
    D2H = 4,
    P2P = 5,
    MLX_UNIFIED_MEMORY = 6,
    SPDK_IO = 7,
    GDS_PD = 8
};

// -------------------------- 通用数据结构（统一所有后端计算图）--------------------------

struct MagiTensorInfo {
    void* data_ptr;              // 张量数据地址
    std::vector<int64_t> shape;  // 张量形状
    std::string dtype;           // 数据类型（float32/float16/int8/int4等）
    std::string device;          // 设备（cpu/cuda/mlx）
    int device_id;               // 设备 ID
    size_t size;                 // 张量字节数
    bool is_on_gpu;              // 是否在 GPU 上
    bool is_pinned;              // 是否钉住内存
    bool needs_copy;             // 是否需要拷贝
    MagiIOPath io_path;          // IO 路径
    std::string stream_id;       // 调度流 ID
    size_t offset;               // 偏移量
    size_t peak_memory;          // 峰值内存

    MagiTensorInfo() :
        data_ptr(nullptr),
        device_id(0),
        size(0),
        is_on_gpu(false),
        is_pinned(false),
        needs_copy(false),
        io_path(MagiIOPath::NONE),
        offset(0),
        peak_memory(0) {}
};

struct MagiGraphNode {
    std::string op_type;                       // 算子类型
    int node_id;                               // 节点 ID
    std::vector<MagiTensorInfo> inputs;        // 输入张量
    std::vector<MagiTensorInfo> outputs;       // 输出张量
    double exec_time;                          // 执行耗时（ms）
    std::unordered_map<std::string, std::string> attrs;  // 节点属性

    MagiGraphNode() : node_id(0), exec_time(0.0) {}
};

struct MagiHardwareInfo {
    std::string device_type;  // 设备类型（cuda/cpu/mlx）
    int device_id;            // 设备 ID
    size_t total_memory;      // 总内存
    size_t free_memory;      // 空闲内存
    int compute_capability_major;  // CUDA 计算能力主版本
    int compute_capability_minor; // CUDA 计算能力次版本
    size_t l2_cache_size;    // L2 缓存大小
    int num_sm;              // SM 数量
    size_t memory_bandwidth;  // 内存带宽 (GB/s)
    bool is_unified_memory;  // 是否为统一内存（MLX）

    MagiHardwareInfo() :
        device_id(0),
        total_memory(0),
        free_memory(0),
        compute_capability_major(0),
        compute_capability_minor(0),
        l2_cache_size(0),
        num_sm(0),
        memory_bandwidth(0),
        is_unified_memory(false) {}
};

struct MagiGraphInfo {
    MagiBackendType backend;                // 所属后端
    MagiExecuteMode mode;                   // 执行模式
    std::vector<MagiGraphNode> nodes;       // 所有节点
    int layer_id;                          // 层 ID（多层模型时）
    std::string stream_id;                  // 数据流 ID
    std::string optimization;               // 优化策略名称
    MagiHardwareInfo hw;                    // 硬件信息
    
    // 防造假机制需要的字段
    double tokens_per_second;               // 实际tok/s
    int batch_size;                         // batch长度
    int kv_block_count;                     // KV block数量
    int prefill_count;                      // prefill次数

    MagiGraphInfo() :
        backend(MagiBackendType::LLAMA_CPP),
        mode(MagiExecuteMode::INFER_DECODE),
        layer_id(-1),
        optimization("none"),
        tokens_per_second(0.0),
        batch_size(0),
        kv_block_count(0),
        prefill_count(0) {}
};

struct MagiPerfCompare {
    double native_latency;
    size_t native_memory;
    double optimized_latency;
    size_t optimized_memory;

    MagiPerfCompare() :
        native_latency(0.0),
        native_memory(0),
        optimized_latency(0.0),
        optimized_memory(0) {}
};

// -------------------------- MagiCompilerBackend 类定义 --------------------------

class MagiCompilerBackend {
public:
    static MagiCompilerBackend& get_instance();

    void set_log_level(int level);
    int get_log_level() const;

    // -------------------------- 核心分析函数 --------------------------
    void analyze_graph(const MagiGraphInfo& graph_info);
    void identify_optimization(const MagiGraphInfo& graph_info);
    void stat_performance(const MagiGraphInfo& graph_info);

    // -------------------------- 自动硬件检测 --------------------------
    MagiHardwareInfo detect_hardware();

    // -------------------------- 自动生成最优代码 --------------------------
    std::string generate_optimal_code(const MagiGraphInfo& graph_info);

    // -------------------------- 自动下发到后端运行 --------------------------
    MagiPerfCompare dispatch_to_backend(MagiBackendType backend_type, const std::string& code);

    // -------------------------- 自动运行原生后端 --------------------------
    MagiPerfCompare run_native_backend(MagiBackendType backend_type, const MagiGraphInfo& graph_info);

    // -------------------------- 自动性能对比 --------------------------
    void compare_performance(const MagiPerfCompare& native, const MagiPerfCompare& optimized, MagiBackendType backend_type);

    // -------------------------- 各后端计算图解析 --------------------------
    std::vector<MagiGraphNode> parse_ggml_graph(
        void* ggml_backend,
        const std::vector<MagiTensorInfo>& inputs,
        const std::vector<MagiTensorInfo>& outputs
    );

    std::vector<MagiGraphNode> parse_vllm_graph(
        const std::vector<MagiTensorInfo>& inputs,
        const std::vector<MagiTensorInfo>& outputs
    );

    std::vector<MagiGraphNode> parse_megatrain_graph(
        const std::vector<MagiTensorInfo>& inputs,
        const std::vector<MagiTensorInfo>& outputs
    );

    std::vector<MagiGraphNode> parse_mlx_graph(
        const std::vector<MagiTensorInfo>& inputs,
        const std::vector<MagiTensorInfo>& outputs
    );

    // -------------------------- MegaTrain 细粒度子图捕获 --------------------------

    void capture_megatrain_subgraph(
        const std::vector<MagiGraphNode>& subgraph_nodes,
        int layer_id = -1,
        const std::string& stream_id = ""
    );

    std::vector<MagiGraphNode> parse_megatrain_layer_graph(
        const std::vector<MagiTensorInfo>& inputs,
        const std::vector<MagiTensorInfo>& outputs,
        const std::string& stream_id
    );

    // -------------------------- 各后端专属劫持接口（适配原有架构）--------------------------

    void capture_llama_compute(
        void* ggml_backend,
        const std::vector<MagiTensorInfo>& inputs,
        const std::vector<MagiTensorInfo>& outputs
    );

    void capture_vllm_forward(
        const std::vector<MagiTensorInfo>& inputs,
        const std::vector<MagiTensorInfo>& outputs,
        MagiExecuteMode mode = MagiExecuteMode::INFER_PREFILL
    );

    void capture_megatrain_exec(
        int layer_id,
        const std::vector<MagiGraphNode>& layer_nodes,
        const std::string& stream_id,
        MagiExecuteMode mode = MagiExecuteMode::LAYER_EXEC
    );

    void capture_mlx_compute(
        const std::vector<MagiTensorInfo>& inputs,
        const std::vector<MagiTensorInfo>& outputs,
        MagiExecuteMode mode = MagiExecuteMode::TUNE_LORA
    );

    // -------------------------- 辅助方法 --------------------------
    std::string backend_name(MagiBackendType bt);
    std::string mode_name(MagiExecuteMode m);

private:
    MagiCompilerBackend();
    ~MagiCompilerBackend();

    MagiCompilerBackend(const MagiCompilerBackend&) = delete;
    MagiCompilerBackend& operator=(const MagiCompilerBackend&) = delete;

    int log_level_;
    double baseline_time_;

    // -------------------------- 内部分析函数 --------------------------
    void analyze_graph_topology(const MagiGraphInfo& g);
    void analyze_tensor_dependencies(const MagiGraphInfo& g);
    void analyze_memory_lifetime(const MagiGraphInfo& g);
    void analyze_device_io(const MagiGraphInfo& g);

    void detect_attention_pattern(const MagiGraphInfo& g);
    void detect_mlp_pattern(const MagiGraphInfo& g);
    void detect_fusable_ops(const MagiGraphInfo& g);
    void select_best_optimizer(const MagiGraphInfo& g);

    void measure_latency(const MagiGraphInfo& g);
    void measure_memory_usage(const MagiGraphInfo& g);
    void measure_flops(const MagiGraphInfo& g);
    void measure_io_bandwidth(const MagiGraphInfo& g);

    void capture_graph(const MagiGraphInfo& graph_info);
};

// -------------------------- 简化调用宏 --------------------------
#define MAGI_BACKEND MagiCompilerBackend::get_instance()
#define MAGI_CAPTURE_LLAMA(backend, inputs, outputs) MAGI_BACKEND.capture_llama_compute(backend, inputs, outputs)
#define MAGI_CAPTURE_VLLM(inputs, outputs, mode) MAGI_BACKEND.capture_vllm_forward(inputs, outputs, mode)
#define MAGI_CAPTURE_MEGATRAIN(layer_id, nodes, stream_id, mode) MAGI_BACKEND.capture_megatrain_exec(layer_id, nodes, stream_id, mode)
#define MAGI_CAPTURE_MLX(inputs, outputs, mode) MAGI_BACKEND.capture_mlx_compute(inputs, outputs, mode)

#endif // MAGI_BACKEND_UNIFIED_H