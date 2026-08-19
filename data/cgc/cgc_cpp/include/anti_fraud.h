#ifndef ANTI_FRAUD_H
#define ANTI_FRAUD_H

#include <vector>
#include <string>
#include <cstdio>
#include <algorithm>
#include <cstdint>
#include <random>
#include <chrono>
#include <unordered_map>

// 前置声明
struct MagiGraphInfo;

// ============================ 防造假核心结构体 ============================

// 性能记录结构体（不可篡改，Agent仅可读）
struct PerfRecord {
    uint64_t run_id;          // 唯一运行ID
    uint64_t start_tsc;       // 开始时间戳（硬件级）
    uint64_t end_tsc;         // 结束时间戳（硬件级）
    
    // 硬件端数据（NVML读取）
    float nvml_used_vram;     // NVML读取的真实显存（MB）
    float nvml_power;         // NVML读取的GPU功耗（W）
    float nvml_utilization;   // NVML读取的GPU利用率（%）
    
    // 引擎端数据（CGC统计）
    double total_latency;     // 真实总耗时（ms）
    uint64_t kv_read_bytes;   // KV缓存真实读取字节数
    uint64_t kv_write_bytes;  // KV缓存真实写入字节数
    size_t peak_memory;       // 峰值内存（字节）
    double bandwidth;         // IO带宽（MB/s）
    
    // 后端层数据
    double tok_per_sec;       // 实际tok/s
    int batch_length;         // batch长度
    int kv_block_count;       // KV block数量
    int prefill_count;        // prefill次数
    
    // 自动计算字段
    double speedup;           // 引擎自动计算的加速比
    
    // 校验字段
    uint32_t crc32_hash;      // 数据校验哈希（不可篡改）
    bool is_valid;            // 数据是否有效
    
    PerfRecord() : 
        run_id(0), start_tsc(0), end_tsc(0),
        nvml_used_vram(-1), nvml_power(-1), nvml_utilization(-1),
        total_latency(0), kv_read_bytes(0), kv_write_bytes(0),
        peak_memory(0), bandwidth(0),
        tok_per_sec(0), batch_length(0), kv_block_count(0), prefill_count(0),
        speedup(0), crc32_hash(0), is_valid(false) {}
    
    // 计算哈希（防止数据被篡改）
    void calculate_hash();
};

// ============================ 防造假核心类 ============================

class AntiFraudEngine {
private:
    static AntiFraudEngine* instance_;
    PerfRecord current_record_;
    
    // 私有构造函数
    AntiFraudEngine() = default;
    
    // CRC32计算
    uint32_t crc32(const std::string& data) const;
    
    // 生成唯一run_id
    uint64_t generate_unique_run_id() const;
    
    // 获取硬件时间戳
    uint64_t get_hardware_timestamp() const;
    
    // 异常值校验
    bool validate_outliers(const PerfRecord& record, double baseline_time) const;
    
    // 三端一致性校验
    bool validate_consistency(const PerfRecord& record) const;
    
public:
    // NVML真实采集（替代模拟值）
    bool nvml_collect(float& vram, float& power, float& utilization);

    // 单例模式
    static AntiFraudEngine* get_instance();
    
    // 禁止拷贝
    AntiFraudEngine(const AntiFraudEngine&) = delete;
    AntiFraudEngine& operator=(const AntiFraudEngine&) = delete;
    
    // 开始性能记录
    void start_recording();
    
    // 结束性能记录并进行校验
    bool end_recording(double baseline_time);
    
    // 设置引擎端数据
    void set_engine_data(double latency, uint64_t kv_read, uint64_t kv_write, 
                        size_t peak_mem, double bw);
    
    // 设置硬件端数据（NVML）
    void set_hardware_data(float vram, float power, float utilization);
    
    // 设置后端层数据
    void set_backend_data(double tok_s, int batch_len, int kv_blocks, int prefill_cnt);
    
    // Agent只读接口（禁止修改）
    const PerfRecord& get_stat() const { return current_record_; }
    
    // 校验性能数据
    bool validate_performance(double baseline_time);
    
    // 重置记录
    void reset();
};

// ============================ 全局函数 ============================

// 嵌入防造假机制的stat_performance函数
void stat_performance_with_anti_fraud(const MagiGraphInfo& graph_info, double baseline_time);

#endif // ANTI_FRAUD_H
