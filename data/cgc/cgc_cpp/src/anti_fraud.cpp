#include "anti_fraud.h"
#include "magi_backend_unified.h"
#include <sstream>
#include <iomanip>

// NVML支持（由 CMakeLists.txt 的 find_library(NVML_LIBRARY nvml) 控制）
#if defined(NVML_AVAILABLE) && NVML_AVAILABLE
#include <nvml.h>
#endif

#ifndef NVML_AVAILABLE
#define NVML_AVAILABLE 0
#endif

// 静态成员初始化
AntiFraudEngine* AntiFraudEngine::instance_ = nullptr;

// ============================ PerfRecord 方法 ============================

void PerfRecord::calculate_hash() {
    std::stringstream ss;
    ss << run_id << start_tsc << end_tsc 
       << std::fixed << std::setprecision(6) 
       << nvml_used_vram << nvml_power << nvml_utilization
       << total_latency << kv_read_bytes << kv_write_bytes
       << peak_memory << bandwidth << tok_per_sec
       << batch_length << kv_block_count << prefill_count
       << speedup;  // 引擎自动计算的加速比也纳入哈希
    
    std::string data = ss.str();
    
    // CRC32计算
    uint32_t crc = 0;
    for (char c : data) {
        crc ^= static_cast<uint8_t>(c);
        for (int i = 0; i < 8; ++i) {
            crc = (crc >> 1) ^ (crc & 1 ? 0xEDB88320 : 0);
        }
    }
    crc32_hash = ~crc;
}

// ============================ AntiFraudEngine 方法 ============================

AntiFraudEngine* AntiFraudEngine::get_instance() {
    if (!instance_) {
        instance_ = new AntiFraudEngine();
    }
    return instance_;
}

uint32_t AntiFraudEngine::crc32(const std::string& data) const {
    uint32_t crc = 0;
    for (char c : data) {
        crc ^= static_cast<uint8_t>(c);
        for (int i = 0; i < 8; ++i) {
            crc = (crc >> 1) ^ (crc & 1 ? 0xEDB88320 : 0);
        }
    }
    return ~crc;
}

uint64_t AntiFraudEngine::generate_unique_run_id() const {
    static std::random_device rd;
    static std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint64_t> dist(0, UINT64_MAX);
    return dist(gen);
}

uint64_t AntiFraudEngine::get_hardware_timestamp() const {
    // 使用rdtsc获取硬件时间戳
    uint64_t lo, hi;
    #if defined(__x86_64__)
    __asm__ volatile ("rdtsc" : "=a"(lo), "=d"(hi));
    return (hi << 32) | lo;
    #else
    return std::chrono::high_resolution_clock::now().time_since_epoch().count();
    #endif
}

bool AntiFraudEngine::nvml_collect(float& vram, float& power, float& utilization) {
    #if NVML_AVAILABLE
    nvmlDevice_t device;
    nvmlReturn_t result = nvmlInit();
    if (result != NVML_SUCCESS) {
        printf("[防造假] NVML初始化失败: %s\n", nvmlErrorString(result));
        return false;
    }
    
    result = nvmlDeviceGetHandleByIndex(0, &device);
    if (result != NVML_SUCCESS) {
        printf("[防造假] 获取GPU设备失败: %s\n", nvmlErrorString(result));
        nvmlShutdown();
        return false;
    }
    
    // 获取显存使用（MB）
    nvmlMemory_t memory;
    result = nvmlDeviceGetMemoryInfo(device, &memory);
    if (result == NVML_SUCCESS) {
        vram = static_cast<float>(memory.used) / 1024.0f / 1024.0f;
    }
    
    // 获取功耗（W）
    result = nvmlDeviceGetPowerUsage(device, &power);
    if (result == NVML_SUCCESS) {
        power /= 1000.0f;  // mW -> W
    }
    
    // 获取GPU利用率（%）
    nvmlUtilization_t util;
    result = nvmlDeviceGetUtilizationRates(device, &util);
    if (result == NVML_SUCCESS) {
        utilization = static_cast<float>(util.gpu);
    }
    
    nvmlShutdown();
    return true;
    #else
    // 非NVIDIA平台返回false，外部将使用引擎计算的估算值
    return false;
    #endif
}

void AntiFraudEngine::start_recording() {
    reset();
    current_record_.run_id = generate_unique_run_id();
    current_record_.start_tsc = get_hardware_timestamp();
    printf("[防造假] 开始记录，run_id=0x%llx\n", current_record_.run_id);
}

bool AntiFraudEngine::end_recording(double baseline_time) {
    current_record_.end_tsc = get_hardware_timestamp();
    
    // 计算加速比
    if (baseline_time > 0 && current_record_.total_latency > 0) {
        current_record_.speedup = baseline_time / current_record_.total_latency;
    }
    
    // 计算哈希
    current_record_.calculate_hash();
    
    // 校验数据
    current_record_.is_valid = validate_performance(baseline_time);
    
    return current_record_.is_valid;
}

void AntiFraudEngine::set_engine_data(double latency, uint64_t kv_read, uint64_t kv_write,
                                      size_t peak_mem, double bw) {
    current_record_.total_latency = latency;
    current_record_.kv_read_bytes = kv_read;
    current_record_.kv_write_bytes = kv_write;
    current_record_.peak_memory = peak_mem;
    current_record_.bandwidth = bw;
}

void AntiFraudEngine::set_hardware_data(float vram, float power, float utilization) {
    current_record_.nvml_used_vram = vram;
    current_record_.nvml_power = power;
    current_record_.nvml_utilization = utilization;
}

void AntiFraudEngine::set_backend_data(double tok_s, int batch_len, int kv_blocks, int prefill_cnt) {
    current_record_.tok_per_sec = tok_s;
    current_record_.batch_length = batch_len;
    current_record_.kv_block_count = kv_blocks;
    current_record_.prefill_count = prefill_cnt;
}

bool AntiFraudEngine::validate_outliers(const PerfRecord& record, double baseline_time) const {
    // 加速比异常（超过20x标记可疑）
    if (record.speedup > 20.0) {
        printf("[防造假] ❌ 异常：加速比%.2fx超出合理范围（≤20x）\n", record.speedup);
        return false;
    }
    
    // 耗时异常（负数或为0）
    if (record.total_latency <= 0) {
        printf("[防造假] ❌ 异常：耗时%.4fms无效（必须>0）\n", record.total_latency);
        return false;
    }
    
    // 显存异常（负数或波动超过合理范围）
    if (record.nvml_used_vram < 0 || record.nvml_used_vram > 48000) { // 双5090最大显存48GB
        printf("[防造假] ❌ 异常：显存%.2fMB超出合理范围\n", record.nvml_used_vram);
        return false;
    }
    
    // KV读写异常（负数）
    if (record.kv_read_bytes < 0 || record.kv_write_bytes < 0) {
        printf("[防造假] ❌ 异常：KV读写字节数无效\n");
        return false;
    }
    
    // KV读写带宽异常
    double kv_bandwidth = (record.kv_read_bytes + record.kv_write_bytes) / record.total_latency / 1024.0;
    if (kv_bandwidth < 100 || kv_bandwidth > 4000) {
        printf("[防造假] ❌ 异常：KV带宽%.2fMB/s超出合理范围\n", kv_bandwidth);
        return false;
    }
    
    // GPU利用率异常（为0但有输出）
    if (record.nvml_utilization >= 0 && record.nvml_utilization < 10 && record.tok_per_sec > 0) {
        printf("[防造假] ❌ 异常：GPU利用率%.1f%%过低但有输出\n", record.nvml_utilization);
        return false;
    }
    
    return true;
}

bool AntiFraudEngine::validate_consistency(const PerfRecord& record) const {
    // 交叉校验：引擎采集与NVML数据一致性（误差≤20%）
    if (record.nvml_used_vram > 0 && record.peak_memory > 0) {
        float engine_vram_mb = static_cast<float>(record.peak_memory) / 1024.0f / 1024.0f;
        float diff = std::fabs(engine_vram_mb - record.nvml_used_vram) / record.nvml_used_vram;
        if (diff > 0.2) {
            printf("[防造假] ❌ 异常：引擎显存(%.2fMB)与NVML显存(%.2fMB)校验失败（误差%.1f%%）\n", 
                   engine_vram_mb, record.nvml_used_vram, diff * 100);
            return false;
        }
    }
    
    // tok/s与GPU利用率一致性校验
    if (record.nvml_utilization >= 0 && record.tok_per_sec > 0) {
        // 简单校验：利用率低但tok/s高可能有问题
        if (record.nvml_utilization < 30 && record.tok_per_sec > 10000) {
            printf("[防造假] ❌ 异常：GPU利用率(%.1f%%)与tok/s(%.0f)不匹配\n", 
                   record.nvml_utilization, record.tok_per_sec);
            return false;
        }
    }
    
    return true;
}

bool AntiFraudEngine::validate_performance(double baseline_time) {
    printf("[防造假] 开始三端一致性校验...\n");
    
    // 1. 异常值校验
    if (!validate_outliers(current_record_, baseline_time)) {
        return false;
    }
    
    // 2. 三端一致性校验
    if (!validate_consistency(current_record_)) {
        return false;
    }
    
    printf("[防造假] ✅ 所有校验通过，数据有效\n");
    printf("[防造假] 结果摘要：run_id=0x%llx, 耗时=%.2fms, 加速比=%.2fx, tok/s=%.0f\n",
           current_record_.run_id, current_record_.total_latency, 
           current_record_.speedup, current_record_.tok_per_sec);
    
    return true;
}

void AntiFraudEngine::reset() {
    current_record_ = PerfRecord();
}

// ============================ 全局函数 ============================

void stat_performance_with_anti_fraud(const MagiGraphInfo& graph_info, double baseline_time) {
    AntiFraudEngine* engine = AntiFraudEngine::get_instance();
    
    // 开始记录
    engine->start_recording();
    
    // -------------------------- 引擎层自动计算 --------------------------
    size_t total_memory = 0;
    size_t peak_memory = 0;
    double total_latency = 0.0;
    uint64_t kv_read_bytes = 0;
    uint64_t kv_write_bytes = 0;
    
    // 遍历所有节点统计（引擎自动采集）
    for (const auto& node : graph_info.nodes) {
        total_latency += node.exec_time;
        for (const auto& in : node.inputs) {
            total_memory += in.size;
            kv_read_bytes += in.size;
        }
        for (const auto& out : node.outputs) {
            total_memory += out.size;
            kv_write_bytes += out.size;
        }
        peak_memory = std::max(peak_memory, total_memory);
    }
    
    // 引擎自动计算带宽（MB/s）
    double bandwidth = (total_latency > 0) ? 
        (kv_read_bytes + kv_write_bytes) / total_latency / 1024.0 : 0.0;
    
    // 设置引擎端数据（全部由引擎自动计算）
    engine->set_engine_data(total_latency, kv_read_bytes, kv_write_bytes, peak_memory, bandwidth);
    
    // 尝试NVML真实采集，失败则使用引擎估算值
    float vram_usage = static_cast<float>(peak_memory) / 1024.0f / 1024.0f;
    float power = 0.0f;
    float utilization = 0.0f;
    
    if (!engine->nvml_collect(vram_usage, power, utilization)) {
        // NVML不可用时，使用引擎自动计算的估算值
        // 功耗估算：基于峰值内存和带宽的粗略估算
        power = 50.0f + static_cast<float>(bandwidth) * 0.05f;  // 基础50W + 带宽加成
        power = std::min(power, 450.0f);  // 限制最大450W
        
        // 利用率估算：基于带宽利用率的估算
        utilization = std::min(100.0f, static_cast<float>(bandwidth) / 20.0f);  // 粗略估算
        
        printf("[防造假] NVML不可用，使用引擎估算值: vram=%.2fMB, power=%.1fW, util=%.1f%%\n",
               vram_usage, power, utilization);
    } else {
        printf("[防造假] NVML真实采集: vram=%.2fMB, power=%.1fW, util=%.1f%%\n",
               vram_usage, power, utilization);
    }
    
    // 设置硬件端数据
    engine->set_hardware_data(vram_usage, power, utilization);
    
    // 设置后端层数据
    engine->set_backend_data(graph_info.tokens_per_second, 
                            graph_info.batch_size,
                            graph_info.kv_block_count,
                            graph_info.prefill_count);
    
    // 结束记录并校验
    engine->end_recording(baseline_time);
    
    // 获取最终结果（Agent只读）
    const PerfRecord& record = engine->get_stat();
    if (record.is_valid) {
        printf("[MagiCompiler] 性能统计完成: 耗时=%.2fms, 加速比=%.2fx, tok/s=%.0f\n",
               record.total_latency, record.speedup, record.tok_per_sec);
    } else {
        printf("[MagiCompiler] ❌ 性能统计数据无效，已拒绝输出\n");
    }
}
