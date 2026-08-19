#ifndef CGC_ROUTING_CONTROLLER_H
#define CGC_ROUTING_CONTROLLER_H

// cgc_routing_controller.h — C 版路由驱动专家预取控制器
//
// 设计参考 (CGC_Qwen36_優化紀錄_心智圖):
//  1. miss-only        : 只预取未命中缓存的专家 (MTP verify union 成本 > 收益,
//                        过度预取会抢 SSD 带宽 => 严格 miss-only + 上限)
//  2. lookahead        : 基于路由历史预测未来 token 的专家 (MoE-SpAc,
//                        MTP 前瞻预测每 32 步 refresh)
//  3. 异步预取线程      : 低优先级批量合并读取, 避免 async 背景读抢 decode miss
//                        的 SSD 带宽 (PRELOAD=sync 必须 教训 => 批量+限速+暂停)
//  4. hot-pool         : top-N mix profile 固定槽 (已由 streamer 实现, 控制器
//                        感知热池专家, 不再重复预取)

#include "cgc_expert_streamer.h"

#include <stdint.h>
#include <stdbool.h>

#ifdef _WIN32
#define _WIN32_WINNT 0x0A00
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define CGC_ROUTING_MAX_LOOKAHEAD 64
#define CGC_ROUTING_MAX_QUEUE     256
#define CGC_ROUTING_HISTORY_MAX   2048

typedef enum {
    CGC_ROUTING_STRATEGY_NONE = 0,       // 关闭预取 (只记录)
    CGC_ROUTING_STRATEGY_CURRENT_ONLY,   // 只预取当前 token 的 miss
    CGC_ROUTING_STRATEGY_LOOKAHEAD,      // 当前 miss + 历史前瞻预测
} cgc_routing_strategy_t;

// 路由预取配置
typedef struct {
    cgc_routing_strategy_t strategy;
    int lookahead_tokens;            // 前瞻预测窗口 (MoE-SpAc: 建议 4~32)
    int max_prefetch_per_step;       // 单步最多预取专家数 (建议 8~16, 防抢带宽)
    int async_prefetch;              // 1 = 异步线程, 0 = 同步
    int enable_history_predict;      // 启用基于历史的 N-gram 前瞻预测
    int history_predict_order;       // N-gram 阶数 (建议 2~3)
    uint64_t min_refresh_interval_nanos; // 前瞻预测刷新间隔 (MoE-SpAc 32 步)
} cgc_routing_config_t;

// 前瞻预测结果 (每层最多一个, 通过 streamer_pool 获取)
typedef struct {
    int layer;
    int expert_ids[CGC_MAX_EXPERTS_PER_LAYER];
    int count;
} cgc_routing_lookahead_t;

// 路由控制器统计
typedef struct {
    uint64_t total_steps;            // 累计 decode 步数
    uint64_t total_route_ids;        // 累计路由 id 数
    uint64_t total_prefetch_attempts;
    uint64_t total_prefetch_loaded;  // 实际读入的专家数
    uint64_t total_prefetch_skipped; // 跳过 (已缓存 / 热池 / 超限)
    uint64_t total_predicted_hits;   // 前瞻预测命中的次数
    uint64_t total_read_wall_nanos;
    uint64_t last_refresh_step;
} cgc_routing_stats_t;

typedef struct {
    cgc_streamer_pool_t* pool;
    cgc_routing_config_t config;
    cgc_routing_stats_t stats;

    // N-gram 历史 (layer, expert) 序列
    int history_layers[CGC_ROUTING_HISTORY_MAX];
    int history_experts[CGC_ROUTING_HISTORY_MAX];
    int history_count;

    // 每层最近路由 (用于 merge 去重)
    int layer_route_layer[CGC_MAX_LAYERS];
    int layer_route_experts[CGC_MAX_LAYERS][CGC_MAX_EXPERTS_PER_LAYER];
    int layer_route_count[CGC_MAX_LAYERS];

    int initialized;
    char error_msg[256];
} cgc_routing_controller_t;

cgc_routing_controller_t* cgc_routing_controller_create(cgc_streamer_pool_t* pool,
                                                         const cgc_routing_config_t* config);

void cgc_routing_controller_destroy(cgc_routing_controller_t* ctrl);

// 记录当前 decode 步的路由 id, 并触发预取 (核心入口)
//   layer_route_ids:  该层本次路由的 expert id 数组
//   count:            数量
// 返回本步实际预取 (或提交异步) 的专家数
int cgc_routing_controller_on_route(cgc_routing_controller_t* ctrl,
                                    int layer,
                                    const int* expert_ids,
                                    int count);

// 批量记录多层路由 (简化调用: 传入每层的 routes)
int cgc_routing_controller_on_route_batch(cgc_routing_controller_t* ctrl,
                                          const int* layers,
                                          const int* const* expert_ids_per_layer,
                                          const int* counts,
                                          int layer_count);

// 手动触发预取: 显式给出期望预取的专家 (用于 MTP draft 前瞻)
int cgc_routing_controller_prefetch_experts(cgc_routing_controller_t* ctrl,
                                            int layer,
                                            const int* expert_ids,
                                            int count);

// 刷新前瞻预测缓存 (MoE-SpAc: 每 N 步)
int cgc_routing_controller_refresh_lookahead(cgc_routing_controller_t* ctrl);

// 等待异步预取线程完成当前批次 (在 decode 阶段间调用, 保证数据就绪)
void cgc_routing_controller_drain(cgc_routing_controller_t* ctrl);

cgc_routing_stats_t cgc_routing_controller_stats(const cgc_routing_controller_t* ctrl);

void cgc_routing_controller_reset_stats(cgc_routing_controller_t* ctrl);

#ifdef __cplusplus
}
#endif

#endif // CGC_ROUTING_CONTROLLER_H
