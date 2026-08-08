#pragma once

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

/* =============================================================================
 * WindowOrthoKVCompressor — 窗口作用域正交 KV 压缩器（重构自 TrueOrthoKDA）
 * -----------------------------------------------------------------------------
 * 设计约束（白皮书 §4）：
 *   1. 只接收 R-SWA 窗口内 K/V，绝不接收 Reference；
 *   2. 淘汰由 R-SWA 窗口推进触发（on_window_advance），不再有独立 32 计数器；
 *   3. 每次窗口滚动后，从「当前窗口原始 K/V」重建局部正交基（re-orthogonalize），
 *      抑制 ring 漂移（风险 1）；
 *   4. 保留 get/set_state，直接兼容 M7.5 的 kda_state_v1 跨节点传输。
 *
 * 压缩语义：窗口 K/V 被压缩为 <= ortho_base_dim 个正交基向量（秩-B 低秩近似）。
 * 设备侧原始窗口仍由 R-SWA 以 O(W) 恒定保存；本压缩器提供 O(1) 传输表征 +
 * 数值稳定的替代注意力路径。「窗口内 O(1) KV」指压缩基，而非原始窗口。
 * ========================================================================== */

typedef struct window_ortho_kv_compressor window_ortho_kv_compressor_t;

window_ortho_kv_compressor_t* wokdc_create(int num_heads, int head_dim,
                                           int ortho_base_dim, float decay_rate);
void                           wokdc_destroy(window_ortho_kv_compressor_t* c);

/* 设定原始窗口环容量 W（应等于 R-SWA 的 window_size，>= ortho_base_dim）。 */
void wokdc_set_window_capacity(window_ortho_kv_compressor_t* c, int window_size);

/* 喂入一个窗口 token 的 K/V（num_heads*head_dim 扁平）。
 * 内部：把该 token 追加进窗口原始环，若窗口满则淘汰最旧，随后重建局部正交基。 */
void wokdc_feed(window_ortho_kv_compressor_t* c, const float* k, const float* v);

/* 由 R-SWA 在窗口滚动时回调：通知某个窗口槽位被淘汰。
 * 触发从「保留窗口原始 K/V」重建正交基（re-orthogonalize）。 */
void wokdc_on_window_advance(window_ortho_kv_compressor_t* c, int evicted_slot);

/* 对 Q（num_heads*head_dim 扁平）做窗口正交基注意力，写入 out。 */
void wokdc_attention(const window_ortho_kv_compressor_t* c, const float* Q, float* out);

/* 当前基数量（<= ortho_base_dim） */
int wokdc_current_dim(const window_ortho_kv_compressor_t* c);

/* 状态序列化（兼容 kda_state_v1）：K/V/decay 均为 [ortho_base_dim][head_dim] / [ortho_base_dim] 扁平。 */
void wokdc_get_state(const window_ortho_kv_compressor_t* c,
                     float* K, float* V, float* decay, int* current_dim);
void wokdc_set_state(window_ortho_kv_compressor_t* c,
                     const float* K, const float* V, const float* decay, int current_dim);

#ifdef __cplusplus
}
#endif
