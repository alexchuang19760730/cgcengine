#pragma once

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

/* =============================================================================
 * R-SWA (Reference Sliding Window Attention) — C/C++ Manager
 * -----------------------------------------------------------------------------
 * 职责（与 TrueOrthoKDA 严格分工，见白皮书 §2 / §4）：
 *   - Reference 永久区：系统 Prompt / 关键事实，永不淘汰，走标准注意力；
 *   - 滑动窗口环形 KV：近期对话，容量固定 W，超出即淘汰最旧；
 *   - 产出「可见性掩码」：visible = Reference ∪ Window；
 *   - 窗口滚动时回调 WindowOrthoKVCompressor（窗口作用域正交 KV 压缩器），
 *     绝不允许 Reference 进入压缩基（§4.4 硬约束）。
 * ========================================================================== */

typedef enum {
    RSWA_LAYER_REFERENCE_STD = 0, /* Reference 区：标准注意力（不进压缩基）      */
    RSWA_LAYER_WINDOW_ORTHO  = 1, /* Window  区：启用局部正交压缩                */
    RSWA_LAYER_WINDOW_STD    = 2, /* Window  区：标准注意力（混合层奇数层兜底）  */
} rswa_layer_role_t;

typedef struct {
    int32_t num_heads;       /* 注意力头数                                        */
    int32_t head_dim;        /* 每头维度                                          */
    int32_t reference_len;   /* M：永久 Reference token 数                        */
    int32_t window_size;     /* W：滑动窗口容量（恒定 KV 上限）                   */
    int32_t ortho_base_dim;  /* 窗口压缩预算（<= window_size），正交基容量         */
    float   decay_rate;      /* 时间衰减率（与 TrueOrthoKDA 一致）                */
    int32_t hybrid_every;    /* 0=全正交；2=偶数层正交/奇数层标准（混合层策略）   */
    int32_t enable_window_ortho; /* 0/1：窗口层是否启用局部正交压缩（默认 1）     */
} rswa_config_t;

typedef struct rswa_manager rswa_manager_t;

/* 生命周期 */
rswa_manager_t* rswa_create(const rswa_config_t* cfg);
void            rswa_destroy(rswa_manager_t* m);

/* 注入 Reference 永久区 K/V（num_heads*head_dim 扁平，逐 token）。
 * 必须在 feed_token 之前调用，且只能调用 reference_len 次。 */
void rswa_set_reference(rswa_manager_t* m, const float* k, const float* v);

/* 喂入一个新窗口 token 的 K/V（num_heads*head_dim 扁平）。
 * 返回本次被淘汰的窗口槽位（-1 表示窗口未满，无淘汰）。
 * 发生淘汰时会自动回调窗口压缩器 on_window_advance。 */
int  rswa_feed_token(rswa_manager_t* m, const float* k, const float* v);

/* 可见性掩码：长度 seq_len，visible 位置 = 1，已淘汰位置 = 0。 */
void rswa_visibility_mask(const rswa_manager_t* m, int32_t seq_len, uint8_t* mask);

/* 当前可见 token 总数（reference + 窗口内） */
int32_t rswa_visible_count(const rswa_manager_t* m);

/* 给定层索引，按混合策略返回该层角色 */
rswa_layer_role_t rswa_layer_role(const rswa_manager_t* m, int32_t layer_idx);

/* 取窗口压缩器句柄（仅当 enable_window_ortho 时非空） */
void* rswa_window_compressor(rswa_manager_t* m);

/* ---- 注意力（供 backend adapter 调用）------------------------------------
 * Q / out 均为单 query（decode 步）扁平 num_heads*head_dim。
 * 合并注意力：对 (Reference 原始 K/V) ∪ (窗口：正交基 or 原始窗口) 做 union
 * softmax，再加权 V。这正是「R-SWA 管可见性、OrthoKDA 管窗口表征」的落地。 */
void rswa_reference_attention(const rswa_manager_t* m, const float* Q, float* out);
void rswa_window_attention(const rswa_manager_t* m, const float* Q, float* out);
void rswa_combined_attention(const rswa_manager_t* m, const float* Q, float* out);

#ifdef __cplusplus
}
#endif
