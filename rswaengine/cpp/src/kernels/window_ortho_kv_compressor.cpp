#include "kernels/window_ortho_kv_compressor.h"

#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>

namespace {

/* CPU Gram-Schmidt：对 v 减去其在前 n 个基向量（扁平 [n][head_dim]）上的投影并归一化。 */
inline void gram_schmidt_cpu(float* v, const float* basis_flat, int n, int head_dim) {
    for (int i = 0; i < n; ++i) {
        const float* bi = basis_flat + i * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) dot += v[d] * bi[d];
        for (int d = 0; d < head_dim; ++d) v[d] -= dot * bi[d];
    }
    float norm = 1e-8f;
    for (int d = 0; d < head_dim; ++d) norm += v[d] * v[d];
    norm = 1.0f / std::sqrt(norm);
    for (int d = 0; d < head_dim; ++d) v[d] *= norm;
}

} // namespace

struct window_ortho_kv_compressor {
    int num_heads;
    int head_dim;
    int ortho_base_dim;   /* 基容量 B（压缩预算） */
    int window_capacity;  /* 原始窗口环容量 W（= R-SWA 的 window_size） */
    float decay_rate;

    /* 原始窗口环（O(W) 恒定）—— 由 R-SWA 以恒定上限持有，本压缩器复用之重建基 */
    std::vector<float> win_k;
    std::vector<float> win_v;
    int win_head;   /* 最旧 token 的物理槽位 */
    int win_count;  /* 当前窗口内 token 数 */

    /* 局部正交基（O(1) 压缩表征，用于传输 kda_state_v1 与替代注意力路径） */
    std::vector<float> K_basis;  /* [num_heads][ortho_base_dim][head_dim] */
    std::vector<float> V_basis;  /* [num_heads][ortho_base_dim][head_dim] */
    std::vector<float> decay;    /* [ortho_base_dim]（按年龄，全局共享） */
    int current_dim;             /* <= ortho_base_dim */
};

/* 从「当前窗口原始 K/V」重建局部正交基（re-orthogonalize，抑制 ring 漂移）。 */
static void wokdc_rebuild(window_ortho_kv_compressor* c) {
    const int hd = c->head_dim;
    const int Wc = c->win_count;
    const int B = std::min(Wc, c->ortho_base_dim);

    std::vector<float> basis(B * hd, 0.0f);
    std::vector<float> tmp(hd);

    for (int h = 0; h < c->num_heads; ++h) {
        for (int i = 0; i < B; ++i) {
            /* 取窗口第 i 个 token（最旧在前），做正交化 */
            int phys = (c->win_head + i) % c->window_capacity;
            const float* kw = &c->win_k[(phys * c->num_heads + h) * hd];
            for (int d = 0; d < hd; ++d) tmp[d] = kw[d];
            gram_schmidt_cpu(tmp.data(), basis.data(), i, hd);

            /* 记录正交基向量 e_i */
            for (int d = 0; d < hd; ++d) basis[i * hd + d] = tmp[d];

            /* V_basis[i] = Σ_j (k_w[j]·e_i) * v_w[j] —— 窗口 V 在 e_i 上的低秩投影 */
            std::vector<float> vb(hd, 0.0f);
            for (int j = 0; j < Wc; ++j) {
                int pj = (c->win_head + j) % c->window_capacity;
                const float* kj = &c->win_k[(pj * c->num_heads + h) * hd];
                const float* vj = &c->win_v[(pj * c->num_heads + h) * hd];
                float proj = 0.0f;
                for (int d = 0; d < hd; ++d) proj += kj[d] * tmp[d];
                for (int d = 0; d < hd; ++d) vb[d] += proj * vj[d];
            }
            for (int d = 0; d < hd; ++d)
                c->V_basis[(h * c->ortho_base_dim + i) * hd + d] = vb[d];

            /* 衰减按年龄：最新 token age=0，权重最高 */
            c->decay[i] = std::exp(-(float)(Wc - 1 - i) * c->decay_rate);
        }
        /* 拷贝本头的 K 基 */
        std::memcpy(&c->K_basis[h * c->ortho_base_dim * hd],
                    basis.data(), B * hd * sizeof(float));
    }
    c->current_dim = B;
}

window_ortho_kv_compressor_t* wokdc_create(int num_heads, int head_dim,
                                           int ortho_base_dim, float decay_rate) {
    auto* c = new window_ortho_kv_compressor();
    c->num_heads = num_heads;
    c->head_dim = head_dim;
    c->ortho_base_dim = ortho_base_dim;
    c->window_capacity = ortho_base_dim > 0 ? ortho_base_dim : 1; /* 默认与基同容量；由 R-SWA 重设 */
    c->decay_rate = decay_rate;
    c->win_head = 0;
    c->win_count = 0;
    c->win_k.resize((size_t)c->window_capacity * num_heads * head_dim, 0.0f);
    c->win_v.resize((size_t)c->window_capacity * num_heads * head_dim, 0.0f);
    c->K_basis.resize((size_t)num_heads * ortho_base_dim * head_dim, 0.0f);
    c->V_basis.resize((size_t)num_heads * ortho_base_dim * head_dim, 0.0f);
    c->decay.assign(ortho_base_dim, 0.0f);
    c->current_dim = 0;
    return c;
}

void wokdc_destroy(window_ortho_kv_compressor_t* c) {
    delete static_cast<window_ortho_kv_compressor*>(c);
}

void wokdc_set_window_capacity(window_ortho_kv_compressor_t* c, int window_size) {
    auto* self = static_cast<window_ortho_kv_compressor*>(c);
    if (window_size <= 0) return;
    if (window_size == self->window_capacity) return;
    /* 重建原始窗口环缓冲为新容量（保留现有内容到最小长度） */
    std::vector<float> new_k((size_t)window_size * self->num_heads * self->head_dim, 0.0f);
    std::vector<float> new_v((size_t)window_size * self->num_heads * self->head_dim, 0.0f);
    int keep = std::min(self->win_count, window_size);
    for (int j = 0; j < keep; ++j) {
        int old_phys = (self->win_head + j) % self->window_capacity;
        int new_phys = j; /* 紧凑到头部 */
        std::memcpy(&new_k[(size_t)new_phys * self->num_heads * self->head_dim],
                    &self->win_k[(size_t)old_phys * self->num_heads * self->head_dim],
                    (size_t)self->num_heads * self->head_dim * sizeof(float));
        std::memcpy(&new_v[(size_t)new_phys * self->num_heads * self->head_dim],
                    &self->win_v[(size_t)old_phys * self->num_heads * self->head_dim],
                    (size_t)self->num_heads * self->head_dim * sizeof(float));
    }
    self->win_k.swap(new_k);
    self->win_v.swap(new_v);
    self->window_capacity = window_size;
    self->win_head = 0;
    self->win_count = keep;
    wokdc_rebuild(self);
}

void wokdc_feed(window_ortho_kv_compressor_t* c, const float* k, const float* v) {
    auto* self = static_cast<window_ortho_kv_compressor*>(c);
    const int hd = self->head_dim;
    const int nh = self->num_heads;
    const int cap = self->window_capacity;

    int phys;
    if (self->win_count < cap) {
        phys = (self->win_head + self->win_count) % cap;
        self->win_count++;
    } else {
        phys = self->win_head;            /* 淘汰最旧 */
        self->win_head = (self->win_head + 1) % cap;
    }
    std::memcpy(&self->win_k[(size_t)phys * nh * hd], k, (size_t)nh * hd * sizeof(float));
    std::memcpy(&self->win_v[(size_t)phys * nh * hd], v, (size_t)nh * hd * sizeof(float));

    wokdc_rebuild(self);
}

void wokdc_on_window_advance(window_ortho_kv_compressor_t* c, int /*evicted_slot*/) {
    /* 解耦接线变体：R-SWA 自行持有窗口并通知压缩器「某槽被淘汰」。
     * 默认 wokdc_feed 路径已内含淘汰+重建；此处仅触发从当前窗口环重建基。 */
    wokdc_rebuild(static_cast<window_ortho_kv_compressor*>(c));
}

void wokdc_attention(const window_ortho_kv_compressor_t* c, const float* Q, float* out) {
    const auto* self = static_cast<const window_ortho_kv_compressor*>(c);
    const int hd = self->head_dim;
    const int nh = self->num_heads;
    const int B = self->ortho_base_dim;

    for (int h = 0; h < nh; ++h) {
        const float* qh = Q + h * hd;
        float* oh = out + h * hd;
        for (int d = 0; d < hd; ++d) oh[d] = 0.0f;

        for (int i = 0; i < self->current_dim; ++i) {
            const float* kb = &self->K_basis[(h * B + i) * hd];
            const float* vb = &self->V_basis[(h * B + i) * hd];
            float score = 0.0f;
            for (int d = 0; d < hd; ++d) score += qh[d] * kb[d];
            float w = score * self->decay[i];
            for (int d = 0; d < hd; ++d) oh[d] += w * vb[d];
        }
    }
}

int wokdc_current_dim(const window_ortho_kv_compressor_t* c) {
    return static_cast<const window_ortho_kv_compressor*>(c)->current_dim;
}

void wokdc_get_state(const window_ortho_kv_compressor_t* c,
                     float* K, float* V, float* decay, int* current_dim) {
    const auto* self = static_cast<const window_ortho_kv_compressor*>(c);
    const size_t cap = (size_t)self->num_heads * self->ortho_base_dim * self->head_dim;
    std::memset(K, 0, cap * sizeof(float));
    std::memset(V, 0, cap * sizeof(float));
    std::memcpy(K, self->K_basis.data(), cap * sizeof(float));
    std::memcpy(V, self->V_basis.data(), cap * sizeof(float));
    std::memcpy(decay, self->decay.data(), self->ortho_base_dim * sizeof(float));
    *current_dim = self->current_dim;
}

void wokdc_set_state(window_ortho_kv_compressor_t* c,
                     const float* K, const float* V, const float* decay, int current_dim) {
    auto* self = static_cast<window_ortho_kv_compressor*>(c);
    const size_t cap = (size_t)self->num_heads * self->ortho_base_dim * self->head_dim;
    std::memcpy(self->K_basis.data(), K, cap * sizeof(float));
    std::memcpy(self->V_basis.data(), V, cap * sizeof(float));
    std::memcpy(self->decay.data(), decay, self->ortho_base_dim * sizeof(float));
    self->current_dim = current_dim;
}
