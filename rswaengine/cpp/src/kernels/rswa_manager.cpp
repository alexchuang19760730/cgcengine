#include "kernels/rswa_manager.h"
#include "kernels/window_ortho_kv_compressor.h"

#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>

namespace {

inline float dot(const float* a, const float* b, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

/* 数值稳定 softmax，返回指针需调用方分配 scores[n] / out 由调用方清零后累加 */
inline void softmax_scale(float* scores, int n, float scale) {
    float maxv = -1e30f;
    for (int i = 0; i < n; ++i) { scores[i] *= scale; if (scores[i] > maxv) maxv = scores[i]; }
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) { scores[i] = std::exp(scores[i] - maxv); sum += scores[i]; }
    float inv = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
    for (int i = 0; i < n; ++i) scores[i] *= inv;
}

} // namespace

struct rswa_manager {
    rswa_config_t cfg;
    std::vector<float> ref_k;   /* [reference_len][num_heads][head_dim] */
    std::vector<float> ref_v;
    int ref_count;

    std::vector<float> win_k;   /* [window_size][num_heads][head_dim] 环形 */
    std::vector<float> win_v;
    int win_head;
    int win_count;

    window_ortho_kv_compressor_t* wokdc; /* 窗口作用域正交压缩器（可空） */
};

rswa_manager_t* rswa_create(const rswa_config_t* cfg) {
    auto* m = new rswa_manager();
    m->cfg = *cfg;
    const int nh = cfg->num_heads;
    const int hd = cfg->head_dim;
    m->ref_k.resize((size_t)cfg->reference_len * nh * hd, 0.0f);
    m->ref_v.resize((size_t)cfg->reference_len * nh * hd, 0.0f);
    m->ref_count = 0;
    m->win_k.resize((size_t)cfg->window_size * nh * hd, 0.0f);
    m->win_v.resize((size_t)cfg->window_size * nh * hd, 0.0f);
    m->win_head = 0;
    m->win_count = 0;
    m->wokdc = nullptr;

    if (cfg->enable_window_ortho != 0 && cfg->window_size > 0 && cfg->ortho_base_dim > 0) {
        m->wokdc = wokdc_create(nh, hd, cfg->ortho_base_dim, cfg->decay_rate);
        wokdc_set_window_capacity(m->wokdc, cfg->window_size);
    }
    return m;
}

void rswa_destroy(rswa_manager_t* m) {
    auto* self = static_cast<rswa_manager*>(m);
    if (self->wokdc) wokdc_destroy(self->wokdc);
    delete self;
}

void rswa_set_reference(rswa_manager_t* m, const float* k, const float* v) {
    auto* self = static_cast<rswa_manager*>(m);
    if (self->ref_count >= self->cfg.reference_len) return;
    const int nh = self->cfg.num_heads;
    const int hd = self->cfg.head_dim;
    const size_t off = (size_t)self->ref_count * nh * hd;
    std::memcpy(&self->ref_k[off], k, (size_t)nh * hd * sizeof(float));
    std::memcpy(&self->ref_v[off], v, (size_t)nh * hd * sizeof(float));
    self->ref_count++;
}

int rswa_feed_token(rswa_manager_t* m, const float* k, const float* v) {
    auto* self = static_cast<rswa_manager*>(m);
    const int nh = self->cfg.num_heads;
    const int hd = self->cfg.head_dim;
    const int cap = self->cfg.window_size;

    int evicted = -1;
    int phys;
    if (self->win_count < cap) {
        phys = (self->win_head + self->win_count) % cap;
        self->win_count++;
    } else {
        evicted = self->win_head;
        phys = self->win_head;
        self->win_head = (self->win_head + 1) % cap;
    }
    std::memcpy(&self->win_k[(size_t)phys * nh * hd], k, (size_t)nh * hd * sizeof(float));
    std::memcpy(&self->win_v[(size_t)phys * nh * hd], v, (size_t)nh * hd * sizeof(float));

    /* 窗口 Token 同时灌入窗口压缩器（其内部完成淘汰 + 重正交化） */
    if (self->wokdc) wokdc_feed(self->wokdc, k, v);

    return evicted;
}

void rswa_visibility_mask(const rswa_manager_t* m, int32_t seq_len, uint8_t* mask) {
    const auto* self = static_cast<const rswa_manager*>(m);
    const int32_t M = self->cfg.reference_len;
    const int32_t W = self->win_count;
    for (int32_t i = 0; i < seq_len; ++i) mask[i] = 0;
    int32_t lim = std::min(seq_len, M + W);
    for (int32_t i = 0; i < lim; ++i) mask[i] = 1; /* [0,M) reference, [M,M+W) window */
}

int32_t rswa_visible_count(const rswa_manager_t* m) {
    const auto* self = static_cast<const rswa_manager*>(m);
    return self->cfg.reference_len + self->win_count;
}

rswa_layer_role_t rswa_layer_role(const rswa_manager_t* m, int32_t layer_idx) {
    const auto* self = static_cast<const rswa_manager*>(m);
    bool ortho = (self->cfg.enable_window_ortho != 0) &&
                 (self->cfg.hybrid_every == 0 ||
                  (layer_idx % self->cfg.hybrid_every) == 0);
    return ortho ? RSWA_LAYER_WINDOW_ORTHO : RSWA_LAYER_WINDOW_STD;
    /* 注：Reference 区恒为 RSWA_LAYER_REFERENCE_STD（硬约束），不在此返回。 */
}

void* rswa_window_compressor(rswa_manager_t* m) {
    return static_cast<rswa_manager*>(m)->wokdc;
}

void rswa_reference_attention(const rswa_manager_t* m, const float* Q, float* out) {
    const auto* self = static_cast<const rswa_manager*>(m);
    const int nh = self->cfg.num_heads;
    const int hd = self->cfg.head_dim;
    const int M = self->ref_count;
    const float scale = 1.0f / std::sqrt((float)hd);

    for (int h = 0; h < nh; ++h) {
        float* oh = out + h * hd;
        for (int d = 0; d < hd; ++d) oh[d] = 0.0f;
        if (M == 0) continue;
        std::vector<float> s(M);
        for (int j = 0; j < M; ++j) {
            const float* kj = &self->ref_k[((size_t)j * nh + h) * hd];
            const float* qh = Q + h * hd;
            s[j] = dot(qh, kj, hd);
        }
        softmax_scale(s.data(), M, scale);
        for (int j = 0; j < M; ++j) {
            const float* vj = &self->ref_v[((size_t)j * nh + h) * hd];
            float p = s[j];
            for (int d = 0; d < hd; ++d) oh[d] += p * vj[d];
        }
    }
}

void rswa_window_attention(const rswa_manager_t* m, const float* Q, float* out) {
    const auto* self = static_cast<const rswa_manager*>(m);
    const int nh = self->cfg.num_heads;
    const int hd = self->cfg.head_dim;
    const float scale = 1.0f / std::sqrt((float)hd);

    if (self->wokdc) { wokdc_attention(self->wokdc, Q, out); return; }

    const int W = self->win_count;
    for (int h = 0; h < nh; ++h) {
        float* oh = out + h * hd;
        for (int d = 0; d < hd; ++d) oh[d] = 0.0f;
        if (W == 0) continue;
        std::vector<float> s(W);
        for (int j = 0; j < W; ++j) {
            int phys = (self->win_head + j) % self->cfg.window_size;
            const float* kj = &self->win_k[((size_t)phys * nh + h) * hd];
            s[j] = dot(Q + h * hd, kj, hd);
        }
        softmax_scale(s.data(), W, scale);
        for (int j = 0; j < W; ++j) {
            int phys = (self->win_head + j) % self->cfg.window_size;
            const float* vj = &self->win_v[((size_t)phys * nh + h) * hd];
            float p = s[j];
            for (int d = 0; d < hd; ++d) oh[d] += p * vj[d];
        }
    }
}

void rswa_combined_attention(const rswa_manager_t* m, const float* Q, float* out) {
    const auto* self = static_cast<const rswa_manager*>(m);
    const int nh = self->cfg.num_heads;
    const int hd = self->cfg.head_dim;
    const int M = self->ref_count;
    const float scale = 1.0f / std::sqrt((float)hd);

    const bool use_ortho = (self->wokdc != nullptr);
    const int B = use_ortho ? wokdc_current_dim(self->wokdc) : 0;

    /* 取窗口压缩基（仅正交模式） */
    std::vector<float> Kbuf, Vbuf, decaybuf;
    if (use_ortho) {
        Kbuf.resize((size_t)nh * self->cfg.ortho_base_dim * hd);
        Vbuf.resize((size_t)nh * self->cfg.ortho_base_dim * hd);
        decaybuf.resize(self->cfg.ortho_base_dim);
        int cdim = 0;
        wokdc_get_state(self->wokdc, Kbuf.data(), Vbuf.data(), decaybuf.data(), &cdim);
    }

    const int W = self->win_count;
    const int Ktot = M + (use_ortho ? B : W);

    for (int h = 0; h < nh; ++h) {
        float* oh = out + h * hd;
        for (int d = 0; d < hd; ++d) oh[d] = 0.0f;
        if (Ktot == 0) continue;

        std::vector<float> s(Ktot);
        /* reference 部分（bias=0） */
        for (int j = 0; j < M; ++j) {
            const float* kj = &self->ref_k[((size_t)j * nh + h) * hd];
            s[j] = dot(Q + h * hd, kj, hd) * scale;
        }
        /* 窗口部分：正交基（bias=log decay）或原始窗口（bias=0） */
        for (int j = 0; j < (use_ortho ? B : W); ++j) {
            int idx = M + j;
            if (use_ortho) {
                const float* kb = &Kbuf[((size_t)h * self->cfg.ortho_base_dim + j) * hd];
                float bias = std::log(decaybuf[j] > 1e-8f ? decaybuf[j] : 1e-8f);
                s[idx] = dot(Q + h * hd, kb, hd) * scale + bias;
            } else {
                int phys = (self->win_head + j) % self->cfg.window_size;
                const float* kj = &self->win_k[((size_t)phys * nh + h) * hd];
                s[idx] = dot(Q + h * hd, kj, hd) * scale;
            }
        }
        /* 数值稳定 softmax（已含 scale 与 bias） */
        float maxv = -1e30f;
        for (int i = 0; i < Ktot; ++i) if (s[i] > maxv) maxv = s[i];
        float sum = 0.0f;
        for (int i = 0; i < Ktot; ++i) { s[i] = std::exp(s[i] - maxv); sum += s[i]; }
        float inv = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
        for (int i = 0; i < Ktot; ++i) s[i] *= inv;

        /* 加权求和 */
        for (int j = 0; j < M; ++j) {
            const float* vj = &self->ref_v[((size_t)j * nh + h) * hd];
            float p = s[j];
            for (int d = 0; d < hd; ++d) oh[d] += p * vj[d];
        }
        for (int j = 0; j < (use_ortho ? B : W); ++j) {
            int idx = M + j;
            const float* vj = use_ortho
                ? &Vbuf[((size_t)h * self->cfg.ortho_base_dim + j) * hd]
                : &self->win_v[((size_t)((self->win_head + j) % self->cfg.window_size) * nh + h) * hd];
            float p = s[idx];
            for (int d = 0; d < hd; ++d) oh[d] += p * vj[d];
        }
    }
}
