// =============================================================================
// CGC Unified IR Demo — R-SWA + Window OrthoKDA (C/C++ CPU path)
// 演示：R-SWA 管理 Reference/recency、窗口 OrthoKDA 压缩、统一注入入口。
// 编译：见 build_unified.sh（纯 g++，不需要 CUDA）
// =============================================================================
#include "unified_ir.h"
#include "kernels/rswa_manager.h"
#include "kernels/window_ortho_kv_compressor.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cstring>

static float rnd() { return ((float)rand() / (float)RAND_MAX - 0.5f) * 0.2f; }

int main() {
    printf("==============================================================\n");
    printf(" CGC Unified IR Demo (R-SWA + Window OrthoKDA) — C/C++\n");
    printf("==============================================================\n");

    const int NH = 2, HD = 8, REF = 4, WIN = 8, ORTHO = 8;

    rswa_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.num_heads = NH;
    cfg.head_dim = HD;
    cfg.reference_len = REF;
    cfg.window_size = WIN;
    cfg.ortho_base_dim = ORTHO;
    cfg.decay_rate = 0.01f;
    cfg.hybrid_every = 0;        /* 全正交 */
    cfg.enable_window_ortho = 1; /* 启用窗口正交压缩 */

    rswa_manager_t* mgr = rswa_create(&cfg);

    /* 注入 Reference 永久区 */
    for (int i = 0; i < REF; ++i) {
        std::vector<float> k(NH * HD), v(NH * HD);
        for (auto& x : k) x = rnd();
        for (auto& x : v) x = rnd();
        rswa_set_reference(mgr, k.data(), v.data());
    }

    /* 喂入超过窗口容量的 token，触发滑动淘汰 + 压缩器重建 */
    const int N = 20;
    for (int t = 0; t < N; ++t) {
        std::vector<float> k(NH * HD), v(NH * HD);
        for (auto& x : k) x = rnd();
        for (auto& x : v) x = rnd();
        int ev = rswa_feed_token(mgr, k.data(), v.data());
        if (t >= WIN && (t % 5 == 0)) {
            printf("[feed] step=%d evicted_ring_slot=%d visible=%d wokdc_dim=%d\n",
                   t, ev, (int)rswa_visible_count(mgr),
                   wokdc_current_dim((window_ortho_kv_compressor_t*)rswa_window_compressor(mgr)));
        }
    }

    /* 可见性掩码 */
    int seq_len = REF + WIN;
    std::vector<uint8_t> mask(seq_len, 0);
    rswa_visibility_mask(mgr, seq_len, mask.data());
    int vis = 0; for (auto b : mask) vis += b;
    printf("[mask] seq_len=%d visible=%d (expect ref+win=%d)\n",
           seq_len, vis, (int)rswa_visible_count(mgr));

    /* 逐层角色（混合策略示例：hybrid_every=2） */
    rswa_config_t cfg2 = cfg; cfg2.hybrid_every = 2;
    rswa_manager_t* mgr2 = rswa_create(&cfg2);
    printf("[layers] hybrid_every=2 over 6 layers: ");
    for (int l = 0; l < 6; ++l) {
        rswa_layer_role_t r = rswa_layer_role(mgr2, l);
        printf("%s ", r == RSWA_LAYER_WINDOW_ORTHO ? "ORTHO" : "STD");
    }
    printf("(ref=STD always)\n");
    rswa_destroy(mgr2);

    /* 合并注意力（reference ∪ window-ortho）数值健全性 */
    std::vector<float> Q(NH * HD), out(NH * HD);
    for (auto& x : Q) x = rnd();
    rswa_combined_attention(mgr, Q.data(), out.data());
    bool finite = true; float sum = 0.0f;
    for (auto x : out) { if (!std::isfinite(x)) finite = false; sum += x * x; }
    printf("[attn] combined output finite=%s norm=%.4f\n", finite ? "yes" : "NO", sqrtf(sum));

    /* 压缩器 state 序列化 round-trip（兼容 kda_state_v1） */
    auto* wok = (window_ortho_kv_compressor_t*)rswa_window_compressor(mgr);
    std::vector<float> K(NH * ORTHO * HD), V(NH * ORTHO * HD), decay(ORTHO);
    int cdim = 0;
    wokdc_get_state(wok, K.data(), V.data(), decay.data(), &cdim);
    window_ortho_kv_compressor_t* wok2 = wokdc_create(NH, HD, ORTHO, 0.01f);
    wokdc_set_window_capacity(wok2, WIN);
    wokdc_set_state(wok2, K.data(), V.data(), decay.data(), cdim);
    std::vector<float> K2(NH * ORTHO * HD), V2(NH * ORTHO * HD), decay2(ORTHO);
    int cdim2 = 0;
    wokdc_get_state(wok2, K2.data(), V2.data(), decay2.data(), &cdim2);
    float maxdiff = 0.0f;
    for (size_t i = 0; i < K.size(); ++i) maxdiff = std::max(maxdiff, std::fabs(K[i] - K2[i]));
    printf("[state] kda_state_v1 round-trip max|K diff|=%.6e dim=%d->%d\n",
           maxdiff, cdim, cdim2);
    wokdc_destroy(wok2);

    /* 统一注入：sglang + cloud_sglang(M7.5) */
    cgc_unified_ir_config_t ucfg;
    memset(&ucfg, 0, sizeof(ucfg));
    ucfg.rswa = cfg;
    ucfg.role = CGC_INJECT_ROLE_DECODE;
    ucfg.adapter = CGC_BACKEND_ADAPTER_SGLANG;
    ucfg.enable_window_ortho = true;
    ucfg.enable_reference_std = true;
    ucfg.num_layers = 32;
    cgc_inject_unified_ir_for_role(ucfg.role, nullptr, &ucfg, ucfg.adapter);

    cgc_unified_ir_config_t ucfg2 = ucfg;
    ucfg2.adapter = CGC_BACKEND_ADAPTER_CLOUD_SGLANG;
    ucfg2.role = CGC_INJECT_ROLE_EDGE_RESUME;
    cgc_inject_unified_ir_for_role(ucfg2.role, nullptr, &ucfg2, ucfg2.adapter);

    char* sum1 = cgc_unified_ir_summary(CGC_INJECT_ROLE_DECODE);
    printf("[summary] %s\n", sum1);
    free(sum1);

    /* IR 层序列 */
    std::vector<rswa_layer_role_t> layers(ucfg.num_layers);
    cgc_build_unified_ir_layers(&ucfg, ucfg.num_layers, layers.data());
    int o = 0; for (auto l : layers) if (l == RSWA_LAYER_WINDOW_ORTHO) ++o;
    printf("[ir] %d layers -> window_ortho=%d window_std=%d (ref_std forced)\n",
           ucfg.num_layers, o, ucfg.num_layers - o);

    rswa_destroy(mgr);
    printf("==============================================================\n");
    printf(" Demo OK\n");
    printf("==============================================================\n");
    return 0;
}
