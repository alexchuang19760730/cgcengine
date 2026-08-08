// =============================================================================
// CGC Engine Linkage Demo — cgc_inject_strategy <-> inject_unified_ir_for_role
// 验证「统一注入入口」与「引擎策略注入」真正联动：
//   1. 调用 cgc_inject_unified_ir_for_role -> 其实现回调 cgc_inject_strategy；
//   2. cgc_inject_strategy 检测到 unified_ir 元数据 -> 自动 cgc_set_kda_replace_mode(true)；
//   3. 随后 cgc_execute_opcode(0x10 SDPA) 应被重路由到 0x11(KDA)。
// 编译：见 build_unified.sh
// =============================================================================
#include "cgc_cpp.h"   // 包含 unified_ir.h（已暴露 cgc_inject_unified_ir_for_role）

#include <cstdio>
#include <cstring>
#include <cstdlib>

int main() {
    printf("==============================================================\n");
    printf(" CGC Engine Linkage Demo (strategy <-> unified IR)\n");
    printf("==============================================================\n");

    cgc_init();

    /* 构造统一 IR 配置并注入（decode 角色，sglang adapter） */
    cgc_unified_ir_config_t ucfg;
    memset(&ucfg, 0, sizeof(ucfg));
    ucfg.rswa.num_heads      = 2;
    ucfg.rswa.head_dim       = 8;
    ucfg.rswa.reference_len  = 4;
    ucfg.rswa.window_size    = 8;
    ucfg.rswa.ortho_base_dim = 8;
    ucfg.rswa.decay_rate     = 0.01f;
    ucfg.rswa.hybrid_every   = 0;
    ucfg.rswa.enable_window_ortho = 1;
    ucfg.role   = CGC_INJECT_ROLE_DECODE;
    ucfg.adapter = CGC_BACKEND_ADAPTER_SGLANG;
    ucfg.enable_window_ortho  = true;
    ucfg.enable_reference_std = true;
    ucfg.num_layers = 32;

    printf("\n-- [1] call cgc_inject_unified_ir_for_role --\n");
    cgc_inject_unified_ir_for_role(ucfg.role, nullptr, &ucfg, ucfg.adapter);

    /* 联动验证：KDA replace mode 应被自动开启 */
    printf("\n-- [2] verify linkage: kda_replace_mode should be ON --\n");
    bool kda = cgc_get_kda_replace_mode();
    printf("[linkage] cgc_get_kda_replace_mode() = %s  <-- expect ON\n",
           kda ? "ON" : "OFF");

    /* 验证全局策略确实被统一 IR 写入 */
    printf("\n-- [3] verify global strategy metadata --\n");
    cgc_strategy_t strat;
    cgc_get_strategy(&strat);
    printf("[strategy] metadata = %s\n", strat.metadata);

    /* 验证重路由：SDPA(0x10) -> KDA(0x11) */
    printf("\n-- [4] verify SDPA->KDA reroute under kda_replace_mode --\n");
    const float* in[1] = {nullptr};
    float* out[1] = {nullptr};
    int64_t odims[1] = {16}; int ondim[1] = {1};
    cgc_execute_opcode(0x10, in, nullptr, nullptr, 0, out, odims, ondim, 1, nullptr);

    /* IR 摘要 */
    char* sum = cgc_unified_ir_summary(CGC_INJECT_ROLE_DECODE);
    printf("\n-- [5] unified IR summary --\n");
    printf("[summary] %s\n", sum ? sum : "(null)");
    if (sum) free(sum);

    printf("\n==============================================================\n");
    printf(" %s\n", (kda && strstr(strat.metadata, "unified_ir:")) ? "Linkage OK ✅" : "Linkage FAILED ❌");
    printf("==============================================================\n");

    cgc_destroy();
    return 0;
}
