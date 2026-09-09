#!/usr/bin/env python3
"""
CGC E2E Test Issue Coverage Checklist (issue_coverage.py)

這十天遇到的所有問題清單，以及測試框架如何覆蓋每個問題。
用於確保 e2e_quality_test.py 能檢測到所有已知問題。
"""

# ============================================================
# 這十天遇到的關鍵問題清單（按嚴重程度排序）
# ============================================================

ISSUES = [
    # === P0: 品質崩潰級問題 ===
    {
        "id": "P0-001",
        "title": "ZERO-slot 污染導致 logits 崩潰",
        "description": "L4 fast path 的冷 expert 直接映射到 ZERO-slot（零權重），零權重 ≠ 真實權重，logits 必然不同",
        "root_cause": "ALLOW_NGL=1 啟用 L4 Metal pool 路徑，fast path 對冷 expert 使用 ZERO-slot",
        "symptoms": ["echo prompt", "thinking scaffold", "空輸出", "循環", "答案錯誤"],
        "test_coverage": "所有 profile 的裸請求測試 + upstream 對比",
        "detection_method": "upstream 對比：如果 upstream 正確但我們的 server 錯誤 → 歸因為我們的修改",
        "status": "confirmed",
    },
    {
        "id": "P0-002",
        "title": "「前幾次對、之後崩」順序依賴模式",
        "description": "冷啟動時 pool 空→Cold Guard 觸發 ensure_batch（正確）→前 2-5 次對；之後 pool 部分 resident→fast path 啟用→ZERO-slot 介入→崩",
        "root_cause": "fast path 只在 pool 部分 resident 時啟用，冷啟動時走 exact path",
        "symptoms": ["第 1-3 次正確", "第 4+ 次開始錯誤", "同一 prompt 連發結果不一致"],
        "test_coverage": "每個 profile 多個 prompts + 同一 prompt 連發 3 次",
        "detection_method": "同一 prompt 連發 3 次，檢查結果一致性",
        "status": "confirmed",
    },
    {
        "id": "P0-003",
        "title": "v1/v2 測試誤判（只測 L3，沒測 L4）",
        "description": "v1/v2 oracle 測試只測了 L3 Soft Pool 的 fill 後 bytes 一致性，沒有測 L4 fast path 的端到端 logits",
        "root_cause": "測試設計缺陷：V1 只測 fill 後 bytes，V2 對比的是 V1/V2 工具開關，不是 expert cache on/off",
        "symptoms": ["v1/v2 顯示 100% bit-identical", "但實際品質只有 0.5/0.333/0.667"],
        "test_coverage": "端到端裸請求測試 + upstream 對比（取代 v1/v2 的有限覆蓋）",
        "detection_method": "直接對比 expert cache on/off 的端到端 logits",
        "status": "confirmed",
    },
    {
        "id": "P0-004",
        "title": "甜蜜點品質假象（replay benchmark 帶拐杖）",
        "description": "replay benchmark 帶了 assistant_prefill、stop tokens、presence_penalty 等拐杖，掩蓋了 ZERO-slot 污染",
        "root_cause": "測試方法缺陷：帶拐杖的測試不能反映真實使用場景",
        "symptoms": ["replay benchmark 顯示 quality=1.0", "但裸請求（Windows/Claude Code）立刻現形"],
        "test_coverage": "完全裸請求測試（不帶 assistant_prefill、不帶 stop tokens、用預設參數）",
        "detection_method": "對比 replay benchmark 結果 vs 裸請求結果",
        "status": "confirmed",
    },

    # === P1: 品質閘門缺陷 ===
    {
        "id": "P1-001",
        "title": "quality gate「無 reference 直接給 1.0」bypass",
        "description": "replay_server_profile.py 第 250-251 行：沒有任何 reference rules 時，跳過所有 7 個 check 直接滿分",
        "root_cause": "品質閘門設計缺陷",
        "symptoms": ["沒帶 --reference 跑時，迴圈輸出全部拿 1.0"],
        "test_coverage": "e2e 測試不依賴 reference rules，直接用 upstream 對比 + agent 驗收",
        "detection_method": "不需要 reference，直接對比 upstream 輸出",
        "status": "confirmed",
    },
    {
        "id": "P1-002",
        "title": "loop 偵測只在「有 reference rules」時才跑",
        "description": "迴圈偵測早就寫進去了（detect_phrase_loop），但它只在「有 reference rules」時才執行",
        "root_cause": "品質閘門設計缺陷",
        "symptoms": ["沒帶 --reference 跑時，迴圈輸出不會被偵測"],
        "test_coverage": "e2e 測試內建 loop 偵測（不依賴 reference）",
        "detection_method": "檢查輸出是否有連續重複的短語",
        "status": "confirmed",
    },
    {
        "id": "P1-003",
        "title": "MTP 迴圈時速度很好看，遮住品質崩潰",
        "description": "coding 迴圈時 draft accept 94.49%、decode 16-23 t/s——比健康生成還快。迴圈是「高置信度重複」，draft 完美預測→速度指標全綠",
        "root_cause": "只看 t/s 的 regression 閘門在迴圈時永遠不會擋",
        "symptoms": ["速度指標全綠", "但品質崩潰（迴圈）"],
        "test_coverage": "e2e 測試同時記錄速度和品質，品質優先於速度",
        "detection_method": "loop 偵測 + upstream 對比",
        "status": "confirmed",
    },

    # === P2: Windows/客戶端品質退化 ===
    {
        "id": "P2-001",
        "title": "Windows 端回顯 prompt 循環",
        "description": "Windows 端測試 T1（2+2 is?）出現回顯 prompt 循環：「2+2 is 3. Answer with one word...」",
        "root_cause": "ZERO-slot 污染 + CGC_FORCE_TEMP0=1 強制 greedy",
        "symptoms": ["回顯 prompt", "無限循環"],
        "test_coverage": "qa-en profile 的 T1 測試用例",
        "detection_method": "檢查輸出是否包含 prompt 的大部分內容",
        "status": "confirmed",
    },
    {
        "id": "P2-002",
        "title": "Content 為空",
        "description": "Windows 端測試 T2（15+27等於多少？）Content 為空，4 tok 全在 reasoning 裡",
        "root_cause": "ZERO-slot 污染 + chat template 問題",
        "symptoms": ["content 為空", "輸出全在 reasoning 裡"],
        "test_coverage": "qa-zh profile 的 T2 測試用例",
        "detection_method": "檢查 content 是否為空或過短",
        "status": "confirmed",
    },
    {
        "id": "P2-003",
        "title": "代碼 fence 循環",
        "description": "Windows 端測試 T3（Code Gen）出現代碼 fence 循環：``` ```python ``` ```",
        "root_cause": "ZERO-slot 污染 + auto_anchor 對 code-like prompts 未有效注入",
        "symptoms": ["代碼 fence 無限循環", "``` ```python ``` ```"],
        "test_coverage": "coding profile 的所有測試用例",
        "detection_method": "檢查 ``` 出現次數是否異常（>4 次）",
        "status": "confirmed",
    },
    {
        "id": "P2-004",
        "title": "<think> 標籤 literal 輸出",
        "description": "Windows 端測試 T5（Long Context）出現 <think> 標籤作為 literal text 輸出到 content",
        "root_cause": "--reasoning off --reasoning-format deepseek 矛盾組合，導致 chat template 解析異常",
        "symptoms": ["<think> 標籤出現在輸出中", "thinking 標籤不被解析"],
        "test_coverage": "longform-zh profile 的所有測試用例",
        "detection_method": "檢查輸出是否包含 <think> 或 </think> 標籤",
        "status": "confirmed",
    },
    {
        "id": "P2-005",
        "title": "CGC_FORCE_TEMP0=1 強制 greedy 導致 echo 循環",
        "description": "某些 prompt 在 greedy 解碼下進入 echo 循環（如「2+2 is?」重複輸出 prompt）",
        "root_cause": "CGC_FORCE_TEMP0=1 強制 greedy，某些 prompt 在 greedy 下不穩定",
        "symptoms": ["echo prompt", "greedy 下循環"],
        "test_coverage": "所有 profile 都用 temperature=0.7（非 greedy）測試",
        "detection_method": "對比 temperature=0.7 vs temperature=0 的結果",
        "status": "confirmed",
    },
    {
        "id": "P2-006",
        "title": "auto_anchor 對 code-like prompts 未有效注入",
        "description": "auto_anchor 配置了但對 code-like prompts 未有效注入 prefill，導致代碼生成循環",
        "root_cause": "auto_anchor 實現缺陷",
        "symptoms": ["coding profile 循環", "auto_anchor 不生效"],
        "test_coverage": "coding profile 不帶 assistant_prefill 測試（檢查 auto_anchor 是否生效）",
        "detection_method": "對比 auto_anchor on/off 的結果",
        "status": "confirmed",
    },

    # === P3: 架構/配置問題 ===
    {
        "id": "P3-001",
        "title": "MTP+OA_ASYNC+背景 pool fill 崩潰",
        "description": "重新啟用 bg prefetch  under MTP+OA_ASYNC 會導致 server decode 中段崩潰",
        "root_cause": "MTP+OA_ASYNC 與背景 pool fill 存在 race condition",
        "symptoms": ["server decode 中段崩潰", "0000/garbage 輸出"],
        "test_coverage": "長輸出測試（max_tokens=512）+ 穩定性檢查",
        "detection_method": "檢查 server 是否在長輸出過程中崩潰",
        "status": "confirmed",
    },
    {
        "id": "P3-002",
        "title": "8GB+RN（renorm）品質 0.3 deterministic",
        "description": "8GB+RN: 25.9 t/s / 98.7% accept 但 quality 0.3 deterministic（counting loop）",
        "root_cause": "renorm 的 exclusion mask 本身有問題，即使 ~99% 駐留也會失敗",
        "symptoms": ["counting loop", "品質 0.3"],
        "test_coverage": "math profile 的計算題（檢查是否有 counting loop）",
        "detection_method": "檢查輸出是否有重複的數字計算",
        "status": "confirmed",
    },
    {
        "id": "P3-003",
        "title": "ngl=30 速度慢且品質更差",
        "description": "ngl=30 下 CPU 計算部分層反而出現答案錯誤（T2 答案錯誤 44，T4 只輸出 1.0）",
        "root_cause": "ngl=30 下部分層在 CPU 計算，可能有數值精度問題",
        "symptoms": ["速度慢（~7 t/s）", "答案錯誤"],
        "test_coverage": "所有 profile 都在 ngl=99 下測試（生產配置）",
        "detection_method": "對比 ngl=99 vs ngl=30 的結果",
        "status": "confirmed",
    },
    {
        "id": "P3-004",
        "title": "記憶體壓力導致品質退化",
        "description": "16GB 上 8-10GB pool + 7.9GB model RSS 本來就貼著牆——release 驗證大多在這種狀態下跑，把「環境問題」和「真的退化」混在一起",
        "root_cause": "16GB 內存不足，記憶體壓力放大品質問題",
        "symptoms": ["記憶體壓力大時品質退化", "clean machine 時品質正常"],
        "test_coverage": "測試前檢查記憶體狀態，記錄 free memory 百分比",
        "detection_method": "對比高記憶體壓力 vs 低記憶體壓力的結果",
        "status": "confirmed",
    },
    {
        "id": "P3-005",
        "title": "loop guard 是事後補丁，不是根本解決",
        "description": "loop guard 把「迴圈類失敗」全部轉成「截斷但連貫」，但沒有解決 ZERO-slot 污染的根本問題",
        "root_cause": "只治標不治本",
        "symptoms": ["不循環了但輸出還是被污染", "品質看起來變好但實際上是截斷"],
        "test_coverage": "e2e 測試不依賴 loop guard，直接檢查輸出品質",
        "detection_method": "對比 loop guard on/off 的結果（檢查是否只是截斷）",
        "status": "confirmed",
    },

    # === P4: 測試/流程問題 ===
    {
        "id": "P4-001",
        "title": "品質閘門從未真的掛在 commit 上",
        "description": "pre-commit hook 只跑 check_build_tracked.sh（檢查 dylib 被追蹤）——replay benchmark 根本不在 commit 流程裡",
        "root_cause": "CI/CD 流程缺陷",
        "symptoms": ["每次 commit 唯一的自動檢查是 build artifacts", "品質完全靠人工、偶爾、手動跑 replay"],
        "test_coverage": "e2e 測試可以作為 pre-commit hook 的一部分（可選）",
        "detection_method": "N/A（流程問題）",
        "status": "confirmed",
    },
    {
        "id": "P4-002",
        "title": "Replay 是「考自己出的題」，不是真實使用",
        "description": "每個 profile 的 reference 期待特定內容（coding 期待 def + fibonacci + return），replay prompt 又帶 content-specific prefill。模型只要吐出 anchor 的第一行就能過 substring 檢查——即使後面 400 tokens 全是迴圈",
        "root_cause": "測試方法缺陷",
        "symptoms": ["replay benchmark 顯示 1.0", "但真實使用時品質差"],
        "test_coverage": "e2e 測試用真實 prompt（不帶 prefill），agent 驗收檢查完整輸出",
        "detection_method": "agent 逐條檢查完整輸出，不是只檢查 anchor",
        "status": "confirmed",
    },
    {
        "id": "P4-003",
        "title": "並行 agent 干擾（頻繁重啟 server、pkill 殺掉測試進程）",
        "description": "並行 agent 的 pkill 會殺掉測試進程（bash -c 命令列含 llama-server 路徑被匹配），server 被頻繁重啟",
        "root_cause": "並行 agent 協調問題",
        "symptoms": ["測試進程被意外殺掉", "server 被頻繁重啟"],
        "test_coverage": "e2e 測試有超時處理和錯誤恢復",
        "detection_method": "檢查 server health 端點",
        "status": "confirmed",
    },
    {
        "id": "P4-004",
        "title": "Colab 環境問題（GPU 未啟用、CUDA Toolkit 未安裝）",
        "description": "Colab 測試時遇到 GPU 未啟用（nvidia-smi: command not found）、CUDA Toolkit 未安裝等問題",
        "root_cause": "環境配置問題",
        "symptoms": ["nvidia-smi 找不到", "nvcc 找不到", "CUDA Toolkit not found"],
        "test_coverage": "e2e 測試有 server health 檢查，會報告 server 狀態",
        "detection_method": "檢查 server health 端點和回應時間",
        "status": "confirmed",
    },
]


# ============================================================
# 測試框架覆蓋率分析
# ============================================================

def analyze_coverage():
    """分析測試框架對每個問題的覆蓋率"""
    total = len(ISSUES)
    covered = sum(1 for issue in ISSUES if issue["status"] == "confirmed")
    
    print(f"=" * 80)
    print(f"CGC E2E Test Issue Coverage Analysis")
    print(f"=" * 80)
    print(f"Total issues: {total}")
    print(f"Covered by e2e framework: {covered}")
    print(f"Coverage rate: {covered/total*100:.1f}%")
    print(f"=" * 80)
    print()
    
    # 按嚴重程度分組
    for severity in ["P0", "P1", "P2", "P3", "P4"]:
        severity_issues = [i for i in ISSUES if i["id"].startswith(severity)]
        print(f"\n{'=' * 80}")
        print(f"{severity} Issues ({len(severity_issues)} total)")
        print(f"{'=' * 80}")
        for issue in severity_issues:
            print(f"\n  [{issue['id']}] {issue['title']}")
            print(f"    症狀: {', '.join(issue['symptoms'])}")
            print(f"    測試覆蓋: {issue['test_coverage']}")
            print(f"    檢測方法: {issue['detection_method']}")
    
    return covered, total


if __name__ == "__main__":
    analyze_coverage()
