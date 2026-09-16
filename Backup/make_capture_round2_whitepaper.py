#!/usr/bin/env python3
"""Emit the round-2 capture whitepaper, reusing the previous one's <style> verbatim.

The style block is lifted from the file rather than retyped so the two documents cannot drift apart
visually -- the same reasoning as reusing the capture kernel instead of writing a second one.
"""
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "docs" / "S1_OUTPUT_CAPTURE_20260916_1955.html"
DST = REPO / "docs" / "S1_CAPTURE_ROUND2_20260916_2035.html"

head = SRC.read_text(encoding="utf-8")
style = head[head.index("<style>"):head.index("</style>") + len("</style>")]

BODY = """
<h1>§9.18.6 第二輪：把「layer 2 之前或之內」切成一句可執行的事實</h1>
<p class="meta">CGC MoE Engine · 2026-09-16 20:03–20:30 · 承接 <code>docs/S1_OUTPUT_CAPTURE_20260916_1955.html</code>。
建置 <code>libggml-metal</code> <code>ec3ece90cf1c</code>（20:09:32）。</p>

<div class="box">
<strong>四句話：</strong>
(1) 擷取點從 <strong>1 個 dispatcher 擴到 5 個</strong>，過濾器從單名改成<strong>逗號分隔清單</strong>，
    所以一次跑就能走完一層的鏈（attention 輸入 → 兩個輸入投影 → 輸出投影 → router 輸入 → router logits → MoE）。
(2) 這一輪最重要的技術發現不在量測裡而在<strong>命名規則</strong>：可融合的 dispatcher 會把 <code>bid_dst</code>
    指向<strong>融合群的最後一個節點</strong>，所以尾端擷取必須命名 <code>node(idx + n_fuse - 1)</code>。
(3) 兩個獨立跨臂對、在兩個同臂對照都乾淨之後，逐項相同：<strong>layer 0 與 layer 1 的輸出逐位元相同；
    layer 2 的 attention 輸入、它的兩個輸入投影（<code>z-2</code>、<code>gate-2</code>）也都逐位元相同；
    但 layer 2 的 attention 輸出投影 <code>linear_attn_out-2</code> 自 graph 4 起不同。</strong>
(4) ⇒ <strong>第一個分歧在 layer 2 的 gated delta-net 核心</strong>（conv1d／delta-rule scan／state），
    不在任何 dense 矩陣裡，也不在 layer 2 之前。上一輪留下的兩個候選，一個被排除、一個被收窄。
</div>

<div class="toc">
<strong>目錄</strong>
<ol>
<li><a href="#ask">這一輪問的是什麼</a></li>
<li><a href="#ins">儀器：三個改動，其中一個是命名規則</a></li>
<li><a href="#ctl">先過三道守門，再讀任何結果</a></li>
<li><a href="#res">跨臂結果</a></li>
<li><a href="#where">定位：layer 2 的 gated delta-net 核心</a></li>
<li><a href="#refute">推翻了什麼</a></li>
<li><a href="#gates">現場閘門原文</a></li>
<li><a href="#limit">誠實邊界與下一步</a></li>
</ol>
</div>

<h2 id="ask">1. 這一輪問的是什麼</h2>

<p>上一輪（19:55 那份）用輸出擷取把 §9.18.4 推翻，並把分歧定位到
<strong>「layer 1 的 MoE 輸出之後、layer 2 的 router 之前」</strong>。那個區間裡有兩個候選：
layer 2 的 attention（q/k/v、rope、KV），以及 layer 1→2 之間的 norm／殘差。</p>

<p>這一輪把區間切開。做法是把擷取點沿著 layer 2 的鏈往前推：attention 的<strong>輸入</strong>、
它的兩個<strong>輸入投影</strong>、它的<strong>輸出投影</strong>、router 的<strong>輸入</strong>、
router 的 <strong>logits</strong>，全部進同一份名單、同一輪跑。</p>

<div class="box warn">
<strong>一個必須先講的模型事實（它改變「attention」是什麼）：</strong>
Qwen3.6-35B-A3B 是<strong>混合</strong>堆疊，<code>qwen35moe.full_attention_interval = 4</code>
⇒ <strong>layers 0,1,2 是 gated delta-net（線性注意力），layer 3 才是第一個 full attention</strong>。
所以分歧所在的 <strong>layer 2 根本沒有 flash attention</strong>：它的「attention 輸出投影」
<code>linear_attn_out-2</code> 是一個普通的 <code>mul_mat</code>。
「擴到 attention 就得掛上另一個 dispatcher」——對這一層而言，那個 dispatcher 是
<code>mul_mat</code>，不是 <code>flash_attn_ext</code>。兩個都掛了，這樣同一份名單走任何一種層都成立。
</div>

<h2 id="ins">2. 儀器：三個改動，其中一個是命名規則</h2>

<h3>2.1 過濾器成為清單</h3>
<p><code>CGC_TENSOR_CAPTURE</code> 現在吃<strong>逗號分隔的節點名清單</strong>（或 <code>*</code>），
每個 token 仍<strong>精確比對</strong>——不用子字串的理由沒變：<code>ffn_moe_down-1</code> 是
<code>ffn_moe_down-10..19</code> 的子字串，而比較器<strong>按名字配對</strong>。
槽位從 1024 提到 <strong>4096</strong>，而且<strong>用完會印一次 WARN</strong>
（原本是靜默截斷，也就是一個安靜的假陰性——這個專案最不想要的那種失效）。</p>

<h3>2.2 四個新的 dispatcher 尾端擷取點</h3>
<p>加上既有的 <code>mul_mat_id</code>，現在涵蓋 5 個：<code>mul_mat</code>、<code>mul_mat_id</code>、
<code>flash_attn_ext</code>、<code>bin</code>（殘差 add）、<code>norm</code>（rms_norm）。</p>

<h3>2.3 ★ 命名規則：<code>node(idx + n_fuse - 1)</code>，不是 <code>node(idx)</code></h3>
<p>這是這一輪真正的技術發現，也是「一行接一個 dispatcher」之所以安全的原因。可融合的 dispatcher
在結尾會把輸出緩衝<strong>重新指向融合群的最後一個節點</strong>：</p>
<pre><code>// ggml-metal-ops.cpp, ggml_metal_op_norm
if (n_fuse &gt; 1) {
    bid_dst = ggml_metal_get_buffer_id(ctx->node(idx + n_fuse - 1));
}
</code></pre>
<p>而不能融合的 dispatcher 以常數 <code>return 1;</code> 收尾（<code>mul_mat</code> 與
<code>flash_attn_ext</code> 都是）。所以同一個運算式在兩種情況下都對：<strong>讀到的 buffer 依構造就是
被命名的那個節點的輸出</strong>，不可能被融合的消費者騙到。若不是這樣，就只得去追每個 dispatcher 的每一條
內部路徑——<code>mul_mat</code> 一個就有八處 <code>set_buffer(..., bid_dst, ...)</code>，而且不是每條路徑都會走到。
<code>fuse</code> 也寫進列的尾端（放在 <code>ids=[...]</code> 之後，既有 reader 的正則不受影響）。</p>

<h3>2.4 先列舉，再量測</h3>
<p><code>CGC_TENSOR_CAPTURE='*'</code> 跑一次就把 <strong>1021 個節點名與它們的 fuse 值</strong>列出來。
<b>我先前從 builder 猜的名字有兩個根本不存在</b>：<code>ffn_out-*</code> 與 <code>post_moe-*</code>；
真名是 <code>ffn_moe_out-*</code> 與 <code>l_out-*</code>（<code>build_cvec</code> 把殘差重新命名）。
猜名字的代價是一整輪跑（四趟、約八分鐘），列舉只要一次。</p>
<div class="box">
<strong>附帶一個不能忘的規則：</strong> <code>node_NNN</code> 形式的<strong>匿名節點不可跨臂比對</strong>。
那些名字來自 ggml 的計數器，而兩臂的圖不同（S1 每層多 4 個節點）⇒ 同名不同物。
只有 builder 明確命名的節點能用。
</div>

<h2 id="ctl">3. 先過三道守門，再讀任何結果</h2>

<h3>3.1 同臂對照：乾淨（這是引用任何跨臂結果的前提）</h3>
<p>兩個臂各跑兩次、互相對照自己，在 graph 1..23 × 11 個節點上<strong>零差異</strong>。
<b>不限範圍時</b>會看到 <code>gate-2</code> 在 graph 27 出差異——那是下面的第 3.3 條造成的假訊號，不是儀器不穩。</p>

<h3>3.2 <code>ABSENT</code> 不是 <code>SAME</code></h3>
<p>一個節點只有<strong>剛好是融合群的最後一個節點</strong>時才會被擷取，而融合狀態取決於<strong>節點順序</strong>，
而順序不穩定（見第 3.4 條）。實際後果：<code>norm-2</code>（layer 2 delta-net 裡的 gated norm）
在 baseline 臂出現 <strong>41/41</strong> 次，在 S1 臂是 <strong>0/41</strong>。</p>
<div class="box bad">
所以分析器現在把 <code>ABSENT</code>（附 <code>present=N</code>）與 <code>never</code> <strong>分開印</strong>。
「一個名字在其中一份日誌裡不存在」永遠不可以被讀成「相同」——那是這個專案最怕的安靜假陰性。
</div>

<h3>3.3 尾段的 graph 邊界不可信</h3>
<p>ids 目的地上限 4096 而一個 forward pass 大約吃掉 114 列 ⇒ <strong>尾端幾個 chunk 根本沒有 graph 標記，
把好幾個 pass 併在一起</strong>，按名字配對就等於拿不同 pass 的列相比。這一輪一律用
<code>--upto 24</code>（只採 graph 1..23）。這也解釋了為什麼不限定時會在 graph 27／30／31 冒出假訊號。</p>

<h3>3.4 發射順序不是計算順序，而且不穩定</h3>
<p>編碼迴圈本身是嚴格遞增的（<code>for idx &lt; n_nodes: ggml_metal_op_encode(ctx, idx)</code>），
但<strong>同一個臂跑兩次，同樣的值以不同順序送出</strong>——<code>linear_attn_out-2</code> 有時在
<code>attn_residual-2</code> 之前、有時在之後，同一份日誌的不同 graph 就不一樣。
所以任何「哪一列先出現＝哪個節點先算」的推論都是讀雜訊。分析器改成用<strong>原始碼的層鏈順序</strong>排序，
而不是流的順序。</p>

<h2 id="res">4. 跨臂結果</h2>

<p>兩個獨立跨臂對（<code>p25-gputime</code> vs <code>p25-slotgpu</code>），只採 graph 1..23，
結果<strong>逐項相同</strong>：</p>

<table>
<tr><th>鏈序</th><th>節點</th><th>是什麼</th><th>run 1</th><th>run 2</th></tr>
<tr><td>1</td><td><code>l_out-0</code></td><td>layer 0 輸出</td><td>SAME</td><td>SAME</td></tr>
<tr><td>2</td><td><code>l_out-1</code></td><td>layer 1 輸出</td><td>SAME</td><td>SAME</td></tr>
<tr><td>3</td><td><code>attn_norm-2</code></td><td>layer 2 attention 的<strong>輸入</strong></td><td>SAME</td><td>SAME</td></tr>
<tr><td>4</td><td><code>z-2</code></td><td>layer 2 的 z 投影（<code>mul_mat</code>）</td><td>SAME</td><td>SAME</td></tr>
<tr><td>5</td><td><code>gate-2</code></td><td>layer 2 的 alpha/gate 投影</td><td>SAME</td><td>SAME</td></tr>
<tr><td>6</td><td><code>norm-2</code></td><td>gated norm（核心之後）</td><td colspan="2"><strong>ABSENT</strong>（跨臂不可比）</td></tr>
<tr><td>7</td><td><code>linear_attn_out-2</code></td><td>layer 2 attention 的<strong>輸出投影</strong></td><td><strong>DIFF@4</strong></td><td><strong>DIFF@4</strong></td></tr>
<tr><td>8</td><td><code>attn_residual-2</code></td><td>殘差</td><td>DIFF@4</td><td>DIFF@4</td></tr>
<tr><td>9</td><td><code>attn_post_norm-2</code></td><td>router 的輸入</td><td>DIFF@4</td><td>DIFF@4</td></tr>
<tr><td>10</td><td><code>ffn_moe_logits_raw-2</code></td><td>router logits（top-k 之前）</td><td>DIFF@4</td><td>DIFF@4</td></tr>
<tr><td>11</td><td><code>l_out-2</code></td><td>layer 2 輸出</td><td>DIFF@4</td><td>DIFF@4</td></tr>
</table>

<pre><code>-- graph 4: l_out-0.dst=SAME  l_out-1.dst=SAME  attn_norm-2.dst=SAME  z-2.dst=SAME  gate-2.dst=SAME
            norm-2.dst=MISSING  linear_attn_out-2.dst=DIFF  attn_residual-2.dst=DIFF
            attn_post_norm-2.dst=DIFF  ffn_moe_logits_raw-2.dst=DIFF  l_out-2.dst=DIFF</code></pre>

<p>第 8–11 項的 DIFF <strong>不攜帶資訊</strong>：它們全在 <code>linear_attn_out-2</code> 的下游，
已被第 7 項解釋。判讀的關鍵是<strong>第 7 項之前全部相同</strong>。</p>

<h2 id="where">5. 定位：layer 2 的 gated delta-net 核心</h2>

<p>layer 2 的鏈（<code>src/llama.cpp/src/models/qwen35moe.cpp</code>）：</p>
<pre><code>attn_norm-2 ──┬─► z-2        (build_qkvz: wqkv_gate)
              ├─► gate-2     (ssm_alpha → softplus → · ssm_a)
              └─► qkv ─► conv1d ─► delta-rule scan ─► gated norm ─► linear_attn_out-2 (ssm_out)</code></pre>

<p>量到的形狀是：<strong>箭頭左邊全部逐位元相同</strong>（輸入、<code>z-2</code>、<code>gate-2</code>），
<strong>箭頭右邊的結果不同</strong>（<code>linear_attn_out-2</code>）。</p>

<div class="box ok">
<strong>⇒ 第一個分歧在 layer 2 的 gated delta-net 核心：conv1d／delta-rule scan／state 那一段。</strong>
layer 2 的<strong>密集矩陣沒有一個是分歧點</strong>（輸入、z、gate 三個都同），
而 layer 0 與 layer 1 的輸出也都同。
</div>

<p>這是一句可執行的事實：路徑上還缺的 dispatcher 只有兩個——<code>ggml_metal_op_ssm_conv</code> 與
<code>ggml_metal_op_gated_delta_net</code>（<code>gate</code>／<code>z</code> 之後、<code>norm-2</code> 之前）。</p>

<h2 id="refute">6. 推翻了什麼</h2>

<ul>
<li><strong>上一輪留下的候選「layer 1→2 之間的 norm／殘差」被排除。</strong>
    <code>l_out-1</code>（layer 1 的輸出）逐位元相同，
    而它的規格化結果 <code>attn_norm-2</code> 也逐位元相同。</li>
<li><strong>「layer 2 的 attention 路徑（q/k/v、rope、KV）」被收窄成「它的遞迴核心」。</strong>
    這一層的 qkv／z／gate 投影都是普通 <code>mul_mat</code>，而它們都相同。</li>
<li><strong>§9.18.4 的「載體是池／slot 的權重內容」在 layer 2 的 attention 上的變體一併排除</strong>：
    該層所有 dense matmul 的輸入與輸出都相同（除了被核心汙染的輸出投影）。</li>
<li><strong>不要過度外推</strong>：這只說「<strong>第一個</strong>分歧在 layer 2 的 delta-net 核心」。
    layer 2 的 MoE 在那之後才分歧（<code>ffn_moe_logits_raw-2</code> DIFF@4），已被解釋，不構成證據。
    池在其他地方有沒有問題，這份文件沒有回答。</li>
</ul>

<h2 id="gates">7. 現場閘門原文</h2>

<pre><code>D5 (--tag capround2)          comparable=True  config_diffs=[]  ref=...ref_..._v5_spac.jsonl
                              GATE capround2: PASS  M1(bit-identical)=9/9  M2(argmax)=9/9  M3(topk)=9/9  n=9
build products                libggml-metal.0.19.0.dylib  2026-09-16 20:09:32  (955288 B)
                              llama-server                2026-09-16 20:09:33  (gate ran after -> attributable)
build fingerprint             8 keys: server=054fb22f04a0  libggml-metal=ec3ece90cf1c  libggml-base=d2099fb798e7
                                     libllama=233a172afb2e  libllama-common=7bceb3bd5320  libggml=8f4abce76052
                                     libggml-cpu=70ebbd7c709a  libggml-blas=52de23d5c405

validate.py                   122 episodes / 38 decisions / 116 lessons  -> OK
selftest.py                   10/10 injected violations rejected
index_assets.py --check       manifest OK: 76 assets, existence + bytes + mtime all agree
build_memory_index.py --check memory index OK: 3 file(s), 67 section(s), no drift
thermal_pressure.py --selftest 15/15 checks passed

instrument (both arms)        CGC-IDS-CAP enabled: slots=4096 stride=8 bytes=131072 shared=1
                              CGC-TENSOR-CAP enabled: node=l_out-0,l_out-1,attn_norm-2,z-2,gate-2,norm-2,
                                linear_attn_out-2,attn_residual-2,attn_post_norm-2,ffn_moe_logits_raw-2,l_out-2
                                words=32 slots=4096 stride=32 bytes=524288
dst rows per arm              G1 451 (11 x 41)   S1 410 (norm-2 缺席 41)   -- ABSENT 是可見的，不是靜默的
thermal at launch             MODERATE(1)  -- 這是 bit-identity 比較，不依賴等級；但仍記錄</code></pre>

<h2 id="limit">8. 誠實邊界與下一步</h2>

<ul>
<li><strong>沒有說池是乾淨的。</strong>只說第一個分歧不在池、不在任何 matmul、不在 layer 2 之前。</li>
<li><strong><code>norm-2</code> 跨臂不可比</strong>（S1 臂整個缺席），所以「核心的哪一半」還沒被切開。
    要切開得先修融合造成的缺席：把「融合群末節點匿名」的情況改成用<strong>群首名字 ＋
    <code>+fuseN</code> 的合成名</strong>，這樣兩臂都看得到。</li>
<li><strong>只採 graph 1..23。</strong>尾段的 chunk 邊界不可信（ids 上限 4096 造成的併接），
    在那之後的任何圖形差異都不該引用。</li>
<li><strong>「同臂對照乾淨」是在這台機器、這份 build、這個 prompt 上量到的。</strong>
    換 build 或換 run 形狀要重量；對照很便宜（兩趟、約四分鐘），沒有理由省。</li>
<li><strong>沒有做「reverse」實驗</strong>：把擷取點換成 layer 3（第一個 full attention 層）可以測
    full-attention 路徑是否也有同一型缺陷，這一輪沒做。</li>
</ul>

<hr>
<p class="small">相關：<code>docs/REMAP_ROUNDTRIP_REMOVAL_PLAN_2026-09-15.md</code> §9.18（§9.18.4 已被
<code>docs/S1_OUTPUT_CAPTURE_20260916_1955.html</code> 推翻，本文件收窄它的殘餘候選）、
<code>Backup/analyze_capture_nodes.py</code>（逐節點定位，含 <code>ABSENT</code> 與 <code>--upto</code>）、
<code>Backup/run_ids_dst_capture.sh</code>（驅動）、
<code>scripts/check/ids_capture_diff.py</code>（09-15 的 ids 比較器，未改動）。
lessons：<code>eng-diag-0027</code>、<code>eng-src-0015</code>、<code>eng-mh-0043</code>、<code>eng-mh-0044</code>。</p>

</div>
</body>
</html>
"""

DST.write_text(
    "<!DOCTYPE html>\n<html lang=\"zh-Hant\">\n<head>\n<meta charset=\"utf-8\">\n"
    "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
    "<title>§9.18.6 第二輪：把「layer 2 之前或之內」切成一句可執行的事實（2026-09-16）</title>\n"
    + style
    + "\n</head>\n<body>\n<div class=\"wrap\">\n"
    + BODY.lstrip("\n"),
    encoding="utf-8",
)
print("wrote", DST.relative_to(REPO), DST.stat().st_size, "bytes")
