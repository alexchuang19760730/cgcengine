#!/usr/bin/env python3
"""Materialise docs/UNIFIED_PROFILE_SPAC_20260916_1955.html from the draft, filling the four
placeholders with what was actually measured (and saying plainly what was NOT)."""
import re
from pathlib import Path

src = Path("Backup/wp_unified_draft.html").read_text(encoding="utf-8")

PREFILL = """
<p><strong>這一節的結論是「尚未定案」，而理由比「雜訊大」具體。</strong>協定跑成了，但被自己的閘門否決。</p>

<h3>5.1 v1（單次讀到 <code>0</code> 就發射）作廢——自變數不是 SPAC</h3>

<table>
<tr><th>臂</th><th>發射前靜置</th><th>req1</th><th>req2</th><th>req3</th><th>三請求均</th></tr>
<tr><td>r1_off（SPAC=0）</td><td>0 s（箱子先前長時間閒置）</td><td>263.03</td><td>264.68</td><td>247.93</td><td>258.5</td></tr>
<tr><td>r1_on（SPAC=1）</td><td><strong>40 s</strong></td><td>232.29</td><td>213.70</td><td>203.03</td><td>216.3</td></tr>
<tr><td>r2_off（SPAC=0）</td><td><strong>40 s</strong></td><td>258.95</td><td>258.10</td><td>220.19</td><td>245.7</td></tr>
<tr><td>r2_on（SPAC=1）</td><td><strong>105 s</strong></td><td>264.87</td><td>273.21</td><td>248.66</td><td><strong>262.2</strong></td></tr>
</table>

<p>SPAC 的兩輪<strong>符號相反</strong>（r1 是 −16%、r2 是 +7%），而<strong>兩個發射前只靜置 40 s 的臂正是最慢的兩個</strong>，
靜置 105 s 的那臂最快 ⇒ 量到的是「箱子安靜了多久」，不是 SPAC。
<span class="small">證據保留在 <code>Backup/cgc_logs/spac_prefill_ab/v1_read0once/</code>，不得當 SPAC 結果引用。</span></p>

<h3>5.2 閘門自己否決了「發射前讀到 0 就夠」</h3>

<p>把靜置改成「連續 120 s 讀到 0 才放行」後，閘門吐出一行先前沒出現過的分類（<code>run_req2_retest.sh:327-350</code>）：</p>

<pre><code>[thermal label] COLD-STATE   quiet since previous arm ended &gt;= 1800 s  -&gt; 交付級冷樣本
[thermal label] HOT-STATE    否則                                     -&gt; "do NOT quote it as the delivery number"
    (measured: the same binary/config/fingerprint gave 254.29/282.38/265.27 COLD and 167.41-200.58 HOT)</code></pre>

<div class="box bad">
<strong>這是比 SPAC 更重要的發現：</strong>本專案自己定義的「可引用的 prefill 數字」需要 <strong>1800 s</strong> 的安靜，
不是「發射前一刻讀到 0」。v2 的 r1_off 的 req2/req3（179.64／169.57）<strong>正落在 HOT 帶裡</strong>，
所以那個臂本身就不是可引用的 artifact 數字。
</div>

<h3>5.3 v2 跑到一半被中止，因為順序錯了</h3>

<table>
<tr><th>臂</th><th>發射前靜置</th><th>req1</th><th>req2</th><th>req3</th></tr>
<tr><td>r1_off（SPAC=0）</td><td>120 s</td><td>265.18</td><td>179.64</td><td>169.57</td></tr>
<tr><td>r1_on（SPAC=1）</td><td>~195 s</td><td>279.86</td><td>285.88</td><td>250.02</td></tr>
</table>

<p>兩個理由停下來：(1) 箱子今晚被反覆加熱，<code>wait_quiet</code> 常常 150 s 以上仍 HEAVY，而
<strong>depth 與 alpha 是吞吐量測</strong>（對熱敏感）⇒ 在這麼熱的箱子上跑它們只會得到低品質資料；
(2) 同一晚的 §9.18.6 是 <strong>bit-identity 比較</strong>（對熱不敏感）⇒ 先做它。
剩餘階段等箱子冷卻後再跑。</p>

<p><strong>而協定本身還有兩個不對稱必須修</strong>：靜置是「≥QUIET_SEC」的<u>下限</u>，實際值取決於前一臂留下的熱
（r1_off 得 120 s、r1_on 得 195 s）⇒ <strong>同一輪的兩臂沒有配對在相同的靜置上</strong>，
所以連 warm 層級的比較都還不成立。</p>

<div class="box warn">
<strong>要回答「統一有沒有弄壞交付」，只需一條臂</strong>：新 profile 的一條 <strong>COLD</strong> 臂
（≥1800 s 靜置），req1–req3 全 ≥250 即通過。SPAC on/off 的<strong>比較</strong>是第二個問題（需要一對 COLD 臂），
不要跟交付認證混在一起問。
</div>
"""

DEFERRED_DEPTH = """
<div class="box warn">
<strong>這一節尚未產出資料，而且是刻意延後的。</strong>驅動已寫好（<code>Backup/run_unified_depth_matrix.sh</code>，
含 <code>wait_quiet()</code> 與 <code>SOLO=0</code> 開關），但第一輪執行時它<strong>根本沒跑</strong>——
<code>run()</code> 收 3 個參數而 <code>run asc "0,512,..."</code> 只給 2 個，在 <code>set -u</code> 之下
<code>$3: unbound variable</code> 直接終止，而整條鏈用 <code>&amp;&amp;</code> 串接 ⇒ 後面兩個階段連一行都沒執行。
已修（label 改 optional）。
<span class="small">可移植的教訓：用 <code>&amp;&amp;</code> 串階段時，前一階段的靜默失敗會讓後面的階段連一行都不執行，
而「沒有產物」與「跑了但沒結果」在這種鏈裡同形。每階段結尾要印哨兵行，或不要用 <code>&amp;&amp;</code>。</span>
</div>
"""

DEFERRED_ALPHA = """
<div class="box warn">
<strong>這一節尚未產出資料。</strong>驅動已寫好（<code>Backup/run_spac_alpha_sweep.sh</code>：SPAC off 對照 ＋
alpha 0.5／0.75／0.9，depth 512 與 1024，ctx 8192）。與上一節同因延後——<strong>它是吞吐量測，需要一個冷箱</strong>。
在跑之前，<code>alpha=0.75</code> 仍然只是「在 ctx 4096 調出來的值」，ctx 8192 上的正確性未經量測。
</div>
"""

GATES = """
<pre><code>D5 (--tag spac_unified)      INVALID COMPARISON (2 diffs) -- expected, see section 3
                             ENV.CGC_SPAC:       ref='&lt;absent&gt;' now='1'
                             ENV.CGC_SPAC_ALPHA: ref='&lt;absent&gt;' now='0.75'
                             observed M1=9/9 M2=9/9 M3=9/9 n=9
D5 (--tag spac_v5)           INVALID COMPARISON (2 diffs) -- v5 dump vs the v4 default ref
  --write-ref ..._v5_spac.jsonl   wrote baseline + .cap
  md5 v5 == md5 v4           a0a0ca742ca94e843c54b39981742738   (byte-identical rows)
  v5 .cap resolved.ENV       CGC_SPAC=1  CGC_SPAC_ALPHA=0.75   (v4: absent)
D5 (--tag spac_unified_v5)   GATE spac_unified_v5: PASS  M1(bit-identical)=9/9  M2(argmax)=9/9  M3(topk)=9/9  n=9
                             comparable=True  config_diffs=[]

validate.py                  122 episodes / 38 decisions / 111 lessons  -&gt; OK
index_assets.py --check      manifest OK: 76 assets, existence + bytes + mtime all agree
build_memory_index.py --check memory index OK: 3 file(s), 65 section(s), no drift
thermal_pressure.py --selftest 15/15 checks passed</code></pre>
"""

out = src
out = out.replace("<!--PREFILL_AB-->", PREFILL)
out = out.replace("<!--DEPTH_MATRIX-->", DEFERRED_DEPTH)
out = out.replace("<!--ALPHA_SWEEP-->", DEFERRED_ALPHA)
out = out.replace("<!--GATES-->", GATES)
out = out.replace(
    "<title>統一 profile：prefill250 + CGC_SPAC=1 落地與 prefill 代價（2026-09-16）</title>",
    "<title>統一 profile：prefill250 + CGC_SPAC=1 落地，與尚未定案的 prefill 代價（2026-09-16）</title>",
)
# markdown bold -> <strong>, skipping <pre> blocks
parts = re.split(r"(<pre>.*?</pre>)", out, flags=re.S)
for i, part in enumerate(parts):
    if not part.startswith("<pre>"):
        parts[i] = re.sub(r"\*\*([^*\n]+)\*\*", r"<strong>\1</strong>", part)
out = "".join(parts)

dst = Path("docs/UNIFIED_PROFILE_SPAC_20260916_1955.html")
dst.write_text(out, encoding="utf-8")
print("wrote", dst, dst.stat().st_size, "bytes")
print("placeholders left:", [m for m in re.findall(r"<!--[A-Z_]+-->", out)])
