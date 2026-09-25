// mindmap 點擊回歸測試：node scripts/check/mindmap_click_smoke.mjs [index.html]
// 需要 jsdom；沒裝就 SKIP（exit 0），不阻塞 CI。
import fs from 'fs';
import path from 'path';
// ESM 不吃 NODE_PATH，所以用 JSDOM_PATH 指到 jsdom 的實作檔（隔離 workspace 安裝時必要）
const JSDOM_CANDIDATES = ['jsdom', process.env.JSDOM_PATH].filter(Boolean);
let JSDOM;
for (const cand of JSDOM_CANDIDATES) {
  try { ({ JSDOM } = await import(cand)); break; } catch { /* 下一個候選 */ }
}
if (!JSDOM) {
  console.log('SKIP：未安裝 jsdom（npm i jsdom 進 workspace，並設 JSDOM_PATH='
    + '<workspace>/node_modules/jsdom/lib/api.js）');
  process.exit(0);
}

const target = process.argv[2] || path.resolve('docs/mindmap/index.html');
const html = fs.readFileSync(target, 'utf8');
const dom = new JSDOM(html, { runScripts: 'dangerously', url: 'file:///docs/mindmap/index.html' });
const { window } = dom;
const doc = window.document;
const errs = [];
window.addEventListener('error', e => errs.push(String(e.error)));

const D = id => doc.getElementById(id);
const txt = () => (D('detail').textContent || '').replace(/\s+/g, ' ');
let pass = true;
const ok = (n, c) => { if (!c) pass = false; console.log('  [' + (c ? 'PASS' : 'FAIL') + '] ' + n); };
const click = el => el.dispatchEvent(new window.MouseEvent('click', { bubbles: true }));

// 0 初始：總覽
ok('初始面板＝總覽（含矩陣）', txt().includes('250 / 25 攻關總覽') && txt().includes('階段 / 子目標'));
ok('初始渲染無 JS 錯誤', errs.length === 0);
const NCHIP = doc.querySelectorAll('#subLegend .chip').length;
ok('子目標 chip 數 = 5（S／M／both／C／na）', NCHIP === 5);
ok('每個子目標 chip 都有 ⊘ 過濾紐', doc.querySelectorAll('#subLegend .chip .x').length === 5);
ok('有 C kernel／頻寬效率 chip',
   [...doc.querySelectorAll('#subLegend .chip')].some(c => c.textContent.includes('C kernel')));
// 狀態橫幅：兩軸未交付 ／ 耦合 ／ C 定位
const banner = doc.querySelector('.banner');
ok('頁首有狀態橫幅', !!banner);
const bt = banner ? banner.textContent.replace(/\s+/g, ' ') : '';
ok('   橫幅含 C 為天花板軸', bt.includes('天花板軸'));
ok('   橫幅含兩軸皆未交付', bt.includes('未真正交付'));
ok('   橫幅含 S／M 不可相加', bt.includes('不可相加'));

// 1 點「S 序列化消減」標籤
const chips = [...doc.querySelectorAll('#subLegend .chip')];
const sChip = chips.find(c => c.textContent.includes('S 序列化消減'));
click(sChip.querySelector('.t'));
const t1 = txt();
ok('① 點 S 標籤 → 面板換成 S 說明', t1.includes('S 序列化消減') && !t1.includes('250 / 25 攻關總覽'));
for (const k of ['1 現狀', '2 推論', '3 量測', '4 目標']) ok('   S 面板含「' + k + '」', t1.includes(k));
ok('   S 面板含原始報告連結', !!D('detail').querySelector('a[href*="S1_TPOT_DECOMPOSITION"]'));
const ns = (t1.match(/拆解的子目標狀況（(\d+) 條目）/) || [])[1];
ok('   S 面板含拆解狀況（' + ns + ' 條目）', !!ns);
ok('   S 面板列出的條目數＝宣告數',
   Number(ns) === D('detail').querySelectorAll('.gi[data-e]').length);
ok('   S chip 高亮 active', sChip.classList.contains('active'));

// 2 內嵌檢視
click(D('emb'));
ok('② 內嵌檢視 → 生出 iframe', !!D('detail').querySelector('iframe'));
ok('   iframe src 指向 S 報告', (D('detail').querySelector('iframe').getAttribute('src') || '').includes('S1_TPOT_DECOMPOSITION'));
click(D('emb'));
ok('   再點一次 → iframe 收起', !D('detail').querySelector('iframe'));

// 3 點「M MTP on 加速」
const mChip = chips.find(c => c.textContent.includes('M MTP on 加速'));
click(mChip.querySelector('.t'));
const t3 = txt();
ok('③ 點 M 標籤 → 面板換成 M 說明', t3.includes('M MTP on 加速') && t3.includes('MTP OFF'));
for (const k of ['1 現狀', '2 推論', '3 量測', '4 目標']) ok('   M 面板含「' + k + '」', t3.includes(k));
ok('   M 面板含原始報告連結', !!D('detail').querySelector('a[href*="mtp_amortization"]'));
const nm = (t3.match(/拆解的子目標狀況（(\d+) 條目）/) || [])[1];
ok('   M 面板含拆解狀況（' + nm + ' 條目）', !!nm);
ok('   M chip 高亮、S chip 解除', mChip.classList.contains('active') && !sChip.classList.contains('active'));
ok('   M 面板含攤薄公式 S=(1+a·k)/(1+m·k)', t3.includes('(1 + a·k) / (1 + m·k)'));

// 3b 點「C kernel／頻寬效率」——第三軸（天花板軸）
const cChip = chips.find(c => c.textContent.includes('C kernel'));
click(cChip.querySelector('.t'));
const tc = txt();
ok('③b 點 C 標籤 → 面板換成 C 說明', tc.includes('C kernel') && !tc.includes('250 / 25 攻關總覽'));
for (const k of ['1 現狀', '2 推論', '3 量測', '4 目標']) ok('   C 面板含「' + k + '」', tc.includes(k));
ok('   C 面板標注「天花板軸」性質', tc.includes('天花板軸'));
ok('   C 面板含原始報告連結', !!D('detail').querySelector('a[href*="DEVICE_BUSY_ATTRIBUTED"]'));
const nc = (tc.match(/拆解的子目標狀況（(\d+) 條目）/) || [])[1];
ok('   C 面板含拆解狀況（' + nc + ' 條目）', !!nc);
ok('   C 面板提到 device span 是最大一塊', tc.includes('56–61%') || tc.includes('裝置忙碌時間'));
ok('   C 面板明說不可引用的數字', tc.includes('gpu_sum'));

// 4 點拆解裡的條目 → 單條細節
const gi = D('detail').querySelector('.gi[data-e]');
const eid = gi.getAttribute('data-e');
click(gi);
const t4 = txt();
ok('④ 點拆解條目（' + eid + '）→ 單條細節', t4.includes('判準') && t4.includes('依據'));

// 5 點 stage 節點
click(doc.querySelector('.node[data-stage]'));
const t5 = txt();
ok('⑤ 點階段節點 → 階段面板', t5.includes('生產') || t5.includes('實驗階段') || t5.includes('已結案'));
ok('   階段面板含「看說明 ›」', t5.includes('看說明'));

// 6 點根節點 → 回總覽
click(doc.querySelector('.node[data-root]'));
ok('⑥ 點根節點 → 回總覽', txt().includes('250 / 25 攻關總覽'));

// 7 ⊘ 過濾仍可用（且不會把面板洗掉）
const before = doc.querySelectorAll('.node[data-id]').length;
click(sChip.querySelector('.x'));
const after = doc.querySelectorAll('.node[data-id]').length;
ok('⑦ ⊘ 過濾生效（條目 ' + before + ' → ' + after + '）', after < before);
ok('   ⊘ 不會誤開說明面板', txt().includes('250 / 25 攻關總覽'));

// 8 單條細節裡的技術白皮書連結與內嵌
const entNode = [...doc.querySelectorAll('.node[data-id]')].find(g => g.getAttribute('data-id'));
click(entNode);
const wantId = entNode.getAttribute('data-id');
const wp = D('detail').querySelector('a.wp');
ok('⑧ 單條細節有「技術白皮書」連結', !!wp);
ok('   連結指向 briefs/' + wantId + '.html',
   !!wp && wp.getAttribute('href') === 'briefs/' + wantId + '.html');
ok('   連結另開視窗', !!wp && wp.getAttribute('target') === '_blank');
const wpmd = [...D('detail').querySelectorAll('a.wp')]
  .find(a => (a.getAttribute('href') || '').endsWith('.md'));
ok('   單條細節有 MD 版白皮書連結', !!wpmd);
ok('   MD 連結指向 briefs/' + wantId + '.md',
   !!wpmd && wpmd.getAttribute('href') === 'briefs/' + wantId + '.md');
ok('   單條細節含「目標」欄', txt().includes('目標'));
ok('   單條細節含「判準／結果／依據」',
   txt().includes('判準') && txt().includes('結果') && txt().includes('依據'));
const emb2 = D('wpemb');
ok('   有「內嵌」按鈕', !!emb2);
click(emb2);
const ifr = D('detail').querySelector('iframe');
ok('   內嵌 → iframe src = briefs/' + wantId + '.html',
   !!ifr && ifr.getAttribute('src') === 'briefs/' + wantId + '.html');
click(emb2);
ok('   再點 → iframe 收起', !D('detail').querySelector('iframe'));

// 9 頁首有白皮書總目錄入口
const hdr = doc.querySelector('.hdr a');
ok('⑨ 頁首有白皮書總目錄連結', !!hdr && hdr.getAttribute('href') === 'briefs/index.html');

// 10 對應報告是可點連結
const docLinks = [...D('detail').querySelectorAll('a[href^="../"]')];
ok('⑩ 對應報告為可點連結（' + docLinks.length + ' 個）', docLinks.length > 0);

ok('全程無 JS 錯誤', errs.length === 0);
if (errs.length) console.log(errs.slice(0, 3));
console.log(pass ? '\nSMOKE OK' : '\nSMOKE FAIL');
process.exit(pass ? 0 : 1);
