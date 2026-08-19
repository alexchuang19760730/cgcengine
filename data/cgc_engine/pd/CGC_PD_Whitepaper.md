# TurboFieldfare CGC PD 分離技術白皮書

> **版本**: v1.0 MVP (單層 hidden state)
> **日期**: 2026-08-12
> **適用機型**: Mac A (Gemma4 26B-A4B) + Mac B (Qwen3.6 35B-A3B)
> **目的**: 提供 Mac 端 TurboFieldfare 完整改動清單, 用於同步更新兩台 Mac 的源碼

---

## 目錄

1. [背景與目標](#1-背景與目標)
2. [系統架構](#2-系統架構)
3. [核心協議設計](#3-核心協議設計)
4. [Swift 端改動總覽](#4-swift-端改動總覽)
5. [改動 1: CGCEmitTypes.swift (新檔)](#5-改動-1-cgcemittypesswift-新檔)
6. [改動 2: RealForwardRunner.swift (Gemma4)](#6-改動-2-realforwardrunnerswift-gemma4)
7. [改動 3: Qwen36ForwardRunner.swift (Qwen3.6)](#7-改動-3-qwen36forwardrunnerswift-qwen36)
8. [改動 4: ServerInference.swift (兩版共用)](#8-改動-4-serverinferenceswift-兩版共用)
9. [改動 5: HTTPServer.swift (兩版共用)](#9-改動-5-httpserverswift-兩版共用)
10. [編譯與部署](#10-編譯與部署)
11. [驗證流程](#11-驗證流程)
12. [已知限制與 v2 路線圖](#12-已知限制與-v2-路線圖)

---

## 1. 背景與目標

### 1.1 什麼是 PD 分離

PD 分離 (Prefill-Decode Disaggregation) 是 CGC (Cloud-Grid Collaboration) 的核心模式:

- **P (Prefill)**: 雲側 (Mac) 負責處理 prompt 的 prefill 階段, 產生 hidden state
- **D (Decode)**: 端側 (Windows) 負責後續 token-by-token 的 decode 階段
- **橋接**: MoT-h (Mixture-of-Translators, hidden variant) 將 Mac 的 hidden state 翻譯成 Windows 端模型可用的格式

### 1.2 為什麼需要 PD 分離

| 場景 | 痛點 | PD 解決 |
|------|------|---------|
| 長 prompt | 端側 prefill 慢 (4096 tokens 要 30s+) | 雲側 M3 Max prefill 只要 2-3s |
| 多模型 | 端側只能跑一個小模型 | 雲側大模型 prefill, 端側小模型 decode |
| 隱私 | prompt 不能上雲 | 只傳 hidden state, 不傳原文 (可选) |

### 1.3 本次目標

支援 **Gemma4 26B-A4B** 和 **Qwen3.6 35B-A3B** 兩個模型分別在兩台 Mac 上跑 prefill, 把完整序列的末層 hidden state 通過 HTTP 接口推送到 Windows 機器, 供 MoT-h 訓練和端側 decode 使用。

---

## 2. 系統架構

```
        ┌──────────────────────────────────────┐
        │  Windows (端側, 192.168.101.8)        │
        │  ┌────────────────────────────────┐  │
        │  │ coordinator.py (FastAPI, :9000)│  │
        │  │   POST /v1/cgc/ingest          │  │
        │  ├────────────────────────────────┤  │
        │  │ verify_pd_emit.py /            │  │
        │  │ collect_batch.py (client)      │  │
        │  ├────────────────────────────────┤  │
        │  │ train_mot_h.py (MoT-h 訓練)    │  │
        │  └────────────────────────────────┘  │
        └──────────┬───────────────┬───────────┘
                   │               │
            POST   │               │  POST
       /v1/cgc/emit           /v1/cgc/emit
                   │               │
        ┌──────────▼────┐  ┌──────▼────────┐
        │   Mac A       │  │   Mac B       │
        │   Gemma4      │  │   Qwen3.6     │
        │   26B-A4B     │  │   35B-A3B     │
        │   :8080       │  │   :8080       │
        │   (Gemma4 repo)│  │ (prime repo)  │
        └───────────────┘  └───────────────┘
```

### 2.1 數據流

```
1. Windows 向 Mac 發送 prompt
   POST /v1/cgc/emit  { prompt, request_id, max_seq_len }

2. Mac 端執行:
   tokenize → cgcEnableAccumulation → prefillChunked → cgcCopyAccumulatedHidden

3. Mac 返回:
   { success, hidden_state_b64 (base64 float32 LE), seq_len, hidden_dim, model_id, ... }

4. Windows 解碼:
   base64 → torch.frombuffer → Tensor[seq_len, hidden_dim]

5. 訓練 / 推理:
   - 訓練: 配對 (h_gemma4, h_qwen36) → MoT-h 學習 2816→2048 翻譯
   - 推理: Mac A prefill → MoT-h 翻譯 → Windows Qwen3.6 decode
```

---

## 3. 核心協議設計

### 3.1 HTTP 端點

```
POST /v1/cgc/emit
Content-Type: application/json

Request:
{
  "prompt": "Hello, world",
  "request_id": "uuid-1234",
  "max_seq_len": 4096
}

Response (200):
{
  "success": true,
  "error": null,
  "hidden_state_b64": "AAAAA...",       // base64(float32 LE [seq_len * hidden_dim])
  "seq_len": 12,
  "hidden_dim": 2816,                    // Gemma4
  "finished_layer": 30,                  // = num_layers
  "model_id": "gemma4-26b-a4b",
  "request_id": "uuid-1234",
  "prefill_ms": 234.5
}

Response (error):
{
  "success": false,
  "error": "prompt 10000 tokens exceeds maxContext 8192",
  "model_id": "gemma4-26b-a4b",
  "request_id": "uuid-1234"
}
```

### 3.2 hidden_state_b64 編碼規範

- **數據類型**: float32 (32-bit IEEE 754)
- **字節序**: little-endian (Apple Silicon 原生)
- **佈局**: row-major `[seq_len, hidden_dim]`
- **編碼**: base64 (二進制安全傳輸)
- **大小**: `seq_len * hidden_dim * 4 bytes`
  - Gemma4: seq_len * 2816 * 4
  - Qwen3.6: seq_len * 2048 * 4

Python 解碼:
```python
import base64, torch
raw = base64.b64decode(resp["hidden_state_b64"])
hidden = torch.frombuffer(raw, dtype=torch.float32) \
    .reshape(resp["seq_len"], resp["hidden_dim"])
```

### 3.3 模型識別

通過 `hidden_dim` 自動識別:

| hidden_dim | model_id |
|------------|----------|
| 2816 | gemma4-26b-a4b |
| 2048 | qwen36-35b-a3b |

---

## 4. Swift 端改動總覽

### 4.1 兩個 repo 對應關係

| 模型 | Repo 路徑 (Mac 上) | Forward Runner |
|------|-------------------|---------------|
| Gemma4 26B-A4B | `turbo-fieldfare-github-official/` | `RealForwardRunner.swift` |
| Qwen3.6 35B-A3B | `prime-agent-worktrees/turbo-fieldfare/` | `Qwen36ForwardRunner.swift` (專用) + `RealForwardRunner.swift` (備用) |

### 4.2 改動清單

每個 repo 都要做以下 5 處改動:

| # | 檔案 | 動作 | 說明 |
|---|------|------|------|
| 1 | `Sources/TurboFieldfareServer/Core/CGCEmitTypes.swift` | 新建 | 協議定義 (Request/Response/CGCEmitCapable) |
| 2 | `Sources/TurboFieldfare/Runtime/Inference/RealForwardRunner.swift` | 修改 | 加 hidden state 累積邏輯 |
| 3 | `Sources/TurboFieldfare/Runtime/Inference/Qwen36ForwardRunner.swift` | 修改 | (僅 Qwen3.6 repo) 同上, 但用 chunkHidden |
| 4 | `Sources/TurboFieldfareServer/Core/ServerInference.swift` | 修改 | 實作 `CGCEmitCapable` 協議 |
| 5 | `Sources/TurboFieldfareServer/Core/HTTPServer.swift` | 修改 | 加 `/v1/cgc/emit` 路由 |

---

## 5. 改動 1: CGCEmitTypes.swift (新檔)

**路徑**: `Sources/TurboFieldfareServer/Core/CGCEmitTypes.swift`
**兩個 repo 都要建立, 內容完全相同**。

### 5.1 設計理念

- 用 `Codable, Sendable` 支援 Swift 6 並發
- `CGCEmitCapable` 協議讓 HTTPServer 不依賴具體模型類別
- 錯誤路徑也返回 `CGCEmitResponse` (success=false), 避免 HTTP 4xx/5xx

### 5.2 完整代碼

```swift
// CGCEmitTypes.swift
// CGC PD 分離的 emit 端點類型定義.
// 被 HTTPServer 和 ServerModelSession 共用.

import Foundation

/// CGC emit 請求 (Windows coordinator → Mac TurboFieldfare).
public struct CGCEmitRequest: Codable, Sendable {
    /// 要 prefill 的文本.
    public let prompt: String
    /// 請求 ID (由 coordinator 生成).
    public let request_id: String
    /// 預期最大序列長度 (用於 accumulation buffer 分配).
    public let max_seq_len: Int

    public init(prompt: String, request_id: String, max_seq_len: Int = 4096) {
        self.prompt = prompt
        self.request_id = request_id
        self.max_seq_len = max_seq_len
    }
}

/// CGC emit 回應 (Mac TurboFieldfare → Windows coordinator).
public struct CGCEmitResponse: Codable, Sendable {
    public let success: Bool
    public let error: String?
    /// base64 編碼的 float32 little-endian 字節流 [seq_len * hidden_dim].
    public let hidden_state_b64: String?
    public let seq_len: Int
    public let hidden_dim: Int
    /// 完成到第幾層 (通常 = num_layers).
    public let finished_layer: Int
    public let model_id: String
    public let request_id: String
    /// 統計信息.
    public let prefill_ms: Double

    public init(success: Bool,
                error: String? = nil,
                hidden_state_b64: String? = nil,
                seq_len: Int = 0,
                hidden_dim: Int = 0,
                finished_layer: Int = 0,
                model_id: String,
                request_id: String,
                prefill_ms: Double = 0) {
        self.success = success
        self.error = error
        self.hidden_state_b64 = hidden_state_b64
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.finished_layer = finished_layer
        self.model_id = model_id
        self.request_id = request_id
        self.prefill_ms = prefill_ms
    }
}

/// 支持 CGC emit 的推理後端協議.
public protocol CGCEmitCapable {
    func cgcEmit(_ request: CGCEmitRequest) async throws -> CGCEmitResponse
}
```

### 5.3 欄位說明

| 欄位 | 類型 | 說明 |
|------|------|------|
| `prompt` | String | 原始文本 (非 token IDs, 避免 tokenizer 不一致) |
| `max_seq_len` | Int | accumulation buffer 預分配大小, 預設 4096 |
| `hidden_state_b64` | String? | base64 編碼, null 表示失敗 |
| `finished_layer` | Int | prefill 完成的層數, MVP = num_layers |
| `prefill_ms` | Double | prefill 耗時, 用於效能監控 |

---

## 6. 改動 2: RealForwardRunner.swift (Gemma4)

**路徑**: `Sources/TurboFieldfare/Runtime/Inference/RealForwardRunner.swift`
**repo**: `turbo-fieldfare-github-official/` (Gemma4) 和 `prime-agent-worktrees/turbo-fieldfare/` (Qwen3.6, 備用)

### 6.1 設計理念

- **零侵入**: `cgcAccumulationEnabled` 預設 false, 不影響正常推理路徑
- **記憶體高效**: 用 `MTLBuffer` 共享模式, CPU 可直接讀取, 無需額外複製
- **分塊累積**: prefill 是 chunked 進行的 (每 chunk 128-512 tokens), 每個 chunk 完成後立刻 blit 到累積 buffer, 避免保存所有 chunk 的中間結果
- **FP16 儲存, FP32 輸出**: 累積用 FP16 省一半記憶體, 輸出時轉 FP32

### 6.2 改動點 A: 新增成員變數

在 `copyLastHiddenState()` 方法後面加入:

```swift
// MARK: - CGC PD accumulation (完整序列 hidden state 捕獲)

private var cgcAccumulationBuffer: MTLBuffer?
private var cgcAccumulatedTokens: Int = 0
private var cgcAccumulationEnabled: Bool = false
```

**說明**:
- `cgcAccumulationBuffer`: 共享模式 MTLBuffer, 大小 = `maxSeqLen * hiddenSize * 2 bytes` (FP16)
- `cgcAccumulatedTokens`: 已累積的 token 數, 0 表示未啟用
- `cgcAccumulationEnabled`: 開關, 預設 false

### 6.3 改動點 B: 新增公開方法

```swift
/// 啟用 hidden state 累積. 必須在 prefillChunked 前呼叫.
/// - Parameter maxSeqLen: 預期最大序列長度 (用於分配 buffer).
public func cgcEnableAccumulation(maxSeqLen: Int) {
    let D = cfg.hiddenSize
    let bytes = maxSeqLen * D * MemoryLayout<Float16>.stride
    if cgcAccumulationBuffer == nil || cgcAccumulationBuffer!.length < bytes {
        cgcAccumulationBuffer = ctx.device.makeBuffer(
            length: bytes, options: .storageModeShared)
        cgcAccumulationBuffer?.label = "cgc.accumulation"
    }
    cgcAccumulatedTokens = 0
    cgcAccumulationEnabled = true
}

/// 關閉累積並釋放 buffer.
public func cgcDisableAccumulation() {
    cgcAccumulationBuffer = nil
    cgcAccumulatedTokens = 0
    cgcAccumulationEnabled = false
}

/// 取得已累積的完整 hidden state [seqLen * D] (fp32).
/// 必須在 prefillChunked 完成後呼叫.
public func cgcCopyAccumulatedHidden() -> [Float] {
    guard let buf = cgcAccumulationBuffer, cgcAccumulatedTokens > 0 else {
        return []
    }
    let D = cfg.hiddenSize
    let n = cgcAccumulatedTokens
    var result = [Float](repeating: 0, count: n * D)
    let ptr = buf.contents().bindMemory(to: Float16.self, capacity: n * D)
    for i in 0..<(n * D) { result[i] = Float(ptr[i]) }
    return result
}

/// 已累積的 token 數.
public var cgcAccumulatedTokenCount: Int { cgcAccumulatedTokens }
```

**關鍵點**:
- `storageModeShared` 讓 CPU/GPU 共享同一塊記憶體 (Apple Silicon UMA 架構)
- buffer 採「惰性擴容」策略: 只有當現有 buffer 不夠大時才重新分配
- `cgcDisableAccumulation()` 把 buffer 設為 nil, 讓 ARC 自動釋放

### 6.4 改動點 C: 新增私有 blit 方法

```swift
/// 在 executePrefillChunk 後呼叫: 把 scratch.hidden 的 [B, D] blit 到累積 buffer.
/// 注意: 必須在 scratch.hidden 被下個 chunk 覆寫前呼叫.
private func cgcAccumulateChunk(scratch: PrefillChunkScratchBuffers,
                                 tokenOffset: Int,
                                 tokenCount: Int,
                                 commandBuffer: MTLCommandBuffer) {
    guard cgcAccumulationEnabled,
          let accBuf = cgcAccumulationBuffer else { return }
    let D = cfg.hiddenSize
    let rowBytes = D * MemoryLayout<Float16>.stride
    guard let blit = commandBuffer.makeBlitCommandEncoder() else { return }
    blit.copy(from: scratch.hidden,
              sourceOffset: 0,
              to: accBuf,
              destinationOffset: tokenOffset * rowBytes,
              size: tokenCount * rowBytes)
    blit.endEncoding()
}
```

**為什麼用 blit**:
- `scratch.hidden` 是 `storageModePrivate` (GPU 專用), CPU 不能直接讀
- blit encoder 是 GPU→GPU/CPU 複製的標準方式, 不佔用計算單元
- 複製到 `storageModeShared` 的 accBuf 後, CPU 可直接讀

### 6.5 改動點 D: prefillChunked 循環內插入累積

找到 `prefillChunked` 方法裡的 span 循環:

```swift
for (spanIndex, span) in spans.enumerated() {
    let lower = tokens.index(tokens.startIndex, offsetBy: span.tokenOffset)
    let upper = tokens.index(lower, offsetBy: span.tokenCount)
    try await executePrefillChunk(
        tokens: tokens[lower..<upper],
        startPosition: span.startPosition,
        outputMode: outputMode,
        logits: logits,
        scratch: scratch,
        config: config,
        writeFinalHead: spanIndex == spans.count - 1)

    // ===== 新增: CGC 累積 =====
    if cgcAccumulationEnabled {
        runSync { cb in
            cgcAccumulateChunk(scratch: scratch,
                                tokenOffset: span.tokenOffset,
                                tokenCount: span.tokenCount,
                                commandBuffer: cb)
        }
        cgcAccumulatedTokens = span.tokenOffset + span.tokenCount
    }
    // ===========================

    onProgress(span.completedCount)
}
```

**時機的重要性**:
- 必須在 `executePrefillChunk` **之後** (此時 scratch.hidden 已是當前 chunk 的結果)
- 必須在下一個 span 迭代**之前** (下一次 executePrefillChunk 會覆寫 scratch.hidden)
- `runSync` 確保 blit 完成後才繼續

---

## 7. 改動 3: Qwen36ForwardRunner.swift (Qwen3.6)

**路徑**: `Sources/TurboFieldfare/Runtime/Inference/Qwen36ForwardRunner.swift`
**repo**: 僅 `prime-agent-worktrees/turbo-fieldfare/`

### 7.1 與 RealForwardRunner 的差異

Qwen36ForwardRunner 有自己的 prefill 實作 (不用 RealForwardRunner 的), 結構差異:

| 項目 | RealForwardRunner (Gemma4) | Qwen36ForwardRunner |
|------|---------------------------|---------------------|
| 當前 chunk hidden buffer | `scratch.hidden` (PrefillChunkScratchBuffers 成員) | `self.chunkHidden` (實例成員, `[C, 2048]`) |
| blit 指令提交 | 復用上層的 commandBuffer | 自己建立獨立 commandBuffer 並 waitUntilCompleted |
| 累積方法簽名 | `cgcAccumulateChunk(scratch:tokenOffset:tokenCount:commandBuffer:)` | `cgcAccumulateChunk(tokenOffset:tokenCount:)` |

### 7.2 改動點 A: 新增成員變數

```swift
private var cgcAccumulationBuffer: MTLBuffer?
private var cgcAccumulatedTokens: Int = 0
private var cgcAccumulationEnabled: Bool = false
```

### 7.3 改動點 B: 新增公開方法 (與 RealForwardRunner 相同)

```swift
public func cgcEnableAccumulation(maxSeqLen: Int) {
    let D = cfg.hiddenSize
    let bytes = maxSeqLen * D * MemoryLayout<Float16>.stride
    if cgcAccumulationBuffer == nil || cgcAccumulationBuffer!.length < bytes {
        cgcAccumulationBuffer = context.device.makeBuffer(
            length: bytes, options: .storageModeShared)
        cgcAccumulationBuffer?.label = "cgc.accumulation"
    }
    cgcAccumulatedTokens = 0
    cgcAccumulationEnabled = true
}

public func cgcDisableAccumulation() {
    cgcAccumulationBuffer = nil
    cgcAccumulatedTokens = 0
    cgcAccumulationEnabled = false
}

public func cgcCopyAccumulatedHidden() -> [Float] {
    guard let buf = cgcAccumulationBuffer, cgcAccumulatedTokens > 0 else {
        return []
    }
    let D = cfg.hiddenSize
    let n = cgcAccumulatedTokens
    var result = [Float](repeating: 0, count: n * D)
    let ptr = buf.contents().bindMemory(to: Float16.self, capacity: n * D)
    for i in 0..<(n * D) { result[i] = Float(ptr[i]) }
    return result
}

public var cgcAccumulatedTokenCount: Int { cgcAccumulatedTokens }
```

### 7.4 改動點 C: 新增私有 blit 方法 (不同!)

```swift
/// 在 prefillChunked 的 span 完成後, blit chunkHidden[0..B] 到累積 buffer.
private func cgcAccumulateChunk(tokenOffset: Int, tokenCount: Int) {
    guard cgcAccumulationEnabled,
          let accBuf = cgcAccumulationBuffer else { return }
    let D = cfg.hiddenSize
    let rowBytes = D * MemoryLayout<Float16>.stride
    guard let cb = context.queue.makeCommandBuffer(),
          let blit = cb.makeBlitCommandEncoder() else { return }
    blit.copy(from: chunkHidden,            // ← 注意: chunkHidden 不是 scratch
              sourceOffset: 0,
              to: accBuf,
              destinationOffset: tokenOffset * rowBytes,
              size: tokenCount * rowBytes)
    blit.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()  // ← 同步等待, 確保複製完成
}
```

**為什麼 Qwen36 版自己建 commandBuffer**:
- Qwen36ForwardRunner 的 prefillChunked 沒有 runSync 機制
- 用 `cb.waitUntilCompleted()` 確保 blit 完成後才返回, 避免資料競爭

### 7.5 改動點 D: prefillChunked 循環內插入累積

找到 `prefillChunked` 方法裡的 span 循環 (約在第 1952 行附近):

```swift
// ... 原有 chunk 處理邏輯 ...

// ===== 新增: CGC 累積 =====
if cgcAccumulationEnabled {
    cgcAccumulateChunk(tokenOffset: span.tokenOffset,
                        tokenCount: span.tokenCount)
    cgcAccumulatedTokens = span.tokenOffset + span.tokenCount
}
// ===========================

onProgress(span.completedCount)
```

---

## 8. 改動 4: ServerInference.swift (兩版共用)

**路徑**: `Sources/TurboFieldfareServer/Core/ServerInference.swift`
**兩個 repo 都要改, 內容完全相同**。

### 8.1 設計理念

- 在 `ServerModelSession` 上加 extension 實作 `CGCEmitCapable`
- 不修改既有方法, 只新增 `cgcEmit` 和輔助計算屬性
- 錯誤路徑全程返回 `CGCEmitResponse` (success=false), 不拋異常給上層

### 8.2 改動位置

在 `ServerModelSession` 類別的**最後一個右大括號 `}` 之前**加入輔助屬性, 之後加入 extension。

### 8.3 完整代碼

```swift
    // MARK: - CGC emit (PD 分離 prefill 端)

    /// 模型 ID (用於 CGCEmitResponse).
    private var cgcModelID: String {
        // 從 model 的 config 或路徑推斷; 若有 modelID 屬性則直接用.
        // 這裡用 hidden_size 粗略辨識.
        switch cfg.hiddenSize {
        case 2816: return "gemma4-26b-a4b"
        case 2048: return "qwen36-35b-a3b"
        default: return "model-\(cfg.hiddenSize)"
        }
    }

    private var cfg: ArchConfig { model.config }
}   // ← ServerModelSession 類別結束

// MARK: - CGCEmitCapable

extension ServerModelSession: CGCEmitCapable {

    /// CGC emit: 執行 prefill 並返回完整序列的 hidden state.
    ///
    /// 流程:
    /// 1. tokenize prompt
    /// 2. 啟用 accumulation buffer
    /// 3. prefillChunked (只跑 prefill, 不 decode)
    /// 4. copy accumulated hidden → [Float]
    /// 5. base64 encode → CGCEmitResponse
    public func cgcEmit(_ request: CGCEmitRequest) async throws -> CGCEmitResponse {
        let t0 = Date()

        // 1. tokenize
        let promptIDs: [Int32] = tokenizer.encode(request.prompt, addBOS: false)
        guard !promptIDs.isEmpty else {
            return CGCEmitResponse(
                success: false, error: "empty prompt after tokenize",
                model_id: cgcModelID, request_id: request.request_id)
        }
        guard promptIDs.count < maxContext else {
            return CGCEmitResponse(
                success: false,
                error: "prompt \(promptIDs.count) tokens exceeds maxContext \(maxContext)",
                model_id: cgcModelID, request_id: request.request_id)
        }

        // 2. 啟用 accumulation
        runner.cgcEnableAccumulation(maxSeqLen: max(request.max_seq_len, promptIDs.count))

        // 3. reset runner (從乾淨狀態開始)
        runner.reset()

        // 4. prefill — 只跑 prefill, 不需要 decode
        do {
            _ = try await runner.prefillChunked(
                tokens: promptIDs[...],
                startPosition: 0,
                outputMode: .logits,
                config: prefillConfig,
                into: scratch.logits,
                onProgress: { _ in })
        } catch {
            runner.cgcDisableAccumulation()
            return CGCEmitResponse(
                success: false, error: "prefill failed: \(error)",
                model_id: cgcModelID, request_id: request.request_id)
        }

        // 5. copy accumulated hidden
        let hiddenFloats = runner.cgcCopyAccumulatedHidden()
        let seqLen = runner.cgcAccumulatedTokenCount
        runner.cgcDisableAccumulation()

        guard !hiddenFloats.isEmpty, seqLen > 0 else {
            return CGCEmitResponse(
                success: false, error: "accumulation empty after prefill",
                model_id: cgcModelID, request_id: request.request_id)
        }

        let hiddenDim = cfg.hiddenSize
        let prefillMs = Date().timeIntervalSince(t0) * 1000

        // 6. base64 encode (float32 little-endian)
        // Apple Silicon 原生即為 little-endian, 直接用 withUnsafeBytes
        let data = Data(bytes: hiddenFloats,
                        count: hiddenFloats.count * MemoryLayout<Float>.size)
        let b64 = data.base64EncodedString()

        return CGCEmitResponse(
            success: true,
            hidden_state_b64: b64,
            seq_len: seqLen,
            hidden_dim: hiddenDim,
            finished_layer: cfg.numLayers,
            model_id: cgcModelID,
            request_id: request.request_id,
            prefill_ms: prefillMs)
    }
}
```

### 8.4 關鍵細節

**1. `addBOS: false`**
- TurboFieldfare 的 tokenizer 內部會處理 BOS, 這裡不加避免重複

**2. `runner.reset()`**
- 從乾淨狀態開始, 清空 KV cache
- PD 模式下每次 emit 都是獨立請求, 不復用上下文

**3. `outputMode: .logits`**
- 雖然我們只要 hidden state, 但 prefillChunked 必須指定一個 outputMode
- `.logits` 會計算最後 token 的 logits (無害, 多算一點點)

**4. `cgcDisableAccumulation()` 必須呼叫**
- 即使失敗路徑也要呼叫, 否則 buffer 不釋放, 下次請求會殘留
- 上面代碼在 catch 分支已處理

**5. base64 編碼**
- `Data(bytes:count:)` 直接拷貝 `[Float]` 的位元組
- Apple Silicon 是 little-endian, Python 端 `torch.frombuffer(dtype=float32)` 也是 LE, 兩端一致

---

## 9. 改動 5: HTTPServer.swift (兩版共用)

**路徑**: `Sources/TurboFieldfareServer/Core/HTTPServer.swift`
**兩個 repo 都要改, 內容完全相同**。

### 9.1 改動點 A: 加路由

在 `switch (head.method, path)` 裡, `/v1/chat/completions` case 之後, 加 `/v1/cgc/emit` case:

```swift
case (.POST, "/v1/chat/completions"):
    guard head.headers.first(name: "content-type")?
        .lowercased().hasPrefix("application/json") == true else {
        writeError(context, status: .unsupportedMediaType,
                   OpenAIErrorEnvelope(message: "content-type must be application/json",
                                       code: "unsupported_media_type"))
        return
    }
    handleCompletion(body: body, context: context)

// ===== 新增 =====
case (.POST, "/v1/cgc/emit"):
    guard head.headers.first(name: "content-type")?
        .lowercased().hasPrefix("application/json") == true else {
        writeError(context, status: .unsupportedMediaType,
                   OpenAIErrorEnvelope(message: "content-type must be application/json",
                                       code: "unsupported_media_type"))
        return
    }
    handleCGCEmit(body: body, context: context)
// ================

// 同時更新 method not allowed case:
case (_, "/health"), (_, "/v1/models"), (_, "/v1/chat/completions"), (_, "/v1/cgc/emit"):
    writeError(context, status: .methodNotAllowed,
               OpenAIErrorEnvelope(message: "method not allowed",
                                   code: "method_not_allowed"))
default:
    writeError(context, status: .notFound,
               OpenAIErrorEnvelope(message: "route not found",
                                   code: "not_found"))
```

### 9.2 改動點 B: 加 handler 方法

在 `handleCompletion` 方法前面加入:

```swift
/// 處理 /v1/cgc/emit 請求 — CGC PD 分離 prefill 端.
private func handleCGCEmit(body: ByteBuffer, context: ChannelHandlerContext) {
    let bytes = body.getBytes(at: body.readerIndex, length: body.readableBytes) ?? []
    do {
        let req = try JSONDecoder().decode(CGCEmitRequest.self, from: Data(bytes))
        Task {
            do {
                guard let cgc = backend as? CGCEmitCapable else {
                    writeError(context, status: .notImplemented,
                               OpenAIErrorEnvelope(
                                message: "this backend does not support CGC emit",
                                code: "cgc_not_supported"))
                    return
                }
                let resp = try await cgc.cgcEmit(req)
                writeCodable(context, status: .ok, resp)
            } catch {
                writeError(context, status: .internalServerError,
                           OpenAIErrorEnvelope(
                            message: "cgc emit failed: \(error)",
                            code: "cgc_emit_error"))
            }
        }
    } catch {
        writeError(context, status: .badRequest,
                   OpenAIErrorEnvelope(message: "invalid JSON: \(error)",
                                       code: "invalid_json"))
    }
}
```

### 9.3 關鍵細節

**1. `backend as? CGCEmitCapable`**
- `backend` 是 HTTPServer 持有的推理後端實例 (型別可能是某個 protocol)
- 用 `as?` 軟轉換, 如果後端不支援 CGC emit, 返回 501 Not Implemented
- 這樣設計是為了未來可以有「不支援 CGC」的後端 (例如純 decode 端)

**2. `Task { ... }`**
- cgcEmit 是 async, 但 channel handler 是 sync
- 用 Task 機制跑異步, 不阻塞 event loop
- 注意: Task 不會阻止 HTTP keep-alive 關閉, swift-nio 會自動管理

**3. 錯誤層級**
- JSON 解析失敗 → 400 BadRequest
- 後端不支援 CGC → 501 Not Implemented
- prefill 執行失敗 → 500 InternalServerError
- 業務錯誤 (prompt 太長等) → 200 + CGCEmitResponse(success=false)

---

## 10. 編譯與部署

### 10.1 環境需求

- macOS 14+ (Sonoma / Sequoia)
- Xcode 16+ (Swift 5.10+)
- Apple Silicon (M1/M2/M3/M4)
- TurboFieldfare 依賴已安裝 (參考 repo README)

### 10.2 Mac A (Gemma4) 編譯步驟

```bash
# 1. 進入 repo
cd /path/to/turbo-fieldfare-github-official

# 2. 確認改動已套用 (5 處)
git diff --stat
# 預期輸出:
#  CGCEmitTypes.swift                                  |  62 +++
#  Sources/TurboFieldfare/Runtime/Inference/RealForwardRunner.swift |  70 ++-
#  Sources/TurboFieldfareServer/Core/HTTPServer.swift  |  40 +-
#  Sources/TurboFieldfareServer/Core/ServerInference.swift | 100 +++

# 3. 編譯 (Debug)
swift build

# 4. 編譯 (Release, 推薦)
swift build -c release

# 5. 啟動 server (用本地模型路徑)
swift run TurboFieldfareServer \
  --model /path/to/gemma4-26b-a4b.mlmodel \
  --port 8080
```

### 10.3 Mac B (Qwen3.6) 編譯步驟

```bash
# 1. 進入 repo
cd /path/to/prime-agent-worktrees/turbo-fieldfare

# 2. 確認改動已套用 (6 處, 多 Qwen36ForwardRunner.swift)
git diff --stat
# 預期輸出:
#  CGCEmitTypes.swift                                  |  62 +++
#  Sources/TurboFieldfare/Runtime/Inference/Qwen36ForwardRunner.swift | 70 ++-
#  Sources/TurboFieldfare/Runtime/Inference/RealForwardRunner.swift  |  70 ++-
#  Sources/TurboFieldfareServer/Core/HTTPServer.swift  |  40 +-
#  Sources/TurboFieldfareServer/Core/ServerInference.swift | 100 +++

# 3. 編譯
swift build -c release

# 4. 啟動 server
swift run TurboFieldfareServer \
  --model /path/to/qwen36-35b-a3b.mlmodel \
  --port 8080
```

### 10.4 防火牆設定

macOS 預設會阻擋外部連入, 啟動 server 後系統會彈窗詢問, 選擇「允許」。

或者手動放行:
```
系統偏好設定 → 安全性與隱私 → 防火牆 → 防火牆選項 → 允許 TurboFieldfareServer
```

### 10.5 驗證服務啟動

從 Windows 或同機測試:

```bash
# 健康檢查
curl http://<mac-ip>:8080/health
# {"status":"ok"}

# 模型清單
curl http://<mac-ip>:8080/v1/models
# {"object":"list","data":[{"id":"...","object":"model",...}]}
```

---

## 11. 驗證流程

### 11.1 單機快速驗證 (curl)

```bash
curl -X POST http://localhost:8080/v1/cgc/emit \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Hello, world. This is a test.",
    "request_id": "test-001",
    "max_seq_len": 1024
  }' | python3 -c "
import json, sys, base64
resp = json.load(sys.stdin)
print('success:', resp['success'])
print('model_id:', resp['model_id'])
print('seq_len:', resp['seq_len'])
print('hidden_dim:', resp['hidden_dim'])
print('prefill_ms:', resp['prefill_ms'])
b64 = resp['hidden_state_b64']
print('b64 length:', len(b64))
print('decoded bytes:', len(base64.b64decode(b64)))
expected = resp['seq_len'] * resp['hidden_dim'] * 4
print('expected bytes:', expected)
assert len(base64.b64decode(b64)) == expected, 'size mismatch!'
print('✅ OK')
"
```

### 11.2 跨機驗證 (Windows → 兩台 Mac)

在 Windows 上執行:

```powershell
# 安裝依賴
py -m pip install aiohttp torch

# 單文件驗證
py verify_pd_emit.py `
  --gemma4-url http://192.168.101.X:8080 `
  --qwen36-url http://192.168.101.Y:8080 `
  --file prompt.txt `
  --output hidden_pair.npz
```

預期輸出:
```
驗證結果
  Gemma4 (source):
    shape: (12, 2816)
    prefill: 234.1ms
    ✅ 數值正常
  Qwen3.6 (target):
    shape: (12, 2048)
    prefill: 312.5ms
    ✅ 數值正常

跨模型比較
  序列長度一致 (12)
  維度差: 768 (需要 MoT-h 翻譯)

✅ 已保存到 hidden_pair.npz
```

### 11.3 批量採集 + 訓練

```powershell
# 1. 生成語料
py gen_sample_corpus.py --output corpus --per-category 20

# 2. 批量採集
py collect_batch.py `
  --gemma4-url http://192.168.101.X:8080 `
  --qwen36-url http://192.168.101.Y:8080 `
  --input corpus `
  --output train.pt

# 3. 訓練 MoT-h (在 Mac B 上用 MPS 加速)
python3 train_mot_h.py `
  --data train.pt `
  --output mot_h.pt `
  --device mps `
  --epochs 10
```

### 11.4 驗收標準

| 階段 | 指標 | 通過標準 |
|------|------|---------|
| emit 端點 | HTTP 200 + success=true | ✅ 必須 |
| hidden state 大小 | bytes == seq_len * hidden_dim * 4 | ✅ 必須 |
| 數值健康 | 無 NaN, 無 Inf | ✅ 必須 |
| 跨機一致 | 兩台 Mac seq_len 一致 | ✅ 必須 (或差 < 5%) |
| 訓練收斂 | val_mse 單調下降 | ✅ 必須 |
| 翻譯質量 | val_mse < 1e-2 | ⭐ 可用 |
| 翻譯質量 | val_mse < 1e-3 | ⭐⭐ 接近無損 |
| 端到端 decode | 文字可讀, 語義對 | ⭐⭐⭐ 目標 |

---

## 12. 已知限制與 v2 路線圖

### 12.1 MVP v1 的限制

1. **單層 hidden state**: 只翻譯末層, 信息量不足
   - 訓練數據只採末層 hidden, MoT-h 學不到中間層資訊
   - Context Replay 用末層反推所有層 KV, 是粗糙近似

2. **序列長度對齊**: 不同 tokenizer 可能導致 seq_len 不一致
   - 目前用截斷到較短長度 (粗暴)
   - 理論上應該用 BPE 對齊, 但複雜度高

3. **單請求同步**: 一次只能處理一個 emit 請求
   - runner.reset() 會清空狀態, 不支援並發
   - 多請求需排隊

4. **無 batch 採集**: 無法一次性 emit 多個 prompt
   - 每個 prompt 獨立 POST, 網路開銷大

### 12.2 v2 路線圖

| 升級 | 預期收益 | 工作量 |
|------|---------|--------|
| **4 層通道 hidden** | val_mse 1e-2 → 1e-3, 接近無損 | Swift +2h, Python +1h |
| **batch emit** | 採集效率 3-5x | Swift +1h |
| **增量 emit** | 長 prompt 分多次 emit, 不超時 | Swift +2h |
| **KV cache 直傳** | 跳過 Context Replay, 真正無損 | Swift +4h, 協議大改 |

### 12.3 v2 升級 4 層通道的具體做法

**Swift 端改動**:
1. `cgcAccumulationBuffer` 改為 `[MTLBuffer]` (4 個, 對應 4 層)
2. `cgcEnableAccumulation(maxSeqLen:layers:[Int])` 接收層索引陣列
3. `executePrefillChunk` 在指定層執行後, blit 該層 hidden
4. `cgcCopyAccumulatedHidden()` 返回 `[[Float]]` (4 個陣列)

**Python 端改動**:
1. `CGCEmitResponse.hidden_state_b64` 改為 `hidden_states_b64: [String]` (4 個)
2. `train_mot_h.py` 數據集 `h_src` shape 從 `[seq, 2816]` 變為 `[4, seq, 2816]`
3. MoT-h 模型直接吃 4 層輸入 (原本就有 window_size=4 的設計)

### 12.4 接續品質的理論分析

| 方案 | 資訊論下界 | 實際可達 |
|------|----------|---------|
| MVP v1 (末層 + Context Replay) | 70-80% 保留 | 文字可讀, 有輕微亂碼 |
| v2 (4 層通道) | 90-95% 保留 | 接近原生, 難以察覺差異 |
| v3 (每層 hidden + 直傳 KV) | 100% | 完全一致 (但傳輸量大) |

**為什麼 100% 不可能 (在不傳每層 KV 的前提下)**:
- Qwen3.6 第 l 層的 K/V = `RMSNorm_l(hidden_l) @ Wk_l`
- 用末層 hidden 反推第 l 層 K/V, 缺少 `RMSNorm_l` 的非線性資訊
- MoT-h 學到的是「平均最佳翻譯」, 無法對每個 token 都完美

---

## 附錄 A: 檔案改動清單

### Mac A (Gemma4, turbo-fieldfare-github-official)

```
Sources/TurboFieldfareServer/Core/CGCEmitTypes.swift                  [新建]
Sources/TurboFieldfare/Runtime/Inference/RealForwardRunner.swift      [修改]
Sources/TurboFieldfareServer/Core/ServerInference.swift               [修改]
Sources/TurboFieldfareServer/Core/HTTPServer.swift                    [修改]
```

### Mac B (Qwen3.6, prime-agent-worktrees/turbo-fieldfare)

```
Sources/TurboFieldfareServer/Core/CGCEmitTypes.swift                  [新建]
Sources/TurboFieldfare/Runtime/Inference/RealForwardRunner.swift      [修改]
Sources/TurboFieldfare/Runtime/Inference/Qwen36ForwardRunner.swift    [修改]
Sources/TurboFieldfareServer/Core/ServerInference.swift               [修改]
Sources/TurboFieldfareServer/Core/HTTPServer.swift                    [修改]
```

### Windows (端側, cgc-engine/pd)

```
verify_pd_emit.py       [新建]  單文件驗證
collect_batch.py        [新建]  文件夾批量採集
train_mot_h.py          [新建]  MoT-h 訓練 (CPU/MPS/CUDA)
gen_sample_corpus.py    [新建]  示例語料生成器
```

---

## 附錄 B: 完整 API Reference

### B.1 POST /v1/cgc/emit

**描述**: 執行 prefill 並返回完整序列的末層 hidden state

**請求**:

| 欄位 | 類型 | 必填 | 說明 |
|------|------|------|------|
| prompt | string | ✅ | 要 prefill 的文本 |
| request_id | string | ✅ | 請求 ID (caller 生成) |
| max_seq_len | int | ❌ | accumulation buffer 預分配, 預設 4096 |

**回應**:

| 欄位 | 類型 | 說明 |
|------|------|------|
| success | bool | 是否成功 |
| error | string? | 失敗原因 |
| hidden_state_b64 | string? | base64(float32 LE [seq_len * hidden_dim]) |
| seq_len | int | 實際序列長度 |
| hidden_dim | int | hidden state 維度 (Gemma4=2816, Qwen3.6=2048) |
| finished_layer | int | prefill 完成的層數 |
| model_id | string | 模型識別碼 |
| request_id | string | 回顯請求 ID |
| prefill_ms | double | prefill 耗時 (毫秒) |

**錯誤碼**:

| HTTP Status | code | 說明 |
|-------------|------|------|
| 400 | invalid_json | 請求 body 不是合法 JSON |
| 415 | unsupported_media_type | Content-Type 不是 application/json |
| 500 | cgc_emit_error | prefill 執行出錯 |
| 501 | cgc_not_supported | 後端不支援 CGC emit |

---

## 附錄 C: 疑難排解

### C.1 編譯錯誤

**「cannot find type 'CGCEmitCapable' in scope」**
- 原因: CGCEmitTypes.swift 沒被加入 target
- 解決: 在 Xcode 確認檔案隸屬於 TurboFieldfareServer target

**「'ArchConfig' is inaccessible due to 'internal' protection level」**
- 原因: ArchConfig 是 internal, 但 ServerInference 在另一個 module
- 解決: 確認 `private var cfg: ArchConfig { model.config }` 中 `model.config` 的存取層級, 必要時改成 `public`

### C.2 執行時錯誤

**「accumulation empty after prefill」**
- 原因: prefillChunked 沒進入 span 循環 (可能 prompt 為空)
- 解決: 檢查 tokenizer.encode 結果, 確認 promptIDs 非空

**「prompt N tokens exceeds maxContext M」**
- 原因: prompt 太長
- 解決: 縮短 prompt, 或調大 `--max-context` 啟動參數

### C.3 網路問題

**Windows 連不上 Mac**
- 檢查 Mac 防火牆是否放行 TurboFieldfareServer
- 檢查兩機在同一網段 (subnet)
- 用 `curl http://<mac-ip>:8080/health` 測試

**hidden_state_b64 解碼後大小不對**
- 檢查 Python 端 `torch.frombuffer` 的 dtype 是否為 float32
- 檢查 base64 解碼後 bytes 是否等於 `seq_len * hidden_dim * 4`

---

## 附錄 D: 與 Windows 端的對接

### D.1 coordinator.py 端點

Windows 端 coordinator.py 已實作 `POST /v1/cgc/ingest` 端點, 接收 Mac 主動推送的 hidden state:

```python
# coordinator.py (簡化)
@app.post("/v1/cgc/ingest")
async def ingest(req: IngestRequest):
    # req.hidden_state_b64, req.model_id, req.request_id, ...
    hidden = decode_hidden_state(req.hidden_state_b64,
                                  req.seq_len, req.hidden_dim)
    # 存入共享記憶體 / 轉發給 MoT-h / Qwen3.6 decode
    return {"success": True}
```

### D.2 兩種工作模式

**模式 A: Windows 主動拉 (pull)**
- Windows 向 Mac POST `/v1/cgc/emit`, Mac 返回 hidden state
- 用於: 訓練數據採集, 單次驗證
- 見: `verify_pd_emit.py`, `collect_batch.py`

**模式 B: Mac 主動推 (push)**
- Mac 推理完成後, 向 Windows POST `/v1/cgc/ingest`
- 用於: 線上服務, 即時 decode
- 見: `coordinator.py` (已實作)

MVP 階段先用模式 A, 模式 B 待 v2。

---

**文件結束**

如有問題, 對照本文檔逐項檢查 Swift 改動是否完整套用。所有 Python 腳本在 Windows 端 `cgc-engine/pd/` 目錄下。
