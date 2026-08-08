# [OPEN] psi0-ddp-wrap

## Goal
- 找出 `2 GPU` 官方 `psi0` 在 `DistributedDataParallel(model, ...)` 建立前，是否已存在模型 state 異常：
- 某些參數或 buffer 不在預期 GPU
- 非法 dtype/device 混用
- wrap 前已帶著不乾淨的 CUDA state

## Current Symptom
- `1 GPU` 可進到訓練循環。
- `2 GPU` 會在 `trainer.prepare(accelerator)` 內、`DistributedDataParallel(...)` 建立期間報 `CUDA illegal memory access`。
- 先前 runtime evidence 已證明：
  - `before trainer.prepare(accelerator)` 事件可送出
  - `before DistributedDataParallel(model)` 事件可送出
  - `after DistributedDataParallel(model)` 事件送不出

## Hypotheses
1. 模型參數或 buffer 存在跨裝置混放，至少有一部分不在當前 rank 對應 GPU。
2. 模型存在 dtype/device 混用，導致 DDP constructor 在同步初始化時觸發底層 CUDA 錯誤。
3. 模型在 wrap 前已經帶著髒的 CUDA state，例如上一段初始化留下非法 tensor/buffer 狀態，直到 DDP 初始化才被放大。
4. 問題不是自訂 DDP kwargs，而是模型本體 state；因為官方 `Accelerator` 沒有傳 `kwargs_handlers`，`kwargs` 已確認為空。

## Evidence Collected
- `CUDA_LAUNCH_BLOCKING=1` 已正確設置。
- `ddp_handler = None`，`kwargs_repr='{}'`。
- `before DistributedDataParallel(model)` 事件可送出，且 `device_ids/output_device/local_process_index` 合理。
- 最新 `pre-fix-state-probe` runtime evidence：
  - rank0:
    - `expected_device='cuda:0'`
    - `param_device_counts={'cuda:0': 806}`
    - `buffer_device_counts={'cuda:0': 2}`
    - `bad_param_device_samples=[]`
    - `bad_buffer_device_samples=[]`
    - `sync_ok_before_ddp=true`
    - `probe_read_ok_before_ddp=true`
  - rank1:
    - `expected_device='cuda:1'`
    - `param_device_counts={'cuda:1': 806}`
    - `buffer_device_counts={'cuda:1': 2}`
    - `bad_param_device_samples=[]`
    - `bad_buffer_device_samples=[]`
    - `sync_ok_before_ddp=true`
    - `probe_read_ok_before_ddp=true`
  - 兩個 rank 的 dtype 分佈一致：
    - `param_dtype_counts={'torch.float32': 181, 'torch.bfloat16': 625}`
    - `buffer_dtype_counts={'torch.float32': 2}`
  - 兩個 rank 都能在 wrap 前成功讀取第一個 CUDA tensor 樣本：
    - `probe_read_sample_before_ddp='action_header.time_ins_embed.w:1.000000'`
  - 之後仍然沒有 `after DistributedDataParallel(model)` 事件，且官方 log 同一 run 仍在 `Num processes (GPUs) = 2` 後立刻報 `CUDA illegal memory access`

## Hypothesis Status
- H1 `某些參數或 buffer 不在預期 GPU`:
  - 目前被否決。
  - 兩個 rank 的 `bad_param_device_samples` / `bad_buffer_device_samples` 都為空，且裝置計數完全落在本 rank GPU。
- H2 `存在 dtype/device 混用`:
  - 部分成立，但目前看不到「非法」證據。
  - 的確存在混合 dtype：`181` 個 `torch.float32` 參數與 `625` 個 `torch.bfloat16` 參數，buffers 為 `torch.float32`。
  - 但此分佈在兩個 rank 完全對稱，且 wrap 前 `sync` 與 `probe_read` 都成功，因此目前只能判定為「存在 mixed dtype」，不能直接判定為 root cause。
- H3 `wrap 前已有不乾淨的 CUDA state`:
  - 目前被否決。
  - `torch.cuda.synchronize()` 與單 tensor `.item()` 探針在兩個 rank 都成功，沒有在 DDP wrap 前提前暴露 CUDA error。
- H4 `不是 DDP kwargs，而是模型/底層初始化問題`:
  - 目前持續成立。
  - `kwargs_repr='{}'`，且 `after DistributedDataParallel(model)` 事件始終缺失。

## Next Action
- 在真正執行的 `FinetuneTrainer.prepare()` 之前，補充更細的模型 state runtime 檢查：
  - 參數/buffer 的裝置集合、dtype 集合、異常項目樣本
  - CUDA memory summary 與 synchronize 前的錯誤探針
  - 必要時在 `accelerate.prepare_model()` 的 DDP constructor 前再補一次同內容快照
