# Debug Session: psi0-real-metadata

- Status: OPEN
- Goal: 補齊官方 Psi0 real-task 訓練所需的 LeRobot 相容 metadata，讓 `finetune-real-psi0.sh` 能從模型初始化推進到 dataset 建立成功並真正進入 step 1。
- Scope: 僅限資料與 metadata 修補、遠端只讀探查、官方 smoke 重跑；不修改官方訓練業務邏輯。

## Known Symptoms

- 官方 `finetune_real_psi0_config` 已可完成模型初始化與權重載入。
- 在建立 real-task dataset 時失敗，報缺少 `meta/info.json`。
- fallback 到 Hugging Face dataset repo 解析後，再次失敗為 `RepositoryNotFoundError`。

## Initial Hypotheses

1. real task 目錄缺少 LeRobot 必要的 `meta/info.json`，導致 loader 無法把它視為本地資料集。
2. `data.root_dir=real` 與 task 目錄層級不匹配，loader 實際期望的是另一層嵌套路徑或 repo root。
3. 除了 `meta/info.json` 外，還缺少其他 LeRobot 必備 metadata，例如 episode / stats / parquet 索引欄位中的部分鍵。
4. `patch_lerobot_meta.py` 可以生成所需 metadata，但現有 real task 目錄結構沒有先滿足它的輸入假設。
5. 官方 `.env` 仍指向 `/hfm`，即使本次 smoke 用環境覆寫成功，後續資料路徑仍可能在某些子流程裡混用舊根路徑。

## Evidence Log

- Pre-fix runtime evidence already captured via remote smoke run.
- Next step: inspect dataset loader expectations and remote task metadata files before any data mutation.
- Evidence update:
  - `LeRobotDatasetWrapper` 以 `real/<task>` 作為 dataset root，會直接讀 `real/<task>/meta/info.json`。
  - 遠端真實資料原本是空的 top-level `data/meta/videos`，完整 LeRobot 內容實際位於 `real/<task>/<task>/...`。
  - 已將 top-level `data/meta/videos` 改為指向 nested dataset 的 symlink。
  - 已補齊官方 shell 所需 `/hfm/data -> /nfs/embodied/psi0_home/data` 與 `/hfm/cache/checkpoints -> /nfs/embodied/psi0_home/cache/checkpoints`。
  - post-fix 官方 `finetune-real-psi0.sh` 已成功完成 dataset 建立，日誌出現：
    - `Downloading data: 100%|...| 80/80`
    - `Generating train split: 77584 examples`
    - `Training dataset size: 77,584`
    - `***** Running training *****`
  - 單卡 `1 GPU` 已進一步出現：
    - `Accelerator runs in: ...`
    - `Traing steps: 0/40000`
    - `Eval at global step 0`
  - `2 GPU` 已重跑並證實：
    - 成功通過 rendezvous、model init、dataset 建立與 `***** Running training *****`
    - 失敗發生在 `train.py:211` 之後、`train.py:218` 之前
    - 也就是 `trainer.prepare(accelerator)` / `resume_from_checkpoint()` 這個窗口，而不是 dataloader 建立、也不是 `global step 0` eval
  - `2 GPU` 報錯為多卡訓練時的 `CUDA illegal memory access` / `ProcessGroupNCCL watchdog`，屬於 distributed 路徑問題。
  - 進一步縮窄：
    - `PosttrainTrainer.prepare()` 的順序是：
      1. `accelerator.prepare(self.model, self.optimizer, self.lr_scheduler)`
      2. `accelerator.prepare(self.train_dataloader)`
      3. `accelerator.prepare(self.val_dataloader)`
    - `accelerate.Accelerator.prepare_model()` 在 `MULTI_GPU` 下的第一個 DDP 點是：
      - `torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, **kwargs)`
    - 以 `2 GPU + small batch + validation_steps=0 + val_num_batches=0` 重跑後，錯誤仍在 `train.py:211 -> 218` 之間發生，且從未進入 `Accelerator runs in ...`。
    - 因此可排除 `step 0 eval collective`，目前最像的第一個故障函式層是：
      - `PosttrainTrainer.prepare()`
      - 內部的 `accelerator.prepare(self.model, self.optimizer, self.lr_scheduler)`
      - 更具體是 `accelerate.prepare_model()` 中的 `DistributedDataParallel(...)` 包裝或其緊鄰的第一個 NCCL 同步。
  - `CUDA_LAUNCH_BLOCKING` 錯值來源已確認為官方 repo 的 `.env`：
    - `CUDA_LAUNCH_BLOCKING=true`
    - 以環境覆寫成 `CUDA_LAUNCH_BLOCKING=1` 後，警告 `Ignoring invalid value for boolean flag CUDA_LAUNCH_BLOCKING` 消失。
  - 使用 `CUDA_LAUNCH_BLOCKING=1` 的原始官方 `2 GPU` 重跑後：
    - 仍然在 `train.py:211` 之後立刻因 `CUDA illegal memory access` 終止
    - 這次 first observed failure 是 `rank 0`
    - stack 仍停留在 `c10_cuda_check_implementation` / `TensorImpl::~TensorImpl()` / `libtorch_python`，沒有前推到 dataloader 或 eval 路徑
  - `Accelerator` 建立處已確認：
    - `scripts/train.py` 只傳 `gradient_accumulation_steps`、`mixed_precision`、`log_with`、`project_config`、`fsdp_plugin`、`deepspeed_plugin`
    - 沒有傳 `kwargs_handlers`
  - `accelerate` 內部初始化規則已確認：
    - `self.ddp_handler = None`
    - 只有在 `kwargs_handlers` 中傳入 `DistributedDataParallelKwargs` 才會賦值
    - 因此本案可排除「自訂 DDP bucket / comm hook / static_graph 參數導致 early NCCL 崩潰」這條線。
  - 新一輪最小 runtime 插樁結果：
    - `scripts/train.py` 的 `A: before trainer.prepare(accelerator)` 事件，兩個 rank 都成功送出。
    - 真正會被執行的 trainer 是 `FinetuneTrainer`，不是先前假設的 `PosttrainTrainer`。
    - 修正插樁到 `FinetuneTrainer` 與實際載入的 `/nfs/embodied/.../accelerate/accelerator.py` 後，兩個 rank 都成功送出：
      - `C: before DistributedDataParallel(model)`
    - 但完全沒有看到：
      - `C: after DistributedDataParallel(model)`
    - `C-before` 事件資料顯示：
      - rank0: `device_ids=[0]`, `output_device=0`, `first_param_device='cuda:0'`, `first_param_dtype='torch.float32'`
      - rank1: `device_ids=[1]`, `output_device=1`, `first_param_device='cuda:1'`, `first_param_dtype='torch.float32'`
      - `kwargs_repr='{}'`
    - 因此可把第一個故障點從「prepare 窗口」進一步縮成：
      - `accelerate.prepare_model()`
      - 內部的 `torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, **{})`
      - 並且是 constructor 本身或其同步初始化過程先觸發 CUDA error，而不是 constructor 返回之後的後續步驟。

## Planned Steps

1. 讀本地 repo 中 LeRobot metadata 載入與 patch 腳本實作。
2. 只讀探查遠端 real task 目錄與現有 meta 檔。
3. 根據 loader 期望補 metadata。
4. 重跑官方 smoke。
5. 比對 pre-fix / post-fix 結果。

## Hypothesis Status

1. Confirmed: 缺失不是「完全沒有 metadata」，而是 loader 讀的 root 層級和 metadata 真實所在層級不一致。
2. Confirmed: `real/<task>` 與 `real/<task>/<task>` 的一層偏差就是 pre-fix 主因。
3. Rejected as primary cause: 不需要先新增新的 metadata 內容，只要把 root 對到既有的完整 LeRobot metadata 即可越過 dataset 建立。
4. Rejected for this fix path: `patch_lerobot_meta.py` 並不是補 `info.json` 的工具，它只清 parquet schema metadata。
5. Rejected for this phase: `/hfm` 路徑確實是官方 shell 的額外前提，但在補 symlink 之後已不再阻擋 dataset 與訓練啟動。
