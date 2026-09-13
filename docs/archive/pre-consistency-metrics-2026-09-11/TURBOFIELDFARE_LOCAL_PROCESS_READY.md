# TurboFieldfare local_process Ready Checklist

這份清單是給 `prebuilt TurboFieldfareServer` 與 `gemma4.gturbo` 的 drop-in 驗收用。

## 固定 staging 口

- `TurboFieldfareServer`
  - `/Users/alexchuang/Documents/flashkv0516/var/external/turbofieldfare/bin/TurboFieldfareServer`
- `gemma4.gturbo`
  - `/Users/alexchuang/Documents/flashkv0516/var/external/turbofieldfare/models/gemma4.gturbo`

## 一步一步測試

### 方式 1：直接拿外部產物進來並測試

```bash
"/Users/alexchuang/Documents/flashkv0516/scripts/test_turbofieldfare_step_by_step.sh" \
  "/path/to/TurboFieldfareServer" \
  "/path/to/gemma4.gturbo"
```

### 方式 2：如果 staging 已經掛好，直接重測

```bash
"/Users/alexchuang/Documents/flashkv0516/scripts/test_turbofieldfare_step_by_step.sh"
```

## 拆開執行

### 1. 掛 staging

```bash
"/Users/alexchuang/Documents/flashkv0516/scripts/stage_turbofieldfare_dropin.sh" \
  "/path/to/TurboFieldfareServer" \
  "/path/to/gemma4.gturbo"
```

### 2. 檢查 env 解出來的路徑

```bash
source "/Users/alexchuang/Documents/flashkv0516/scripts/setup_turbofieldfare_env.sh"
```

期待至少看到：

```text
CGC_TURBOFIELDFARE_SERVER_BIN=/.../TurboFieldfareServer
CGC_TURBOFIELDFARE_MODEL=/.../gemma4.gturbo
CGC_TURBOFIELDFARE_READY=1
```

### 3. 跑 local_process smoke

```bash
"/Users/alexchuang/Documents/flashkv0516/scripts/smoke_turbofieldfare_local_process.sh"
```

期待結果：

```json
{
  "ready": true,
  "delivery": "local_process_ready"
}
```

## 現在已準備好的腳本

- `scripts/stage_turbofieldfare_dropin.sh`
- `scripts/setup_turbofieldfare_env.sh`
- `scripts/smoke_turbofieldfare_local_process.sh`
- `scripts/test_turbofieldfare_step_by_step.sh`

## 失敗時先看哪裡

- `server_bin` 是空：`TurboFieldfareServer` 還沒掛進 staging
- `model_dir` 是空：`gemma4.gturbo` 還沒掛進 staging
- `CGC_TURBOFIELDFARE_READY=0`：env 還沒解出完整 server + model
- `delivery = local_launch_blocked`：還在缺外部產物
- `delivery = local_process_failed`：server 啟動了，但 `/health` 沒 ready

## Smoke 產物位置

- request contract
  - `var/colibri/turbofieldfare_tf-smoke_request.json`
- receipt
  - `var/colibri/sessions/tf-smoke/turbofieldfare_receipt.json`
- log
  - `var/colibri/sessions/tf-smoke/turbofieldfare.log`
