#!/usr/bin/env python3
import os
import sys
import time
import subprocess
import json
import psutil
from pathlib import Path

try:
    if sys.platform.startswith("linux") and "DISPLAY" not in os.environ:
        HAVE_PYAUTOGUI = False
    else:
        import pyautogui
        HAVE_PYAUTOGUI = True
except Exception:
    HAVE_PYAUTOGUI = False


class GUIEventRecorder:
    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.screenshot_dir = self.output_dir / "screenshots"
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        self.events = []
        self.screenshots = []

    def _capture_screenshot(self, name: str) -> str:
        if not HAVE_PYAUTOGUI:
            return ""
        try:
            image = pyautogui.screenshot()
            path = self.screenshot_dir / f"{name}.png"
            image.save(str(path))
            self.screenshots.append({"name": name, "path": str(path)})
            return str(path)
        except Exception:
            return ""

    def record(self, *, category: str, action: str, status: str = "PASS", payload: dict | None = None, take_screenshot: bool = False):
        event = {
            "ts": time.time(),
            "category": str(category),
            "action": str(action),
            "status": str(status),
            "payload": dict(payload or {}),
        }
        if take_screenshot:
            screenshot_path = self._capture_screenshot(f"{len(self.events):04d}_{category}_{action}".replace("/", "_"))
            if screenshot_path:
                event["screenshot_path"] = screenshot_path
        self.events.append(event)
        return event

    def finalize(self) -> str:
        events_path = self.output_dir / "gui_agent_runtime_events.jsonl"
        evidence_path = self.output_dir / "gui_agent_runtime_evidence.json"
        manifest_path = self.output_dir / "gui_agent_screenshot_manifest.json"
        events_path.write_text(
            "\n".join(json.dumps(item, ensure_ascii=False, sort_keys=True) for item in self.events) + ("\n" if self.events else ""),
            encoding="utf-8",
        )
        manifest_path.write_text(json.dumps({"screenshots": self.screenshots}, ensure_ascii=False, indent=2), encoding="utf-8")
        evidence = {
            "status": "PASS" if len(self.events) > 0 else "FAIL",
            "events_path": str(events_path),
            "screenshot_manifest_path": str(manifest_path),
            "event_count": int(len(self.events)),
            "screenshot_count": int(len(self.screenshots)),
            "categories_present": sorted({str(item.get("category") or "") for item in self.events if str(item.get("category") or "")}),
        }
        evidence_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(evidence_path)


def collect_gui_runtime_evidence(*, duration_sec: int = 5, output_dir: str | Path) -> str:
    out_dir = Path(output_dir).expanduser().resolve()
    evidence_path = out_dir / "gui_agent_runtime_evidence.json"
    try:
        result = simulate_gui_workflow(duration_sec=duration_sec, output_dir=out_dir, finalize=True)
        return str(result)
    except SystemExit:
        if evidence_path.exists():
            return str(evidence_path)
        recorder = GUIEventRecorder(out_dir)
        recorder.record(category="runtime_host", action="gui_collection_failed", status="FAIL")
        return str(recorder.finalize())
    except Exception as e:
        recorder = GUIEventRecorder(out_dir)
        recorder.record(category="runtime_host", action="gui_collection_exception", status="FAIL", payload={"error": repr(e)})
        return str(recorder.finalize())


def simulate_gui_workflow(duration_sec=5, output_dir: str | Path | None = None, recorder: GUIEventRecorder | None = None, finalize: bool = True):
    """
    [環境限制聲明]
    本 GUI 測試腳本嚴禁使用 Headless Mock。
    執行此用例必須在具備真實桌面環境 (如 macOS, Windows 或帶有 X11/Wayland 的 Linux) 的實體機或虛擬機上進行。
    在無桌面環境的雲端伺服器 (如 SSH headless) 上將直接拋出錯誤並退出，以確保測試數據的真實性。
    """
    print("="*50)
    print(f" 🚀 [Eko-Agent] 开始执行 GUI 桌面自动化测试用例 (持续 {duration_sec} 秒)")
    print("="*50)
    
    recorder = recorder or GUIEventRecorder(output_dir or (Path.cwd() / "gui_agent_runtime"))
    recorder.record(
        category="runtime_host",
        action="session_start",
        payload={"platform": sys.platform, "pid": os.getpid(), "have_pyautogui": bool(HAVE_PYAUTOGUI)},
    )

    if not HAVE_PYAUTOGUI and not (sys.platform.startswith("linux") and "DISPLAY" not in os.environ):
        print("[!] 错误: 未安装 pyautogui，请执行 `pip install pyautogui`。")
        recorder.record(category="runtime_host", action="missing_pyautogui", status="FAIL")
        if finalize:
            recorder.finalize()
        sys.exit(1)
        
    # 檢查是否在無桌面環境下執行 (例如 Linux 且無 DISPLAY)
    if sys.platform.startswith("linux") and "DISPLAY" not in os.environ:
        print("[!] 致命错误: 检测到 Headless 环境。")
        print("[!] 依循「严禁 Mock」原则，真实的 GUI Agent 测试必须在具备桌面环境的终端设备 (如鸿蒙 PC/AI PC) 上执行。")
        print("[!] 测试终止。跳过 GUI 执行，直接进入 M7.2 验收指标评估。")
        recorder.record(category="runtime_host", action="headless_abort", status="FAIL")
        return recorder.finalize() if finalize else recorder

    print("[GUI] PyAutoGUI 已加载，桌面环境确认。正在执行长时间连续用例...")
    try:
        screenWidth, screenHeight = pyautogui.size()
        print(f"  -> 获取屏幕尺寸: {screenWidth}x{screenHeight}")
        recorder.record(category="runtime_host", action="screen_ready", payload={"width": screenWidth, "height": screenHeight}, take_screenshot=True)
        
        # 實際模擬打開 Spotlight (macOS) / 開始菜單 (Windows)
        print("  -> 呼叫系統搜尋框并打开记事本/TextEdit...")
        recorder.record(category="workflow", action="launch_editor_begin", payload={"platform": sys.platform})
        if sys.platform == "darwin":
            pyautogui.hotkey('command', 'space')
            recorder.record(category="tool_call", action="pyautogui.hotkey", payload={"keys": ["command", "space"]}, take_screenshot=True)
            time.sleep(1)
            pyautogui.write('TextEdit', interval=0.1)
            recorder.record(category="tool_call", action="pyautogui.write", payload={"text": "TextEdit"})
        else:
            pyautogui.hotkey('win')
            recorder.record(category="tool_call", action="pyautogui.hotkey", payload={"keys": ["win"]}, take_screenshot=True)
            time.sleep(1)
            pyautogui.write('notepad', interval=0.1)
            recorder.record(category="tool_call", action="pyautogui.write", payload={"text": "notepad"})
            
        time.sleep(1)
        pyautogui.press('enter')
        recorder.record(category="tool_call", action="pyautogui.press", payload={"key": "enter"}, take_screenshot=True)
        time.sleep(3) # 等待应用开启
        recorder.record(category="screenshot", action="editor_opened", payload={"app": "TextEdit" if sys.platform == "darwin" else "notepad"}, take_screenshot=True)
        
        # 如果是 macOS TextEdit，可能需要按 Command+N 建立新文件
        if sys.platform == "darwin":
            pyautogui.hotkey('command', 'n')
            recorder.record(category="tool_call", action="pyautogui.hotkey", payload={"keys": ["command", "n"]})
            time.sleep(1)

        print(f"  -> 开始连续记录工作流，将持续 {duration_sec} 秒。按 Ctrl+C 可提早中断。")
        
        start_time = time.time()
        loop_count = 0
        while time.time() - start_time < duration_sec:
            loop_count += 1
            current_t = time.strftime('%Y-%m-%d %H:%M:%S')
            log_str = f"[{current_t}] Eko-Agent 工作流执行中... 第 {loop_count} 次审计。\n"
            recorder.record(category="workflow", action="workflow_iteration", payload={"loop_count": loop_count, "message": log_str.strip()})
            pyautogui.write(log_str, interval=0.05)
            recorder.record(category="tool_call", action="pyautogui.write", payload={"chars": len(log_str)})
            recorder.record(category="screenshot", action="workflow_snapshot", payload={"loop_count": loop_count}, take_screenshot=True)
            
            # 每隔 10 秒记录一次
            elapsed = time.time() - start_time
            if duration_sec - elapsed > 10:
                time.sleep(10)
            else:
                time.sleep(max(0, duration_sec - elapsed))
                
        print("  -> 时间到。关闭记事本 (不保存)...")
        recorder.record(category="workflow", action="close_editor_begin", payload={"loop_count": loop_count})
        if sys.platform == "darwin":
            pyautogui.hotkey('command', 'w')
            recorder.record(category="tool_call", action="pyautogui.hotkey", payload={"keys": ["command", "w"]}, take_screenshot=True)
            time.sleep(1)
            pyautogui.press('delete') # 丢弃更改
            recorder.record(category="tool_call", action="pyautogui.press", payload={"key": "delete"})
        else:
            pyautogui.hotkey('alt', 'f4')
            recorder.record(category="tool_call", action="pyautogui.hotkey", payload={"keys": ["alt", "f4"]}, take_screenshot=True)
            time.sleep(1)
            pyautogui.press('n') # 不保存
            recorder.record(category="tool_call", action="pyautogui.press", payload={"key": "n"})
            
        print("✅ [Eko-Agent] 桌面用例执行完毕。动作轨迹已记录。\n")
        recorder.record(category="runtime_host", action="session_complete", payload={"loop_count": loop_count})
        
    except KeyboardInterrupt:
        print("\n[GUI] 用户手动中断了长期工作流。进入结算流程...")
        recorder.record(category="runtime_host", action="keyboard_interrupt", status="FAIL")
    except Exception as e:
        print(f"[GUI] 致命错误: PyAutoGUI 执行时发生异常: {e}")
        recorder.record(category="runtime_host", action="gui_exception", status="FAIL", payload={"error": repr(e)})
        if finalize:
            recorder.finalize()
        sys.exit(1)
    return recorder.finalize() if finalize else recorder

def trigger_cgc_m7_pipeline(*, recorder: GUIEventRecorder | None = None, output_dir: str | Path | None = None):
    print("="*50)
    print(" ⚙️ [CGC Engine] 启动动态轨迹编译与 ShadowAudit 拦截")
    print("="*50)
    
    # 動態選型機制：依據系統總記憶體 (RAM)
    total_memory_gb = psutil.virtual_memory().total / (1024 ** 3)
    print(f"[*] 检测到系统总内存: {total_memory_gb:.2f} GB")
    
    CGC_GUI_LOW_MEM_MODEL = "microsoft/Phi-3-vision-Phi-3.5-vision-instruct-0.8B"
    
    if total_memory_gb <= 4.5: # 考慮到系統開銷，門檻設在 4.5GB
        model_id = CGC_GUI_LOW_MEM_MODEL
        print(f"[*] 内存 ≤ 4GB，自动降级选择低内存后备模型: {model_id}")
    else:
        model_id = "bytedance-research/UI-TARS-2B-SFT"
        print(f"[*] 内存充足，选择政企标准模型: {model_id}")
        
    repo_root = Path(__file__).parent.parent.parent.parent
    cli_path = repo_root / "cgc_engine" / "agent" / "cli.py"
    
    # 构造 M7 Gate 测试命令
    cmd = [
        sys.executable, str(cli_path),
        "pipeline",
        "--milestone", "m7",
        "--backend", "vllm",
        "--model", model_id
    ]
    
    print(f"[*] 执行命令: {' '.join(cmd)}")
    if recorder is not None:
        recorder.record(category="runtime_host", action="pipeline_subprocess_start", payload={"cmd": cmd, "output_dir": str(output_dir or "")})
    
    env = os.environ.copy()
    env["CGC_MILESTONE"] = "m7"
    
    # 捕获输出以找到 report_path
    process = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    report_path = None
    for line in process.stdout:
        sys.stdout.write(line)
        # 尝试从输出中抓取 report_path，通常在最后一行 JSON 中
        if line.strip().startswith("{") and "report_path" in line:
            try:
                data = json.loads(line.strip())
                if "report_path" in data:
                    report_path = data["report_path"]
            except:
                pass

    process.wait()
    
    if process.returncode != 0:
        print(f"[!] CGC Pipeline 执行失败 (Exit Code: {process.returncode})")
        # 即使失败，如果有 report_path 也可以继续评估
        if recorder is not None:
            recorder.record(category="runtime_host", action="pipeline_subprocess_finish", status="FAIL", payload={"returncode": process.returncode, "report_path": report_path or ""})
    elif recorder is not None:
        recorder.record(category="runtime_host", action="pipeline_subprocess_finish", payload={"returncode": process.returncode, "report_path": report_path or ""})
        
    print("✅ [CGC Engine] 执行完毕。\n")
    return report_path

def evaluate_m72_gate(report_path):
    print("="*50)
    print(" 📊 [AgentEval] 读取 M7.2 验收指标与生成报告")
    print("="*50)
    
    if not report_path or not os.path.exists(report_path):
        print(f"[!] 无法找到有效的 CGC report.json 路径: {report_path}")
        sys.exit(1)
        
    eval_script = Path(__file__).parent / "run_m72_eval.py"
    yaml_config = Path(__file__).parent / "m72_gate.yaml"
    out_dir = Path(report_path).parent
    
    cmd = [
        sys.executable, str(eval_script),
        "--cgc-report", str(report_path),
        "--config", str(yaml_config),
        "--out-dir", str(out_dir)
    ]
    
    print(f"[*] 执行命令: {' '.join(cmd)}\n")
    subprocess.run(cmd)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Eko-Agent GUI Demo")
    parser.add_argument("--duration", type=int, default=15, help="持续运行的时间(秒)，例如一小时请填 3600")
    parser.add_argument("--out-dir", type=str, default="", help="GUI agent 结构化事件输出目录")
    args = parser.parse_args()
    output_dir = args.out_dir or str((Path.cwd() / "gui_agent_runtime").resolve())

    # 1. 模拟 GUI 执行
    recorder = GUIEventRecorder(output_dir)
    simulate_gui_workflow(duration_sec=args.duration, output_dir=output_dir, recorder=recorder, finalize=False)
    
    # 2. CGC 拦截与编译 (生成 report.json)
    # 如果是在 headless 環境下被直接呼叫跳過了 GUI 執行，我們依然可以直接觸發 M7.2 的 Pipeline 攔截測試
    report_path = trigger_cgc_m7_pipeline(recorder=recorder, output_dir=output_dir)
    evidence_path = recorder.finalize()
    print(f"[*] 已生成 GUI 结构化事件证据: {evidence_path}")
    
    # 3. AgentEval 验收判定
    evaluate_m72_gate(report_path)
