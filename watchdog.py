#!/usr/bin/env python3
"""
CGC Watchdog — Auto-operations daemon.

Runs alongside the load balancer and handles:
1. Log rotation (prevent disk fill — the 26GB nixl_edge.log incident)
2. Disk space monitoring + auto-cleanup
3. GPU temperature monitoring + alerts
4. sglang process monitoring (zombie detection)
5. System resource monitoring (CPU, RAM, swap)
6. Alert logging to file + optional webhook

Usage:
  python3 watchdog.py --interval 30 --log-dir /tmp --max-log-size 500MB

Or as a systemd service:
  [Unit]
  Description=CGC Watchdog
  After=network.target
  [Service]
  ExecStart=/usr/bin/python3 /opt/cgc/watchdog.py
  Restart=always
  [Install]
  WantedBy=multi-user.target
"""
import argparse
import asyncio
import os
import sys
import time
import shutil
import logging
from logging.handlers import RotatingFileHandler
import subprocess
from pathlib import Path
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        RotatingFileHandler("/var/log/cgc_watchdog.log", maxBytes=10_000_000, backupCount=3),
    ],
)
log = logging.getLogger("watchdog")


class Watchdog:
    def __init__(self, args):
        self.interval = args.interval
        self.log_dirs = [d.strip() for d in args.log_dirs.split(",") if d.strip()]
        self.max_log_size = self._parse_size(args.max_log_size)
        self.disk_threshold = args.disk_threshold  # percent
        self.gpu_temp_threshold = args.gpu_temp_threshold  # Celsius
        self.gpu_ports = [int(p) for p in args.ports.split(",")] if args.ports else []
        self.webhook_url = args.webhook_url
        self.cleanup_dirs = [d.strip() for d in args.cleanup_dirs.split(",") if d.strip()]
        self.alert_cooldown = defaultdict(float)  # key -> last_alert_time
        self.alert_cooldown_s = args.alert_cooldown

    @staticmethod
    def _parse_size(s):
        s = s.upper().strip()
        if s.endswith("MB"):
            return int(s[:-2]) * 1_000_000
        elif s.endswith("GB"):
            return int(s[:-2]) * 1_000_000_000
        elif s.endswith("KB"):
            return int(s[:-2]) * 1_000
        return int(s)

    async def run(self):
        log.info("=" * 60)
        log.info("CGC Watchdog started")
        log.info(f"  Interval: {self.interval}s")
        log.info(f"  Log dirs: {self.log_dirs}")
        log.info(f"  Max log size: {self.max_log_size / 1e6:.0f}MB")
        log.info(f"  Disk threshold: {self.disk_threshold}%")
        log.info(f"  GPU temp threshold: {self.gpu_temp_threshold}°C")
        log.info(f"  sglang ports: {self.gpu_ports}")
        log.info("=" * 60)

        while True:
            try:
                await self._check_logs()
                await self._check_disk()
                await self._check_gpu()
                await self._check_processes()
                await self._check_system()
            except Exception as e:
                log.error(f"Watchdog cycle error: {e}", exc_info=True)
            await asyncio.sleep(self.interval)

    # --------------------------------------------------------
    #  Log Rotation
    # --------------------------------------------------------
    async def _check_logs(self):
        """Rotate logs that exceed max size."""
        for log_dir in self.log_dirs:
            if not os.path.isdir(log_dir):
                continue
            for entry in os.scandir(log_dir):
                if not entry.is_file():
                    continue
                if not entry.name.endswith(".log"):
                    continue
                try:
                    size = entry.stat().st_size
                    if size > self.max_log_size:
                        old_path = entry.path + ".old"
                        # Remove old backup if exists
                        if os.path.exists(old_path):
                            os.remove(old_path)
                        os.rename(entry.path, old_path)
                        saved_mb = size / 1e6
                        log.info(f"Log rotated: {entry.name} ({saved_mb:.1f}MB -> {entry.name}.old)")
                        # Truncate the .old to last 1MB for debugging
                        if os.path.exists(old_path):
                            with open(old_path, "r+b") as f:
                                f.seek(-1_000_000, 2) if os.path.getsize(old_path) > 1_000_000 else f.seek(0)
                                tail = f.read()
                            with open(old_path, "wb") as f:
                                f.write(tail)
                except Exception as e:
                    log.warning(f"Failed to rotate {entry.path}: {e}")

    # --------------------------------------------------------
    #  Disk Space
    # --------------------------------------------------------
    async def _check_disk(self):
        """Monitor disk usage and auto-clean if threshold exceeded."""
        for mount in ["/", "/data", "/tmp"]:
            try:
                usage = shutil.disk_usage(mount)
                pct = usage.used / usage.total * 100
                free_gb = usage.free / 1e9

                if pct > self.disk_threshold:
                    self._alert(
                        f"disk_{mount}",
                        f"Disk {mount} at {pct:.1f}% (free: {free_gb:.1f}GB)",
                        level="WARNING",
                    )
                    # Auto-cleanup
                    await self._auto_cleanup()
                elif pct > self.disk_threshold - 10:
                    log.debug(f"Disk {mount}: {pct:.1f}% used, {free_gb:.1f}GB free")
            except Exception:
                pass

    async def _auto_cleanup(self):
        """Clean up known large temporary files."""
        cleaned = 0
        for d in self.cleanup_dirs:
            if not os.path.isdir(d):
                continue
            try:
                for entry in os.scandir(d):
                    if entry.is_file() and entry.name.endswith((".log.old", ".tmp", ".pid")):
                        size = entry.stat().st_size
                        os.remove(entry.path)
                        cleaned += size
                        log.info(f"Cleanup: removed {entry.path} ({size/1e6:.1f}MB)")
                    elif entry.is_file() and entry.name.endswith(".log") and entry.stat().st_size > 100_000_000:
                        # Truncate logs over 100MB
                        size = entry.stat().st_size
                        with open(entry.path, "w") as f:
                            f.write(f"[truncated by watchdog at {time.strftime('%Y-%m-%d %H:%M:%S')}, was {size/1e6:.1f}MB]\n")
                        cleaned += size
                        log.info(f"Cleanup: truncated {entry.path} ({size/1e6:.1f}MB)")
            except Exception as e:
                log.warning(f"Cleanup failed for {d}: {e}")

        if cleaned > 0:
            log.info(f"Auto-cleanup freed {cleaned / 1e6:.1f}MB")
            self._send_alert(f"Auto-cleanup freed {cleaned / 1e6:.1f}MB disk space", level="INFO")

    # --------------------------------------------------------
    #  GPU Monitoring
    # --------------------------------------------------------
    async def _check_gpu(self):
        """Monitor GPU temperature, memory, and utilization."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "nvidia-smi",
                "--query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw",
                "--format=csv,noheader,nounits",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)

            for line in stdout.decode().strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 7:
                    continue
                gpu_id = parts[0]
                name = parts[1]
                temp = float(parts[2]) if parts[2] != "[N/A]" else 0
                util = float(parts[3]) if parts[3] != "[N/A]" else 0
                mem_used = float(parts[4]) if parts[4] != "[N/A]" else 0
                mem_total = float(parts[5]) if parts[5] != "[N/A]" else 0
                power = float(parts[6]) if parts[6] != "[N/A]" else 0

                mem_pct = mem_used / mem_total * 100 if mem_total > 0 else 0

                if temp > self.gpu_temp_threshold:
                    self._alert(
                        f"gpu_temp_{gpu_id}",
                        f"GPU {gpu_id} ({name}) temperature {temp}°C > {self.gpu_temp_threshold}°C "
                        f"(util={util}%, mem={mem_pct:.0f}%, power={power}W)",
                        level="WARNING",
                    )
                elif temp > self.gpu_temp_threshold - 10:
                    log.debug(f"GPU {gpu_id} warm: {temp}°C (util={util}%, power={power}W)")

        except Exception as e:
            log.debug(f"GPU check failed: {e}")

    # --------------------------------------------------------
    #  Process Monitoring
    # --------------------------------------------------------
    async def _check_processes(self):
        """Monitor sglang processes for zombies or crashes."""
        try:
            # Count sglang processes
            proc = await asyncio.create_subprocess_exec(
                "bash", "-c", "ps aux | grep 'sglang.launch_server' | grep -v grep | wc -l",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
            count = int(stdout.decode().strip())

            expected = len(self.gpu_ports)
            if expected > 0 and count < expected:
                self._alert(
                    "sglang_count",
                    f"sglang instances: {count}/{expected} running ({expected - count} missing!)",
                    level="CRITICAL",
                )
            else:
                log.debug(f"sglang processes: {count}/{expected}")

            # Check for zombie processes
            proc = await asyncio.create_subprocess_exec(
                "bash", "-c", "ps aux | grep -w Z | grep -v grep | wc -l",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
            zombies = int(stdout.decode().strip())
            if zombies > 0:
                self._alert(
                    "zombies",
                    f"{zombies} zombie processes detected",
                    level="WARNING",
                )

        except Exception as e:
            log.debug(f"Process check failed: {e}")

    # --------------------------------------------------------
    #  System Resources
    # --------------------------------------------------------
    async def _check_system(self):
        """Monitor CPU, RAM, swap."""
        try:
            # Load average
            load1, load5, load15 = os.getloadavg()
            cpu_count = os.cpu_count() or 1
            if load1 > cpu_count * 2:
                self._alert(
                    "load",
                    f"High load average: {load1:.1f} (CPUs={cpu_count})",
                    level="WARNING",
                )

            # Memory
            proc = await asyncio.create_subprocess_exec(
                "bash", "-c", "free -m | grep -E '^Mem|^Swap'",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
            for line in stdout.decode().strip().split("\n"):
                parts = line.split()
                if len(parts) >= 3 and parts[0] in ("Mem:", "Swap:"):
                    total = int(parts[1])
                    used = int(parts[2])
                    pct = used / total * 100 if total > 0 else 0
                    if pct > 90:
                        self._alert(
                            f"mem_{parts[0].rstrip(':')}",
                            f"{parts[0]} usage {pct:.0f}% ({used}/{total}MB)",
                            level="WARNING",
                        )
        except Exception as e:
            log.debug(f"System check failed: {e}")

    # --------------------------------------------------------
    #  Alerting
    # --------------------------------------------------------
    def _alert(self, key, message, level="WARNING"):
        """Send alert with cooldown to prevent spam."""
        now = time.time()
        last = self.alert_cooldown.get(key, 0)
        if now - last < self.alert_cooldown_s:
            return  # Still in cooldown

        self.alert_cooldown[key] = now
        getattr(log, level.lower(), log.warning)(f"[ALERT:{key}] {message}")

        # Write to alert log
        alert_file = "/var/log/cgc_alerts.log"
        try:
            with open(alert_file, "a") as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [{level}] [{key}] {message}\n")
        except Exception:
            pass

        # Send webhook if configured
        if self.webhook_url:
            asyncio.create_task(self._send_webhook(message, level))

    async def _send_webhook(self, message, level):
        """Send alert to webhook (e.g., Slack, Discord, DingTalk)."""
        try:
            import aiohttp
            payload = {
                "text": f"[CGC Watchdog {level}] {message}",
                "level": level,
                "host": os.uname().nodename,
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            async with aiohttp.ClientSession() as session:
                async with session.post(self.webhook_url, json=payload, timeout=5) as resp:
                    if resp.status != 200:
                        log.debug(f"Webhook returned {resp.status}")
        except Exception as e:
            log.debug(f"Webhook failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="CGC Watchdog — Auto-operations daemon")
    parser.add_argument("--interval", type=int, default=30, help="Check interval (seconds)")
    parser.add_argument("--log-dirs", type=str, default="/tmp,/var/log", help="Comma-separated log directories to rotate")
    parser.add_argument("--max-log-size", type=str, default="500MB", help="Max log file size before rotation")
    parser.add_argument("--disk-threshold", type=int, default=85, help="Disk usage % threshold for alert+cleanup")
    parser.add_argument("--gpu-temp-threshold", type=int, default=85, help="GPU temperature alert threshold (°C)")
    parser.add_argument("--ports", type=str, default="30000,30001,30002,30003,30004,30005,30006,30007", help="Expected sglang ports")
    parser.add_argument("--cleanup-dirs", type=str, default="/tmp", help="Dirs to auto-clean on disk full")
    parser.add_argument("--webhook-url", type=str, default="", help="Webhook URL for alerts (Slack/Discord/DingTalk)")
    parser.add_argument("--alert-cooldown", type=int, default=300, help="Min seconds between same alert")
    args = parser.parse_args()

    wd = Watchdog(args)
    asyncio.run(wd.run())


if __name__ == "__main__":
    main()
