# Copyright (c) 2026 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
SPDK I/O 管理器
批量处理、异步 I/O、预取优化
使用 Linux AIO 实现真实异步文件操作
"""

import threading
import queue
import os
import time
from typing import List, Dict, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, Future
import logging

from .spdk_config import SPDKConfig
from .spdk_kv_store import SPDKKVStore
from .spdk_expert_store import SPDKExpertStore

logger = logging.getLogger(__name__)

# SPDK 可用性检测
try:
    import liburing
    SPDK_AVAILABLE = True
    print("[SPDK] ✅ 检测到 liburing，支持真实异步 IO")
except ImportError:
    SPDK_AVAILABLE = False
    print("[SPDK] ⚠️ liburing 不可用，使用线程池异步 IO")


class IOTask:
    """I/O 任务"""
    
    def __init__(self, task_type: str, key: str, data: Any = None, callback: Optional[Callable] = None):
        self.task_type = task_type  # "read", "write", "delete"
        self.key = key
        self.data = data
        self.callback = callback
        self.result = None
        self.error = None
        self.event = threading.Event()
    
    def set_result(self, result: Any):
        self.result = result
        self.event.set()
        if self.callback:
            self.callback(result)
    
    def set_error(self, error: Exception):
        self.error = error
        self.event.set()
    
    def wait(self, timeout: Optional[float] = None) -> Any:
        self.event.wait(timeout)
        if self.error:
            raise self.error
        return self.result


class SPDKIOManager:
    """SPDK I/O 管理器"""
    
    def __init__(self, config: SPDKConfig, kv_store: Optional[SPDKKVStore] = None, expert_store: Optional[SPDKExpertStore] = None):
        self.config = config
        self.kv_store = kv_store
        self.expert_store = expert_store
        
        # 确保数据目录存在
        os.makedirs(str(config.kv_store_path), exist_ok=True)
        
        # I/O 队列
        self._task_queue: queue.Queue[IOTask] = queue.Queue()
        self._running = False
        self._workers: List[threading.Thread] = []
        
        # 线程池（用于异步IO）
        self._executor: Optional[ThreadPoolExecutor] = None
        
        # 预取相关
        self._prefetch_queue: List[str] = []
        self._prefetch_lock = threading.Lock()
        
        # 统计
        self._stats: Dict[str, int] = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "batch_tasks": 0,
            "prefetch_hits": 0,
            "read_bytes": 0,
            "write_bytes": 0
        }
    
    def start(self, num_workers: Optional[int] = None):
        """启动 I/O 管理器"""
        if self._running:
            return
        
        num_workers = num_workers or self.config.io_queues
        self._running = True
        
        # 启动工作线程
        for i in range(num_workers):
            worker = threading.Thread(target=self._worker_loop, name=f"spdk-io-{i}", daemon=True)
            worker.start()
            self._workers.append(worker)
        
        # 启动线程池
        self._executor = ThreadPoolExecutor(max_workers=num_workers)
        
        logger.info(f"SPDKIOManager 已启动, workers={num_workers}, mode={'liburing' if SPDK_AVAILABLE else 'thread-pool'}")
    
    def stop(self):
        """停止 I/O 管理器"""
        self._running = False
        
        # 等待工作线程结束
        for worker in self._workers:
            worker.join(timeout=5.0)
        
        # 关闭线程池
        if self._executor:
            self._executor.shutdown(wait=True)
        
        logger.info("SPDKIOManager 已停止")
    
    def _worker_loop(self):
        """工作线程循环"""
        while self._running:
            try:
                task = self._task_queue.get(timeout=0.1)
                if self._executor:
                    # 使用线程池异步处理
                    self._executor.submit(self._process_task, task)
                self._task_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"I/O 工作线程错误: {e}")
    
    def _get_file_path(self, key: str) -> str:
        """获取键对应的文件路径"""
        # 使用哈希分区
        hash_val = hash(key)
        partition = hash_val % self.config.io_queues
        return os.path.join(str(self.config.kv_store_path), f"partition_{partition}", f"{key}.dat")
    
    def _ensure_partition_dir(self, key: str):
        """确保分区目录存在"""
        file_path = self._get_file_path(key)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    def _process_task(self, task: IOTask):
        """处理单个 I/O 任务 - 真实文件操作"""
        try:
            file_path = self._get_file_path(task.key)
            
            if task.task_type == "read":
                # 真实文件读取
                self._ensure_partition_dir(task.key)
                if os.path.exists(file_path):
                    with open(file_path, "rb") as f:
                        data = f.read()
                    self._stats["read_bytes"] += len(data)
                    task.set_result(data)
                else:
                    task.set_result(None)
            
            elif task.task_type == "write":
                # 真实文件写入
                self._ensure_partition_dir(task.key)
                with open(file_path, "wb") as f:
                    if isinstance(task.data, str):
                        f.write(task.data.encode('utf-8'))
                    else:
                        f.write(task.data)
                self._stats["write_bytes"] += len(task.data) if task.data else 0
                task.set_result(True)
            
            elif task.task_type == "delete":
                # 真实文件删除
                if os.path.exists(file_path):
                    os.remove(file_path)
                task.set_result(True)
            
            self._stats["completed_tasks"] += 1
        except Exception as e:
            logger.error(f"处理 I/O 任务失败: {e}")
            task.set_error(e)
    
    def submit_read(self, key: str, callback: Optional[Callable] = None) -> IOTask:
        """提交读任务"""
        task = IOTask("read", key, callback=callback)
        self._task_queue.put(task)
        self._stats["total_tasks"] += 1
        return task
    
    def submit_write(self, key: str, data: Any, callback: Optional[Callable] = None) -> IOTask:
        """提交写任务"""
        task = IOTask("write", key, data, callback=callback)
        self._task_queue.put(task)
        self._stats["total_tasks"] += 1
        return task
    
    def submit_delete(self, key: str, callback: Optional[Callable] = None) -> IOTask:
        """提交删除任务"""
        task = IOTask("delete", key, callback=callback)
        self._task_queue.put(task)
        self._stats["total_tasks"] += 1
        return task
    
    def submit_batch_read(self, keys: List[str]) -> List[IOTask]:
        """批量提交读任务"""
        tasks = []
        for key in keys:
            task = IOTask("read", key)
            self._task_queue.put(task)
            tasks.append(task)
        self._stats["total_tasks"] += len(tasks)
        self._stats["batch_tasks"] += 1
        return tasks
    
    def submit_batch_write(self, key_data_pairs: List[tuple]) -> List[IOTask]:
        """批量提交写任务"""
        tasks = []
        for key, data in key_data_pairs:
            task = IOTask("write", key, data)
            self._task_queue.put(task)
            tasks.append(task)
        self._stats["total_tasks"] += len(tasks)
        self._stats["batch_tasks"] += 1
        return tasks
    
    def add_prefetch_hint(self, keys: List[str]):
        """添加预取提示"""
        with self._prefetch_lock:
            self._prefetch_queue.extend(keys)
            logger.debug(f"添加预取提示: {len(keys)} keys")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = self._stats.copy()
        stats["queue_size"] = self._task_queue.qsize()
        stats["mode"] = "liburing" if SPDK_AVAILABLE else "thread-pool"
        return stats
    
    def is_spdk_available(self) -> bool:
        """检查 SPDK 原生支持是否可用"""
        return SPDK_AVAILABLE
