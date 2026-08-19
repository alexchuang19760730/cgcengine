
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
AOT (Ahead-of-Time) 预编译器
在模型加载前预先编译热门路径
"""

import threading
from typing import Optional, List, Dict, Any, Callable
import logging

from .jitload_config import JITLoadConfig
from .jitload_manager import JITLoadManager

logger = logging.getLogger(__name__)


class PrecompileTask:
    """预编译任务"""
    
    def __init__(self, model_tag: str, compile_func: Callable, priority: int = 0):
        self.model_tag = model_tag
        self.compile_func = compile_func
        self.priority = priority
        self.result = None
        self.error = None
        self.done = False


class AOTPrecompiler:
    """AOT 预编译器"""
    
    def __init__(self, config: Optional[JITLoadConfig] = None, jitload_manager: Optional[JITLoadManager] = None):
        self.config = config or JITLoadConfig()
        self.jitload = jitload_manager or JITLoadManager(self.config)
        self._tasks: List[PrecompileTask] = []
        self._lock = threading.Lock()
        self._running = False
        self._worker: Optional[threading.Thread] = None
    
    def start(self):
        """启动预编译"""
        if not self.config.enable_aot:
            logger.info("AOT 预编译未启用")
            return
        
        if self._running:
            return
        
        self._running = True
        self._worker = threading.Thread(target=self._worker_loop, name="aot-precompile", daemon=True)
        self._worker.start()
        logger.info("AOT 预编译器已启动")
    
    def stop(self):
        """停止预编译"""
        self._running = False
        if self._worker:
            self._worker.join(timeout=10.0)
        logger.info("AOT 预编译器已停止")
    
    def submit_task(self, model_tag: str, compile_func: Callable, priority: int = 0):
        """提交预编译任务"""
        task = PrecompileTask(model_tag, compile_func, priority)
        
        with self._lock:
            self._tasks.append(task)
            # 按优先级排序
            self._tasks.sort(key=lambda t: -t.priority)
        
        logger.debug(f"预编译任务已提交: {model_tag}")
        return task
    
    def _worker_loop(self):
        """工作线程循环"""
        while self._running:
            task = None
            
            # 获取任务
            with self._lock:
                if self._tasks:
                    task = self._tasks.pop(0)
            
            if task:
                try:
                    logger.info(f"开始预编译: {task.model_tag}")
                    result = task.compile_func()
                    task.result = result
                    task.done = True
                    logger.info(f"预编译完成: {task.model_tag}")
                except Exception as e:
                    logger.error(f"预编译失败: {task.model_tag}, {e}")
                    task.error = e
                    task.done = True
            else:
                # 没有任务，等待
                import time
    
    def precompile_hot_paths(self, model_tags: List[str]):
        """预编译热门路径"""
        for tag in model_tags:
            # 这里应该创建真实的编译函数
            # 简化版本，只是占位
            def compile_func(tag=tag):
                logger.debug(f"模拟编译: {tag}")
                import time
                return {"compiled": True, "tag": tag}
            
            self.submit_task(tag, compile_func, priority=10)
    
    def wait_for_task(self, task: PrecompileTask, timeout: Optional[float] = None) -> Any:
        """等待任务完成"""
        import time
        start = time.time()
        
        while not task.done:
            if timeout and (time.time() - start) > timeout:
                raise TimeoutError(f"任务超时: {task.model_tag}")
        
        if task.error:
            raise task.error
        
        return task.result

