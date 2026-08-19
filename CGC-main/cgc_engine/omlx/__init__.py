# Copyright (c) 2025 SandAI. All Rights Reserved.
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
oMLX - 专家激活预测与缓存调度

调度层职责：
- 专家激活预测（预测哪些专家会被调用）
- 两级缓存调度（显存 + SSD）
- LRU/FIFO 淘汰策略
"""

from .client import OMLXClient

__version__ = "0.1.0"
__all__ = ["OMLXClient"]