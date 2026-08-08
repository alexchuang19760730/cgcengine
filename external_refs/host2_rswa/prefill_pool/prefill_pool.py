# Copyright (c) 2026 SandAI. All Rights Reserved.

import torch
import os
import hashlib
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple
from pathlib import Path
from threading import Lock
from collections import OrderedDict

from ..gds_service.cufile_wrapper import cuFileRead, cuFileWrite, is_gds_available


@dataclass
class ChunkMetadata:
    chunk_id: str
    token_count: int
    ref_k_shape: Tuple[int, int, int, int]  # (B, H, T, D)
    ref_v_shape: Tuple[int, int, int, int]
    access_time: int = 0
    is_hot: bool = False
    file_path: Optional[str] = None


@dataclass
class Chunk:
    chunk_id: str
    token_ids: torch.Tensor
    ref_k: Optional[torch.Tensor] = None
    ref_v: Optional[torch.Tensor] = None
    metadata: Optional[ChunkMetadata] = None


class PrefillPool:
    """
    Prefill Pool - 实现真正 Unlimited 无限上下文的核心模块
    
    功能：
    1. 将超长参考文本切分为 Chunk 块
    2. 热块常驻显存，冷块卸载到 GDS/NFS
    3. 按需自动换入换出，不触发全量重 Prefill
    4. 与 R-SWA Reference KV 直接对接
    """
    
    def __init__(
        self,
        max_hot_chunks: int = 4,
        chunk_size: int = 8192,
        storage_path: str = "/data/nfs/prefill_pool",
        enable_gds: bool = True,
    ):
        self.max_hot_chunks = max_hot_chunks
        self.chunk_size = chunk_size
        self.storage_path = Path(storage_path)
        self.enable_gds = enable_gds and is_gds_available()
        
        # 热块缓存（显存）
        self.hot_chunks: Dict[str, Chunk] = OrderedDict()
        # 冷块元数据（内存）
        self.cold_metadata: Dict[str, ChunkMetadata] = {}
        # 全局步骤计数器
        self.step = 0
        # 锁
        self._lock = Lock()
        
        # 确保存储目录存在
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # 日志
        print(f"[PrefillPool] 初始化完成")
        print(f"[PrefillPool] 最大热块数: {max_hot_chunks}")
        print(f"[PrefillPool] 块大小: {chunk_size} tokens")
        print(f"[PrefillPool] 存储路径: {storage_path}")
        print(f"[PrefillPool] GDS 直写: {'✅ 启用' if self.enable_gds else '❌ 禁用'}")
    
    def _generate_chunk_id(self, data: str) -> str:
        """生成唯一的 Chunk ID"""
        return hashlib.md5(data.encode()).hexdigest()[:16]
    
    def _chunk_file_path(self, chunk_id: str) -> Path:
        """获取 Chunk 文件路径"""
        return self.storage_path / f"{chunk_id}.pt"
    
    def _save_chunk_to_storage(self, chunk: Chunk) -> bool:
        """将 Chunk 保存到存储（使用 GDS 直写显存）"""
        if chunk.ref_k is None or chunk.ref_v is None:
            return False
        
        try:
            file_path = str(self._chunk_file_path(chunk.chunk_id))
            
            # 使用 GDS 直写显存
            if self.enable_gds and chunk.ref_k.device.type == 'cuda':
                # 保存 ref_k
                cuFileWrite(f"{file_path}.k", chunk.ref_k)
                # 保存 ref_v
                cuFileWrite(f"{file_path}.v", chunk.ref_v)
                # 保存 token_ids
                cuFileWrite(f"{file_path}.tokens", chunk.token_ids)
                print(f"[PrefillPool] ✅ GDS 直写 Chunk {chunk.chunk_id}")
            else:
                # 回退到标准方式
                torch.save({
                    'ref_k': chunk.ref_k.cpu(),
                    'ref_v': chunk.ref_v.cpu(),
                    'token_ids': chunk.token_ids.cpu(),
                }, file_path)
            
            return True
        except Exception as e:
            print(f"[PrefillPool] ❌ 保存 Chunk 失败: {e}")
            return False
    
    def _load_chunk_from_storage(self, chunk_id: str, device: str = "cuda") -> Optional[Chunk]:
        """从存储加载 Chunk（使用 GDS 直读显存）"""
        try:
            file_path = str(self._chunk_file_path(chunk_id))
            metadata = self.cold_metadata.get(chunk_id)
            
            if not metadata:
                return None
            
            chunk = Chunk(chunk_id=chunk_id, token_ids=None, ref_k=None, ref_v=None)
            
            # 使用 GDS 直读显存
            if self.enable_gds:
                # 读取 ref_k
                ref_k = torch.empty(metadata.ref_k_shape, device=device, dtype=torch.bfloat16)
                cuFileRead(f"{file_path}.k", ref_k)
                chunk.ref_k = ref_k
                
                # 读取 ref_v
                ref_v = torch.empty(metadata.ref_v_shape, device=device, dtype=torch.bfloat16)
                cuFileRead(f"{file_path}.v", ref_v)
                chunk.ref_v = ref_v
                
                # 读取 token_ids
                token_ids = torch.empty((metadata.token_count,), device=device, dtype=torch.long)
                cuFileRead(f"{file_path}.tokens", token_ids)
                chunk.token_ids = token_ids
                
                print(f"[PrefillPool] ✅ GDS 直读 Chunk {chunk_id}")
            else:
                # 回退到标准方式
                data = torch.load(file_path, map_location=device)
                chunk.ref_k = data['ref_k'].to(device)
                chunk.ref_v = data['ref_v'].to(device)
                chunk.token_ids = data['token_ids'].to(device)
            
            chunk.metadata = metadata
            return chunk
        except Exception as e:
            print(f"[PrefillPool] ❌ 加载 Chunk 失败: {e}")
            return None
    
    def _evict_cold(self) -> Optional[str]:
        """
        驱逐最冷的块到冷存储
        返回被驱逐的 chunk_id，如果没有驱逐则返回 None
        """
        with self._lock:
            if len(self.hot_chunks) < self.max_hot_chunks:
                return None
            
            # 找到访问时间最早的块
            oldest_chunk_id = None
            oldest_time = float('inf')
            
            for chunk_id, chunk in self.hot_chunks.items():
                if chunk.metadata and chunk.metadata.access_time < oldest_time:
                    oldest_time = chunk.metadata.access_time
                    oldest_chunk_id = chunk_id
            
            if oldest_chunk_id:
                # 保存到冷存储
                chunk = self.hot_chunks[oldest_chunk_id]
                if self._save_chunk_to_storage(chunk):
                    # 更新元数据
                    if chunk.metadata:
                        chunk.metadata.is_hot = False
                        self.cold_metadata[oldest_chunk_id] = chunk.metadata
                    # 从热缓存移除
                    del self.hot_chunks[oldest_chunk_id]
                    print(f"[PrefillPool] 驱逐 Chunk {oldest_chunk_id} 到冷存储")
                else:
                    print(f"[PrefillPool] ❌ 驱逐失败，保留在热缓存")
                    oldest_chunk_id = None
            
            return oldest_chunk_id
    
    def add_hot_chunk(self, token_ids: torch.Tensor, ref_k: torch.Tensor, ref_v: torch.Tensor) -> str:
        """
        添加热块到显存
        返回 chunk_id
        """
        with self._lock:
            self.step += 1
            
            # 生成唯一 ID
            chunk_data = f"{token_ids.shape[0]}_{token_ids.dtype}_{ref_k.shape}"
            chunk_id = self._generate_chunk_id(chunk_data)
            
            # 检查是否已存在
            if chunk_id in self.hot_chunks:
                # 更新访问时间
                self.hot_chunks[chunk_id].metadata.access_time = self.step
                return chunk_id
            
            # 驱逐冷块（如果需要）
            self._evict_cold()
            
            # 创建元数据
            metadata = ChunkMetadata(
                chunk_id=chunk_id,
                token_count=token_ids.shape[0],
                ref_k_shape=ref_k.shape,
                ref_v_shape=ref_v.shape,
                access_time=self.step,
                is_hot=True,
                file_path=str(self._chunk_file_path(chunk_id)),
            )
            
            # 创建 Chunk
            chunk = Chunk(
                chunk_id=chunk_id,
                token_ids=token_ids.contiguous(),
                ref_k=ref_k.contiguous(),
                ref_v=ref_v.contiguous(),
                metadata=metadata,
            )
            
            # 添加到热缓存
            self.hot_chunks[chunk_id] = chunk
            
            print(f"[PrefillPool] 添加热块 {chunk_id} (tokens: {token_ids.shape[0]})")
            return chunk_id
    
    def load_chunk(self, chunk_id: str, device: str = "cuda") -> Optional[Chunk]:
        """
        加载 Chunk（优先从热缓存，否则从冷存储换入）
        返回 Chunk 对象，如果不存在则返回 None
        """
        with self._lock:
            self.step += 1
            
            # 1. 检查热缓存
            if chunk_id in self.hot_chunks:
                chunk = self.hot_chunks[chunk_id]
                chunk.metadata.access_time = self.step
                print(f"[PrefillPool] 命中热缓存 {chunk_id}")
                return chunk
            
            # 2. 检查冷存储
            if chunk_id in self.cold_metadata:
                # 驱逐一个冷块
                self._evict_cold()
                
                # 从冷存储加载
                chunk = self._load_chunk_from_storage(chunk_id, device)
                if chunk:
                    # 更新元数据
                    chunk.metadata.access_time = self.step
                    chunk.metadata.is_hot = True
                    
                    # 添加到热缓存
                    self.hot_chunks[chunk_id] = chunk
                    
                    # 从冷元数据移除
                    del self.cold_metadata[chunk_id]
                    
                    print(f"[PrefillPool] 从冷存储换入 {chunk_id}")
                    return chunk
            
            return None
    
    def get_all_ref_kv(self, device: str = "cuda") -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        获取所有热块的 Reference KV
        返回 (ref_k, ref_v) 拼接后的张量
        """
        with self._lock:
            ref_ks = []
            ref_vs = []
            
            for chunk in self.hot_chunks.values():
                if chunk.ref_k is not None and chunk.ref_v is not None:
                    ref_ks.append(chunk.ref_k)
                    ref_vs.append(chunk.ref_v)
            
            if not ref_ks:
                return None, None
            
            # 拼接所有热块的 KV
            ref_k = torch.cat(ref_ks, dim=2)
            ref_v = torch.cat(ref_vs, dim=2)
            
            print(f"[PrefillPool] 拼接 {len(ref_ks)} 个热块, 总长度: {ref_k.shape[2]}")
            return ref_k.to(device), ref_v.to(device)
    
    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        """获取 Chunk（不更新访问时间）"""
        with self._lock:
            return self.hot_chunks.get(chunk_id)
    
    def remove_chunk(self, chunk_id: str) -> bool:
        """移除 Chunk"""
        with self._lock:
            # 从热缓存移除
            if chunk_id in self.hot_chunks:
                del self.hot_chunks[chunk_id]
            # 从冷元数据移除
            if chunk_id in self.cold_metadata:
                del self.cold_metadata[chunk_id]
            # 删除文件
            try:
                os.remove(self._chunk_file_path(chunk_id))
                os.remove(f"{self._chunk_file_path(chunk_id)}.k")
                os.remove(f"{self._chunk_file_path(chunk_id)}.v")
                os.remove(f"{self._chunk_file_path(chunk_id)}.tokens")
            except:
                pass
            return True
    
    def clear(self):
        """清空所有数据"""
        with self._lock:
            self.hot_chunks.clear()
            self.cold_metadata.clear()
            # 删除所有文件
            for f in self.storage_path.glob("*.pt*"):
                try:
                    f.unlink()
                except:
                    pass
            print("[PrefillPool] 已清空所有数据")
    
    def info(self) -> Dict:
        """返回 Pool 状态信息"""
        with self._lock:
            hot_tokens = sum(c.token_ids.shape[0] for c in self.hot_chunks.values())
            cold_tokens = sum(m.token_count for m in self.cold_metadata.values())
            
            return {
                "step": self.step,
                "hot_chunks": len(self.hot_chunks),
                "cold_chunks": len(self.cold_metadata),
                "hot_tokens": hot_tokens,
                "cold_tokens": cold_tokens,
                "max_hot_chunks": self.max_hot_chunks,
                "chunk_size": self.chunk_size,
                "gds_enabled": self.enable_gds,
                "storage_path": str(self.storage_path),
            }
