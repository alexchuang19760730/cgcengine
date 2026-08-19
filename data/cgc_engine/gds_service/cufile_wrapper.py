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
cufile_wrapper.py - GDS 核心封装：零拷贝 GPU ↔ SSD

支持两种模式：
1. cuda.bindings.cufile - Python 绑定方式（优先）
2. ctypes 直接调用 libcufile.so - 备用方式
"""

import os
from pathlib import Path
from typing import Dict, List

import torch

# GDS 可用性标志
CUFILE_AVAILABLE = False
CUFILE_BACKEND = "none"  # "cuda_bindings", "ctypes", "none"

# 全局 cufile 模块引用
_cufile = None
_CUDA_BINDINGS_DRIVER_READY = False
_CUFILE_IO_CHUNK_BYTES = 4 * 1024 * 1024


def _env_enabled(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _read_proc_mounts() -> List[Dict[str, str]]:
    mounts: List[Dict[str, str]] = []
    mounts_path = Path("/proc/mounts")
    if not mounts_path.exists():
        return mounts
    try:
        for line in mounts_path.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.split()
            if len(parts) < 4:
                continue
            mounts.append(
                {
                    "source": parts[0],
                    "target": parts[1],
                    "fstype": parts[2],
                    "options": parts[3],
                }
            )
    except Exception:
        return []
    return mounts


def _detect_gds_storage_capabilities() -> Dict[str, object]:
    has_nvme = any(os.path.exists(f"/dev/nvme{n}") for n in range(10)) or any(
        os.path.exists(f"/dev/nvme{n}n1") for n in range(10)
    )
    mounts = _read_proc_mounts()
    nfs_mounts = [
        mount
        for mount in mounts
        if mount["fstype"] in {"nfs", "nfs4"}
    ]
    nfs_rdma_mounts = [
        mount
        for mount in nfs_mounts
        if "rdma" in mount["options"] or "proto=rdma" in mount["options"]
    ]
    proc_modules = Path("/proc/modules")
    module_names: set[str] = set()
    if proc_modules.exists():
        try:
            for line in proc_modules.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.strip():
                    module_names.add(line.split()[0])
        except Exception:
            module_names = set()
    rdma_modules_present = any(
        module in module_names
        for module in ("rpcrdma", "xprtrdma", "svcrdma", "mlx5_ib")
    )
    rdma_devices: List[str] = []
    infiniband_path = Path("/sys/class/infiniband")
    if infiniband_path.exists():
        try:
            rdma_devices = sorted(
                entry.name
                for entry in infiniband_path.iterdir()
                if entry.is_dir()
            )
        except Exception:
            rdma_devices = []
    return {
        "has_nvme": has_nvme,
        "nfs_mounts": nfs_mounts,
        "nfs_rdma_mounts": nfs_rdma_mounts,
        "rdma_modules_present": rdma_modules_present,
        "rdma_devices": rdma_devices,
        "storage_path_eligible": has_nvme or bool(nfs_rdma_mounts),
    }


def _init_cufile():
    """初始化 GDS 模块"""
    global CUFILE_AVAILABLE, CUFILE_BACKEND, _cufile, _CUDA_BINDINGS_DRIVER_READY

    capabilities = _detect_gds_storage_capabilities()
    if not capabilities["storage_path_eligible"]:
        print(
            "[GDS] ⚠️ 未检测到本地 NVMe 或 NFSoRDMA 挂载，"
            "当前不具备直写显存所需的底层存储路径"
        )
        if capabilities["nfs_mounts"] and not capabilities["nfs_rdma_mounts"]:
            print("[GDS] ⚠️ 已检测到 NFS 挂载，但不是 RDMA 传输；这仍会退回 CPU 中转路径")
        CUFILE_AVAILABLE = False
        CUFILE_BACKEND = "none"
        return
    if capabilities["nfs_rdma_mounts"] and not capabilities["rdma_devices"]:
        print("[GDS] ⚠️ 检测到 NFSoRDMA 挂载声明，但当前主机没有 RDMA 设备")
        CUFILE_AVAILABLE = False
        CUFILE_BACKEND = "none"
        return
    if capabilities["nfs_rdma_mounts"]:
        joined_targets = ", ".join(
            str(entry["target"]) for entry in capabilities["nfs_rdma_mounts"]
        )
        print(f"[GDS] ✅ 检测到 NFSoRDMA 挂载: {joined_targets}")
    elif capabilities["has_nvme"]:
        print("[GDS] ✅ 检测到本地 NVMe 存储路径，可尝试本地 GDS")
    
    # 首先尝试使用 cuda.bindings.cufile（推荐方式）
    try:
        import cuda.bindings.cufile as cufile
        # 优先要求同步 read/write + 描述符注册接口完整可用。
        if all(
            hasattr(cufile, attr)
            for attr in ("Descr", "FileHandleType", "handle_register", "handle_deregister", "read", "write")
        ):
            _cufile = cufile
            _CUDA_BINDINGS_DRIVER_READY = False
            CUFILE_AVAILABLE = True
            CUFILE_BACKEND = "cuda_bindings"
            print(f"[GDS] ✅ 成功加载 cuda.bindings.cufile (后端: {CUFILE_BACKEND})")
            return
    except ImportError as e:
        print(f"[GDS] ⚠️ cuda.bindings.cufile 不可用: {e}")
    
    # 备用方案：使用 ctypes 直接调用 libcufile.so
    try:
        import ctypes
        from ctypes import c_int, c_char_p, c_size_t, c_void_p, POINTER
        
        try:
            from ctypes import c_off_t
        except ImportError:
            c_off_t = ctypes.c_longlong
        
        # 加载 libcufile.so
        cufile_paths = [
            "/usr/local/cuda/lib64/libcufile.so",
            "/usr/local/cuda-13.0/targets/x86_64-linux/lib/libcufile.so",
            "/home/gs01/.local/lib/python3.10/site-packages/nvidia/cu13/lib/libcufile.so.0"
        ]
        
        libcufile = None
        for path in cufile_paths:
            try:
                libcufile = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                print(f"[GDS] 找到 libcufile.so: {path}")
                break
            except Exception as e:
                continue
        
        if libcufile:
            _cufile = libcufile
            CUFILE_AVAILABLE = True
            CUFILE_BACKEND = "ctypes"
            print(f"[GDS] ✅ 成功加载 libcufile.so (后端: {CUFILE_BACKEND})")
            return
            
    except Exception as e:
        print(f"[GDS] ⚠️ libcufile.so 不可用: {e}")
    
    # 都不可用
    CUFILE_AVAILABLE = False
    CUFILE_BACKEND = "none"
    print("[GDS] ⚠️ GDS 不可用，将使用标准文件 IO")


def _ensure_cuda_bindings_driver_open() -> None:
    global _CUDA_BINDINGS_DRIVER_READY
    if _CUDA_BINDINGS_DRIVER_READY:
        return
    if _cufile is None or not hasattr(_cufile, "driver_open"):
        _CUDA_BINDINGS_DRIVER_READY = True
        return
    _cufile.driver_open()
    _CUDA_BINDINGS_DRIVER_READY = True


# 初始化
_init_cufile()


class cuFileWrapper:
    """GDS 文件操作包装器"""
    
    def __init__(self, path: str, mode: str = "r"):
        self.path = path
        self.mode = mode
        self.fd = -1
        self.fh = 0
        self._backend = CUFILE_BACKEND
        self._registered_buffers = set()
        self._descr = None
        
        if CUFILE_AVAILABLE:
            if self._backend == "cuda_bindings":
                _ensure_cuda_bindings_driver_open()
                flags = os.O_RDONLY
                if "w" in mode:
                    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
                elif "+" in mode:
                    flags = os.O_RDWR | os.O_CREAT
                self.fd = os.open(path, flags)
                self._descr = _cufile.Descr()
                self._descr.type = int(_cufile.FileHandleType.OPAQUE_FD)
                self._descr.handle.fd = self.fd
                try:
                    # 保留原始 handle 对象，避免在 deregister 时因强制 int 转换破坏 ABI。
                    self.fh = _cufile.handle_register(self._descr.ptr)
                except Exception:
                    os.close(self.fd)
                    self.fd = -1
                    raise
            elif self._backend == "ctypes":
                # ctypes 方式
                import ctypes
                from ctypes import c_int, c_char_p
                
                O_RDONLY = 0o00000000
                O_WRONLY = 0o00000001
                O_RDWR = 0o00000002
                O_CREAT = 0o00000100
                
                if mode == "r":
                    flags = O_RDONLY
                elif mode == "w":
                    flags = O_WRONLY | O_CREAT
                elif mode == "rw":
                    flags = O_RDWR | O_CREAT
                else:
                    flags = O_RDONLY
                
                c_path = c_char_p(path.encode('utf-8'))
                try:
                    if hasattr(_cufile, 'cuFileDriverOpen'):
                        self.fd = _cufile.cuFileDriverOpen(c_path, c_int(flags))
                except:
                    self.fd = -1
    
    def _register_buffer(self, dev_ptr, size):
        """注册 GPU 缓冲区到 GDS"""
        # 显式 buf_register 在当前驱动/compat 组合上会触发 libcufile 崩溃；
        # 先退回 cuFile 内部注册路径，优先保证应用层读写稳定。
        return

    @staticmethod
    def _tensor_byte_view(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.detach().contiguous().view(torch.uint8)

    @staticmethod
    def _chunked_cufile_io(op, fh, base_ptr: int, size: int, offset: int) -> int:
        transferred = 0
        while transferred < size:
            chunk = min(_CUFILE_IO_CHUNK_BYTES, size - transferred)
            done = int(op(fh, base_ptr, chunk, offset + transferred, transferred))
            if done <= 0:
                return transferred if transferred > 0 else done
            transferred += done
            if done < chunk:
                break
        return transferred
    
    def read(self, dev_ptr: torch.Tensor, size: int, offset: int = 0):
        """从文件读取到 GPU 内存"""
        if CUFILE_AVAILABLE and self.fd >= 0:
            if self._backend == "cuda_bindings":
                try:
                    # 注册缓冲区
                    self._register_buffer(dev_ptr.data_ptr(), size)
                    return self._chunked_cufile_io(
                        _cufile.read,
                        self.fh,
                        dev_ptr.data_ptr(),
                        size,
                        offset,
                    )
                except Exception as e:
                    print(f"[GDS] cuda_bindings read 失败: {e}")
            elif self._backend == "ctypes":
                try:
                    import ctypes
                    from ctypes import c_size_t, c_void_p
                    dev_ptr_ptr = c_void_p(dev_ptr.data_ptr())
                    result = _cufile.cuFileRead(
                        c_int(self.fd),
                        dev_ptr_ptr,
                        c_size_t(size),
                        ctypes.c_longlong(offset)
                    )
                    return result
                except Exception as e:
                    print(f"[GDS] ctypes cuFileRead 失败: {e}")
        
        # Fallback: 标准文件 IO
        with open(self.path, "rb") as f:
            f.seek(offset)
            data = f.read(size)
        if isinstance(dev_ptr, torch.Tensor):
            byte_view = self._tensor_byte_view(dev_ptr)
            byte_src = torch.frombuffer(data, dtype=torch.uint8)
            byte_view[: len(data)].copy_(byte_src.to(byte_view.device))
        return len(data)

    def write(self, dev_ptr: torch.Tensor, size: int, offset: int = 0):
        """从 GPU 内存写入文件"""
        if CUFILE_AVAILABLE and self.fd >= 0:
            if self._backend == "cuda_bindings":
                try:
                    # 注册缓冲区
                    self._register_buffer(dev_ptr.data_ptr(), size)
                    return self._chunked_cufile_io(
                        _cufile.write,
                        self.fh,
                        dev_ptr.data_ptr(),
                        size,
                        offset,
                    )
                except Exception as e:
                    print(f"[GDS] cuda_bindings write 失败: {e}")
            elif self._backend == "ctypes":
                try:
                    import ctypes
                    from ctypes import c_size_t, c_void_p
                    dev_ptr_ptr = c_void_p(dev_ptr.data_ptr())
                    result = _cufile.cuFileWrite(
                        c_int(self.fd),
                        dev_ptr_ptr,
                        c_size_t(size),
                        ctypes.c_longlong(offset)
                    )
                    return result
                except Exception as e:
                    print(f"[GDS] ctypes cuFileWrite 失败: {e}")
        
        # Fallback: 标准文件 IO
        if isinstance(dev_ptr, torch.Tensor):
            data = self._tensor_byte_view(dev_ptr).cpu().numpy().tobytes()
        else:
            data = bytes(dev_ptr)
        with open(self.path, "r+b" if os.path.exists(self.path) else "wb") as f:
            f.seek(offset)
            f.write(data)
        return size

    def close(self):
        """关闭文件"""
        if self.fd >= 0:
            if self._backend == "cuda_bindings":
                if self.fh:
                    try:
                        _cufile.handle_deregister(self.fh)
                    except Exception:
                        pass
                os.close(self.fd)
            elif self._backend == "ctypes":
                if hasattr(_cufile, 'cuFileDriverClose'):
                    try:
                        import ctypes
                        _cufile.cuFileDriverClose(ctypes.c_int(self.fd))
                    except:
                        pass
            self.fd = -1
            self.fh = 0


def cuFileRead(
    path: str,
    tensor: torch.Tensor,
    offset: int = 0
) -> int:
    """GDS 读取函数"""
    wrapper = cuFileWrapper(path, "r")
    try:
        size = tensor.numel() * tensor.element_size()
        return wrapper.read(tensor, size, offset)
    finally:
        wrapper.close()


def cuFileWrite(
    path: str,
    tensor: torch.Tensor,
    offset: int = 0
) -> int:
    """GDS 写入函数"""
    wrapper = cuFileWrapper(path, "w")
    try:
        size = tensor.numel() * tensor.element_size()
        return wrapper.write(tensor, size, offset)
    finally:
        wrapper.close()


def is_gds_available() -> bool:
    """检查 GDS 是否可用"""
    return CUFILE_AVAILABLE


def get_gds_backend() -> str:
    """获取当前 GDS 后端"""
    return CUFILE_BACKEND


def get_gds_capabilities() -> Dict[str, object]:
    """返回当前主机的 GDS/NFSoRDMA 能力探测结果"""
    return _detect_gds_storage_capabilities()
