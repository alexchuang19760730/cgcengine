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
PD Service Module - Prefetch Distribution Service

功能:
- KV Cache 块管理
- Prefix KV 缓存
- CGC 命令绑定
- FlashKDA 融合
- 多机分布式支持

使用:
    # Server
    from cgc_engine.pd import serve
    serve(port=50051)

    # Client
    from cgc_engine.pd import PDClient
    client = PDClient("localhost:50051")
"""

try:
    from .pd_server import (
        PDServerServicer,
        DistributedKVCache,
        CGCCommandExecutor,
        serve,
        serve_sync,
    )
    PD_SERVER_AVAILABLE = True
except ImportError as e:
    PD_SERVER_AVAILABLE = False
    print(f"[PD] Server not available: {e}")

try:
    from .pd_client import (
        PDClient,
        PDClientConfig,
        decode_pd_resume_payload,
        get_pd_client,
        create_pd_client,
    )
    PD_CLIENT_AVAILABLE = True
except ImportError as e:
    PD_CLIENT_AVAILABLE = False
    print(f"[PD] Client not available: {e}")

try:
    from .dopd_schema import (
        DOPDResumePayloadV2,
        DOPD_RESUME_PAYLOAD_MAGIC_V2,
        compute_dopd_resume_checksum,
        decode_dopd_resume_payload_v2,
        encode_dopd_resume_payload_v2,
        normalize_dopd_resume_payload_v2,
    )
except ImportError:
    pass

try:
    from .dopd_runtime import (
        DOPDHandoffRecord,
        DOPDSessionRuntime,
        DOPDWorkerAdapter,
    )
except ImportError:
    pass

__all__ = [
    "PDServerServicer",
    "DistributedKVCache",
    "CGCCommandExecutor",
    "serve",
    "serve_sync",
    "PDClient",
    "PDClientConfig",
    "DOPDResumePayloadV2",
    "DOPDHandoffRecord",
    "DOPD_RESUME_PAYLOAD_MAGIC_V2",
    "DOPDSessionRuntime",
    "DOPDWorkerAdapter",
    "compute_dopd_resume_checksum",
    "decode_dopd_resume_payload_v2",
    "encode_dopd_resume_payload_v2",
    "normalize_dopd_resume_payload_v2",
    "decode_pd_resume_payload",
    "get_pd_client",
    "create_pd_client",
    "PD_SERVER_AVAILABLE",
    "PD_CLIENT_AVAILABLE",
]
