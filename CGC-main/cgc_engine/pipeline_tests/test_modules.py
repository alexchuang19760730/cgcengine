#!/usr/bin/env python3
import sys

print('=== Testing flash_moe module ===')
try:
    from cgc_engine.flash_moe import FlashMoEClient
    print('✅ flash_moe import OK')
    client = FlashMoEClient()
    print(f'   flash_moe client info: {client.info()}')
except Exception as e:
    print(f'⚠️ flash_moe: {e}')
    import traceback
    traceback.print_exc()

print()
print('=== Testing omlx module ===')
try:
    from cgc_engine.omlx import OMLXClient
    print('✅ omlx import OK')
    omlx = OMLXClient()
    print(f'   omlx client info: {omlx.info()}')
except Exception as e:
    print(f'⚠️ omlx: {e}')
    import traceback
    traceback.print_exc()

print()
print('=== Testing gds_service module ===')
try:
    from cgc_engine.gds_service import GDSManager, is_gds_available
    print('✅ gds_service import OK')
    print(f'   GDS available: {is_gds_available()}')
except Exception as e:
    print(f'⚠️ gds_service: {e}')
    import traceback
    traceback.print_exc()

print()
print('=== Testing cgc_commands with new opcodes ===')
try:
    from cgc_engine.cgc.cgc_commands import (
        GDS_LOAD_KV_CMD, GDS_STORE_KV_CMD, GDS_LOAD_WEIGHT_CMD,
        FLASH_MOE_LOAD_EXPERTS_CMD, FLASH_MOE_RUN_MLP_CMD,
        OMLX_FLASH_PREDICT_CMD, OMLX_FLASH_CACHE_CMD,
        JIT_LOAD_COMPILED_CMD, JIT_COMPILE_KERNEL_CMD, JIT_DISPATCH_CMD,
        SPDK_READ_CMD, SPDK_WRITE_CMD, SPDK_ALLOC_BUF_CMD,
    )
    print('✅ All new CGC commands import OK')
    print(f'   GDS opcodes: 0x{GDS_LOAD_KV_CMD.opcode:02X}, 0x{GDS_STORE_KV_CMD.opcode:02X}, 0x{GDS_LOAD_WEIGHT_CMD.opcode:02X}')
    print(f'   FlashMoE opcodes: 0x{FLASH_MOE_LOAD_EXPERTS_CMD.opcode:02X}, 0x{FLASH_MOE_RUN_MLP_CMD.opcode:02X}')
    print(f'   oMLX opcodes: 0x{OMLX_FLASH_PREDICT_CMD.opcode:02X}, 0x{OMLX_FLASH_CACHE_CMD.opcode:02X}')
    print(f'   JIT opcodes: 0x{JIT_LOAD_COMPILED_CMD.opcode:02X}, 0x{JIT_COMPILE_KERNEL_CMD.opcode:02X}, 0x{JIT_DISPATCH_CMD.opcode:02X}')
    print(f'   SPDK opcodes: 0x{SPDK_READ_CMD.opcode:02X}, 0x{SPDK_WRITE_CMD.opcode:02X}, 0x{SPDK_ALLOC_BUF_CMD.opcode:02X}')
except Exception as e:
    print(f'⚠️ CGC commands: {e}')
    import traceback
    traceback.print_exc()

print()
print('=== Testing cgc_simd_executor import ===')
try:
    from cgc_engine.cgc.cgc_simd_executor import CGCExecutor, CGCCommand
    print('✅ cgc_simd_executor import OK')
    executor = CGCExecutor()
    print('   CGCExecutor initialized')
except Exception as e:
    print(f'⚠️ cgc_simd_executor: {e}')
    import traceback
    traceback.print_exc()

print()
print('=== All tests completed ===')
