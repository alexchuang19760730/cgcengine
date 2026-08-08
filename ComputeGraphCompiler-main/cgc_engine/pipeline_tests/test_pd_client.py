#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/gs01')

print("Testing PD Client import and connection...")

from cgc_engine.pd import PDClient, PD_PROTO_AVAILABLE

print(f"PD_PROTO_AVAILABLE: {PD_PROTO_AVAILABLE}")

if PD_PROTO_AVAILABLE:
    print("Creating PDClient...")
    client = PDClient("localhost:50051")
    print("PDClient created successfully!")

    print("Testing health check...")
    healthy, stats = client.health_check()
    print(f"Health Check: {healthy}")
    print(f"Stats: {stats}")

    print("Testing block allocation...")
    block_ids, success = client.allocate_blocks(sequence_ids=[1], num_blocks=4)
    print(f"Allocated blocks: {block_ids}, Success: {success}")

    client.close()
    print("\n✅ PD Client integration successful!")
else:
    print("\n❌ PD Proto not available!")
