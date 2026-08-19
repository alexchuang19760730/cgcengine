#!/usr/bin/env python3
import sys

sys.path.insert(0, '/home/gs01')

print("Testing cgc_engine.pd imports...")
from cgc_engine.pd import pd_service_pb2, pd_service_pb2_grpc, PD_PROTO_AVAILABLE
print(f"PD_PROTO_AVAILABLE: {PD_PROTO_AVAILABLE}")

if PD_PROTO_AVAILABLE:
    print("\nTesting PDClient connection...")
    from cgc_engine.pd import PDClient
    client = PDClient("localhost:50051")
    healthy, stats = client.health_check()
    print(f"Health: {healthy}")
    print(f"Stats: {stats}")
    client.close()
    print("\n✅ PD Client integration successful!")
else:
    print("\n❌ PD Proto not available!")
