"""test_rdma_cm_exchange.py — host1↔host2 RDMA CM out-of-band 交换协议测试

对应能力：nfsordma_rdma_cm_oob_exchange

无 pyverbs / 无 RDMA 设备时，仍可测试 TCP 交换协议层。
"""

import os
import sys
import threading
import time
import unittest


def _cgc_backend_path():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, "..", "Backend", "CGC"))


def _ensure_paths():
    cgc = _cgc_backend_path()
    if os.path.isdir(cgc) and cgc not in sys.path:
        sys.path.insert(0, cgc)


class RDMAEndpointDataclassTest(unittest.TestCase):
    """RDMAEndpoint 数据类基础测试"""

    @classmethod
    def setUpClass(cls):
        _ensure_paths()
        from edge_moe_transport.rdma_cm_exchange import (
            RDMAEndpoint, EndpointPair, generate_psn,
        )
        cls.RDMAEndpoint = RDMAEndpoint
        cls.EndpointPair = EndpointPair
        cls.generate_psn = staticmethod(generate_psn)

    def test_endpoint_to_json_roundtrip(self):
        """JSON 序列化 / 反序列化"""
        ep = self.RDMAEndpoint(
            qpn=12345,
            gid="fe:80:00:00:00:00:00:00:00:00:00:00:00:00:00:01",
            gid_index=3,
            port_num=1,
            psn=6789,
            lid=0,
            mtu=4096,
            device="rocep0s2",
            cloud_node="cloud_host1",
        )
        s = ep.to_json()
        ep2 = self.RDMAEndpoint.from_json(s)
        self.assertEqual(ep2.qpn, 12345)
        self.assertEqual(ep2.gid, ep.gid)
        self.assertEqual(ep2.gid_index, 3)
        self.assertEqual(ep2.cloud_node, "cloud_host1")

    def test_generate_psn_in_range(self):
        """PSN 在 24-bit 范围内"""
        for _ in range(100):
            psn = self.generate_psn()
            self.assertGreaterEqual(psn, 0)
            self.assertLess(psn, 1 << 24)

    def test_endpoint_pair_construction(self):
        """EndpointPair 包含 local + remote"""
        local = self.RDMAEndpoint(qpn=1, gid="00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:01",
                                  gid_index=0, port_num=1, psn=100)
        remote = self.RDMAEndpoint(qpn=2, gid="00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:02",
                                   gid_index=0, port_num=1, psn=200)
        pair = self.EndpointPair(local=local, remote=remote)
        self.assertEqual(pair.local.qpn, 1)
        self.assertEqual(pair.remote.qpn, 2)


class RDMAEndpointExchangeTCPTest(unittest.TestCase):
    """TCP out-of-band 交换协议测试（不需 RDMA 设备）"""

    @classmethod
    def setUpClass(cls):
        _ensure_paths()
        from edge_moe_transport.rdma_cm_exchange import (
            RDMAEndpoint, RDMAEndpointExchangeServer, RDMAEndpointExchangeClient,
            exchange_endpoint_pair,
        )
        cls.RDMAEndpoint = RDMAEndpoint
        cls.RDMAEndpointExchangeServer = RDMAEndpointExchangeServer
        cls.RDMAEndpointExchangeClient = RDMAEndpointExchangeClient
        cls.exchange_endpoint_pair = staticmethod(exchange_endpoint_pair)

    def _free_port(self):
        import socket as s
        sock = s.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()
        return port

    def test_client_server_exchange(self):
        """client → server 单次交换：双方互收对端 endpoint"""
        port = self._free_port()

        server_local = self.RDMAEndpoint(
            qpn=100, gid="fe:80::1", gid_index=0, port_num=1, psn=1000,
            cloud_node="cloud_host1")
        client_local = self.RDMAEndpoint(
            qpn=200, gid="fe:80::2", gid_index=0, port_num=1, psn=2000,
            cloud_node="edge_node1")

        # 启动 server 线程
        server_result = {}
        def server_thread():
            try:
                server = self.RDMAEndpointExchangeServer(
                    host="127.0.0.1", port=port)
                remote = server.wait_for_endpoint(
                    local_endpoint=server_local, timeout_s=5.0)
                server_result["remote"] = remote
            except Exception as e:
                server_result["error"] = str(e)

        t = threading.Thread(target=server_thread)
        t.start()
        time.sleep(0.1)  # 等 server bind

        client = self.RDMAEndpointExchangeClient(timeout_s=5.0)
        remote = client.exchange_with_peer(
            local_endpoint=client_local,
            peer_addr=("127.0.0.1", port),
        )

        t.join(timeout=5.0)

        # client 收到 server 的 endpoint
        self.assertEqual(remote.qpn, 100)
        self.assertEqual(remote.gid, "fe:80::1")
        self.assertEqual(remote.psn, 1000)

        # server 收到 client 的 endpoint
        self.assertNotIn("error", server_result,
                         f"server error: {server_result.get('error')}")
        self.assertEqual(server_result["remote"].qpn, 200)
        self.assertEqual(server_result["remote"].gid, "fe:80::2")
        self.assertEqual(server_result["remote"].psn, 2000)

    def test_exchange_endpoint_pair_helper(self):
        """exchange_endpoint_pair 一站式入口"""
        port = self._free_port()

        server_local = self.RDMAEndpoint(
            qpn=300, gid="fe:80::3", gid_index=1, port_num=1, psn=3000,
            cloud_node="cloud_host2")
        client_local = self.RDMAEndpoint(
            qpn=400, gid="fe:80::4", gid_index=1, port_num=1, psn=4000,
            cloud_node="edge_node2")

        server_result = {}
        def server_thread():
            try:
                pair = self.exchange_endpoint_pair(
                    local_endpoint=server_local,
                    is_server=True,
                    listen_host="127.0.0.1",
                    listen_port=port,
                    timeout_s=5.0,
                )
                server_result["pair"] = pair
            except Exception as e:
                server_result["error"] = str(e)

        t = threading.Thread(target=server_thread)
        t.start()
        time.sleep(0.1)

        client_pair = self.exchange_endpoint_pair(
            local_endpoint=client_local,
            peer_addr=("127.0.0.1", port),
            is_server=False,
            timeout_s=5.0,
        )
        t.join(timeout=5.0)

        # client pair.local == client_local, pair.remote == server_local
        self.assertEqual(client_pair.local.qpn, 400)
        self.assertEqual(client_pair.remote.qpn, 300)

        # server pair.local == server_local, pair.remote == client_local
        self.assertNotIn("error", server_result)
        self.assertEqual(server_result["pair"].local.qpn, 300)
        self.assertEqual(server_result["pair"].remote.qpn, 400)


class EndpointCacheTest(unittest.TestCase):
    """EndpointCache 持久化缓存"""

    @classmethod
    def setUpClass(cls):
        _ensure_paths()
        from edge_moe_transport.rdma_cm_exchange import (
            RDMAEndpoint, EndpointCache,
        )
        cls.RDMAEndpoint = RDMAEndpoint
        cls.EndpointCache = EndpointCache

    def test_cache_put_get_invalidate(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = self.EndpointCache(cache_dir=tmpdir)

            ep = self.RDMAEndpoint(
                qpn=999, gid="fe:80::9", gid_index=2, port_num=1, psn=9999,
                cloud_node="cloud_cache_test")

            # 初始为空
            self.assertIsNone(cache.get("cloud_cache_test"))

            # put 后可读
            cache.put("cloud_cache_test", ep)
            got = cache.get("cloud_cache_test")
            self.assertIsNotNone(got)
            self.assertEqual(got.qpn, 999)
            self.assertEqual(got.gid, "fe:80::9")

            # invalidate 后为空
            cache.invalidate("cloud_cache_test")
            self.assertIsNone(cache.get("cloud_cache_test"))

    def test_cache_corrupt_file_returns_none(self):
        """缓存文件损坏时返回 None（不抛异常）"""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = self.EndpointCache(cache_dir=tmpdir)
            # 写入损坏的 JSON
            with open(os.path.join(tmpdir, "bad_node.json"), "w") as f:
                f.write("{not valid json")
            self.assertIsNone(cache.get("bad_node"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
