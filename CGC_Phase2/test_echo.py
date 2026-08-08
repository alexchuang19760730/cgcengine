"""Trivial socket echo to isolate the SSH tunnel from the transport protocol.

server (Host2): accepts, reads 9 bytes, echoes b"ECHO"+those bytes, closes.
client (Host1): connects to 127.0.0.1:31000 (tunnel endpoint), sends 9 bytes,
                prints the 13-byte echo response.
"""
import socket
import sys


def run_server(port=31000):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", port))
    srv.listen(8)
    print(f"[echo] listening 0.0.0.0:{port}", flush=True)
    while True:
        c, _ = srv.accept()
        data = c.recv(9)
        c.sendall(b"ECHO" + data)
        c.close()


def run_client(port=31000):
    c = socket.create_connection(("127.0.0.1", port), timeout=5)
    c.sendall(b"G" + (0).to_bytes(4, "big") + (0).to_bytes(4, "big"))
    resp = c.recv(13)
    print("ECHO resp:", resp, flush=True)
    c.close()


if __name__ == "__main__":
    if sys.argv[1] == "server":
        run_server(int(sys.argv[2]) if len(sys.argv) > 2 else 31000)
    else:
        run_client(int(sys.argv[2]) if len(sys.argv) > 2 else 31000)
