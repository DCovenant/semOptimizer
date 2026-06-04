"""Wire protocol for the out-of-process YOLO inference service.

Shared verbatim by both sides of the Python gap:
  * the service  — Python 3.12 + ROCm torch  (inference/service.py)
  * the client   — Python 3.7  + the CARLA egg (app/core/analysis_worker.py)

So this module must stay pure-Python and 3.7-compatible: no Qt, no torch, no
numpy-only-3.8 features, no f-string-debug or walrus in hot paths that 3.7 can't
parse. It only knows how to frame messages on a stream socket.

Framing (one message = one header + optional raw payload):

    [4 bytes big-endian]  header length H
    [H bytes]             UTF-8 JSON header
    [N bytes]             raw payload (N = header["nbytes"], 0 if absent)

The header always carries a "type". For a "track" request the payload is the
raw RGB frame (height*width*3 uint8, C-contiguous); the header carries its
shape so the service can rebuild the array without guessing.
"""
import json
import socket
import struct

DEFAULT_SOCKET_PATH = "/tmp/sem_inference.sock"

_LEN = struct.Struct(">I")          # 4-byte unsigned big-endian length prefix


def _recv_exactly(sock, n):
    """Read exactly n bytes from sock or raise ConnectionError on early EOF."""
    chunks = []
    remaining = n
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("socket closed mid-message")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_message(sock, header, payload=b""):
    """Send one framed message: JSON header dict + optional raw bytes payload."""
    if payload:
        header = dict(header, nbytes=len(payload))
    raw_header = json.dumps(header).encode("utf-8")
    sock.sendall(_LEN.pack(len(raw_header)))
    sock.sendall(raw_header)
    if payload:
        sock.sendall(payload)


def recv_message(sock):
    """Receive one framed message → (header_dict, payload_bytes)."""
    (hlen,) = _LEN.unpack(_recv_exactly(sock, _LEN.size))
    header = json.loads(_recv_exactly(sock, hlen).decode("utf-8"))
    nbytes = header.get("nbytes", 0)
    payload = _recv_exactly(sock, nbytes) if nbytes else b""
    return header, payload
