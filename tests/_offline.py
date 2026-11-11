"""테스트를 항상 오프라인으로 실행한다.

PC에 인터넷과 .env의 API 키가 있으면 엔진이 실제 야후·FRED·DART 데이터를 받아 와서
가짜 데이터로 정한 기대값과 결과가 달라질 수 있다. 외부 연결을 막아
어느 컴퓨터에서 돌려도 같은 결과가 나오게 한다.
(Windows의 asyncio는 내부적으로 localhost 소켓을 쓰므로 localhost 연결은 허용한다.)
"""
import socket

_LOCAL = {"localhost", "127.0.0.1", "::1", "", None}
_orig_connect = socket.socket.connect
_orig_connect_ex = socket.socket.connect_ex
_orig_create_connection = socket.create_connection
_orig_getaddrinfo = socket.getaddrinfo


def _is_local(address):
    host = address[0] if isinstance(address, tuple) and address else address
    return host in _LOCAL or str(host).startswith("127.")


def _err(target):
    return OSError(f"테스트는 오프라인으로 실행합니다 (tests/_offline.py): {target}")


def _connect(self, address):
    if not _is_local(address):
        raise _err(address)
    return _orig_connect(self, address)


def _connect_ex(self, address):
    if not _is_local(address):
        raise _err(address)
    return _orig_connect_ex(self, address)


def _create_connection(address, *args, **kwargs):
    if not _is_local(address):
        raise _err(address)
    return _orig_create_connection(address, *args, **kwargs)


def _getaddrinfo(host, *args, **kwargs):
    if not _is_local((host,)):
        raise socket.gaierror(f"테스트는 오프라인으로 실행합니다 (tests/_offline.py): {host}")
    return _orig_getaddrinfo(host, *args, **kwargs)


socket.socket.connect = _connect
socket.socket.connect_ex = _connect_ex
socket.create_connection = _create_connection
socket.getaddrinfo = _getaddrinfo
