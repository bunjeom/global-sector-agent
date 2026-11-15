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


# yfinance는 curl_cffi(C 라이브러리 libcurl)로 접속해서 위의 파이썬 소켓 차단을 거치지 않는다.
# 그래서 인터넷이 되는 PC에서는 테스트 중에도 실제 야후에 접속해 느려지거나(재시도·속도 제한) 결과가 달라질 수 있었다.
# curl_cffi 요청도 실제 접속 실패와 같은 종류의 오류(ConnectionError)로 바로 막는다.
try:
    import curl_cffi.requests as _cr
    from curl_cffi.requests.exceptions import ConnectionError as _CurlConnErr

    def _curl_blocked(self, method, url, *args, **kwargs):
        raise _CurlConnErr(f"Failed to perform, curl: (7) 테스트는 오프라인으로 실행합니다 (tests/_offline.py): {url}")

    async def _curl_blocked_async(self, method, url, *args, **kwargs):
        raise _CurlConnErr(f"Failed to perform, curl: (7) 테스트는 오프라인으로 실행합니다 (tests/_offline.py): {url}")

    _cr.Session.request = _curl_blocked
    if hasattr(_cr, "AsyncSession"):
        _cr.AsyncSession.request = _curl_blocked_async
except ImportError:   # curl_cffi가 없는 환경(구버전 yfinance)은 소켓 차단만으로 충분
    pass
