"""Standard-library-only rendezvous configuration; never imports torch."""

import errno
import math
import os
import socket
from datetime import timedelta


def configured_port(env=None):
    env = os.environ if env is None else env
    port = int(env.get("LOCKSTEP_NCCL_PORT", "2333"))
    if not 1 <= port <= 65535:
        raise ValueError("LOCKSTEP_NCCL_PORT must be between 1 and 65535")
    return port


def check_port(port, *, dual_stack=False):
    """Check, then release: init_process_group still owns the actual bind."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        # A departed TCPStore can leave TIME_WAIT connections. Reuse allows
        # those, while listen() still rejects a genuinely occupied listener.
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", port))
        listener.listen(1)
        selected = listener.getsockname()[1]
        if dual_stack and socket.has_ipv6:
            # Original upstream uses localhost, which can resolve to ::1.
            with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as ipv6:
                ipv6.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
                ipv6.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    ipv6.bind(("::1", selected))
                except OSError as exc:
                    if exc.errno not in (errno.EAFNOSUPPORT, errno.EADDRNOTAVAIL):
                        raise
                else:
                    ipv6.listen(1)
        return selected


def rendezvous_settings(env=None):
    env = os.environ if env is None else env
    seconds = float(env.get("LOCKSTEP_NCCL_INIT_TIMEOUT_SECONDS", "120"))
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("LOCKSTEP_NCCL_INIT_TIMEOUT_SECONDS must be finite and positive")
    return f"tcp://127.0.0.1:{configured_port(env)}", timedelta(seconds=seconds)
