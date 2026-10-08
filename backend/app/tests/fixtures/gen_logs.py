"""
Seeded synthetic log generator for scale tests (no real data).

`generate(target_bytes, path)` writes repetitive INFO/DEBUG noise for ~90% of
the size, then the `pool_leak_cascade.log` incident (plus one multiline Java
stack trace), then a short healthy tail - i.e. the critical evidence sits
near the END of a large file.
"""
from __future__ import annotations

import os
import random
from datetime import datetime, timedelta

FIXTURES = os.path.dirname(os.path.abspath(__file__))
_SERVICES = ["web-gateway", "ledger-service", "account-service", "notify-service", "ledger-db"]
_NOISE = [
    "INFO Request received GET /health",
    "INFO Transfer completed txnId=TX{n}",
    "INFO Account ACC{n} updated",
    "INFO SQL execution completed in {m} ms",
    "INFO DbPool active={a} idle={i} total=100",
    "DEBUG cache hit key=acct:{n}",
    "INFO HTTP 200 GET /accounts/{n} took {m}ms",
]
_STACK = (
    "2030-03-20 09:06:01 ledger-service ERROR Transfer persistence failed\n"
    "java.sql.SQLTimeoutException: timeout waiting for connection\n"
    "\tat com.example.ledger.TransferDao.save(TransferDao.java:88)\n"
    "\tat com.example.ledger.TransferService.commit(TransferService.java:51)\n"
    "Caused by: java.net.SocketTimeoutException: Read timed out\n"
    "\t... 9 more\n"
)


def generate(target_bytes: int, path: str, seed: int = 7) -> str:
    rng = random.Random(seed)
    incident = open(os.path.join(FIXTURES, "pool_leak_cascade.log"), encoding="utf-8").read().splitlines()
    incident = [l for l in incident if l.strip()]
    # insert the multiline stack trace right after the first SQLTimeoutException line
    idx = next(i for i, l in enumerate(incident) if "SQLTimeoutException" in l)
    noise_budget = int(target_bytes * 0.9)
    start = datetime(2030, 3, 20, 9, 0, 0) - timedelta(seconds=noise_budget // 60)
    size = 0
    t = start
    with open(path, "w", encoding="utf-8", newline="\n") as out:
        while size < noise_budget:
            a = rng.randint(5, 20)
            line = f"{t:%Y-%m-%d %H:%M:%S} {rng.choice(_SERVICES)} " + rng.choice(_NOISE).format(
                n=rng.randint(1000, 99999), m=rng.randint(5, 60), a=a, i=100 - a
            )
            out.write(line + "\n")
            size += len(line) + 1
            if rng.random() < 0.02:
                t += timedelta(seconds=1)
        for i, line in enumerate(incident):
            out.write(line + "\n")
            size += len(line) + 1
            if i == idx:
                out.write(_STACK)
                size += len(_STACK)
        while size < target_bytes:
            line = "2030-03-20 09:11:00 web-gateway INFO HTTP 200 GET /health"
            out.write(line + "\n")
            size += len(line) + 1
    return path
