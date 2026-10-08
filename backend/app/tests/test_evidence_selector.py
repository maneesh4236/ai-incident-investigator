import random
import re
from pathlib import Path

import pytest

from app.services.ingestion.event_processor import EventProcessor
from app.services.ingestion.log_loader import LogEventParser
from app.services.reasoning.evidence_selector import EvidenceSelector, render_event
from app.services.reasoning.token_budget import TokenBudget, estimate_tokens
from app.tests.fixtures.gen_logs import generate

FIX = Path(__file__).parent / "fixtures"


def load(path):
    events = LogEventParser().parse_file(str(path), "d")
    groups = EventProcessor().process(events)
    return events, groups


@pytest.fixture(scope="module")
def big(tmp_path_factory):
    path = generate(1_000_000, str(tmp_path_factory.mktemp("big") / "big.log"))
    return (path, *load(path))


def test_token_budget_never_overflows():
    budget = TokenBudget(100)
    for text in ["x" * 50, "y" * 200, "z" * 90, "w" * 10]:
        budget.add(text)
        assert budget.used <= budget.limit
    assert estimate_tokens("") == 0 and estimate_tokens("abc") >= 1


@pytest.mark.parametrize("seed", range(8))
def test_budget_never_exceeded_property(big, seed):
    _, events, groups = big
    rng = random.Random(seed)
    budget = rng.choice([150, 400, 900, 2000, 4000, 8000])
    pack = EvidenceSelector().select(events, groups, "why did it fail?", budget_tokens=budget)
    assert pack.tokens_used <= budget
    assert estimate_tokens(pack.rendered) <= budget


def test_critical_evidence_near_end_of_large_file_survives(big):
    _, events, groups = big
    pack = EvidenceSelector().select(events, groups, "What was the root cause?", budget_tokens=4000)
    for key in ["SQLTimeoutException timeout waiting for connection", "Session leak detected", "DbPool exhausted",
                "Session leak identified", "INCIDENT RESOLVED", "query latency increased to 700 ms"]:
        assert key in pack.rendered, key
    # massive INFO noise: each INFO template contributes at most one event
    info_items = [i for i in pack.items if i.event.level == "INFO"]
    assert len({i.group.id for i in info_items}) == len(info_items) or len(info_items) <= len(groups)
    noise_groups = {i.group.id for i in info_items if i.group.count > 1000}
    assert sum(1 for i in info_items if i.group.id in noise_groups) == len(noise_groups)


def test_stack_trace_rendered_coherently(big):
    _, events, groups = big
    pack = EvidenceSelector().select(events, groups, "root cause", budget_tokens=4000)
    lines = pack.rendered.split("\n")
    idx = next(i for i, l in enumerate(lines) if "Transfer persistence failed" in l)
    assert lines[idx + 1].strip().startswith("java.sql.SQLTimeoutException")
    assert any("Caused by: java.net.SocketTimeoutException" in l for l in lines[idx + 1 : idx + 7])


def test_rendered_lines_are_whole_original_lines(big):
    path, events, groups = big
    original = set(Path(path).read_text(encoding="utf-8").splitlines())
    stripped = {l.strip() for l in original}
    pack = EvidenceSelector().select(events, groups, "root cause", budget_tokens=1200)
    for line in pack.rendered.split("\n"):
        m = re.match(r"^\[(E\d+)\] (.*)$", line)
        if m:
            assert m.group(2) in original
        elif line.startswith("    ") and not line.strip().startswith("^") and not line.strip().endswith("omitted]"):
            assert line.strip() in stripped
    assert pack.valid_event_ids <= {e.id for e in events}


def test_tiny_budget_compacts_and_reports_omissions():
    events, groups = load(FIX / "bracketed_multiline.txt")
    full = EvidenceSelector().select(events, groups, "root cause", budget_tokens=4000)
    tiny = EvidenceSelector().select(events, groups, "root cause", budget_tokens=220)
    assert tiny.tokens_used <= 220 < full.tokens_used
    assert tiny.omitted, "omissions must be reported"
    assert any(i.event.level == "ERROR" for i in tiny.items)


def test_distinct_occurrences_survive_dedup():
    events, groups = load(FIX / "pool_leak_cascade.log")
    pack = EvidenceSelector().select(events, groups, "root cause", budget_tokens=4000)
    sql = [i for i in pack.items if "SQLTimeoutException" in i.event.message]
    assert len(sql) >= 2  # first occurrence and the later burst are both visible
    assert any("same template x3" in i.annotation for i in sql)


def test_deterministic_selection():
    events, groups = load(FIX / "pool_leak_cascade.log")
    a = EvidenceSelector().select(events, groups, "root cause", budget_tokens=600)
    b = EvidenceSelector().select(events, groups, "root cause", budget_tokens=600)
    assert a.rendered == b.rendered


def test_chat_mode_protects_literal_id_matches():
    events, groups = load(FIX / "bracketed_multiline.txt")
    target = next(e for e in events if "req-5004" in e.raw)
    pack = EvidenceSelector().select(events, groups, "what happened to req-5004?", budget_tokens=150,
                                     mode="chat", question_matches=[target])
    assert target.id in pack.valid_event_ids


def test_render_modes_keep_whole_lines():
    events, _ = load(FIX / "bracketed_multiline.txt")
    trace = next(e for e in events if e.has_stack_trace)
    full = render_event(trace, "full")
    header = render_event(trace, "header")
    assert full.count("\n") == trace.continuation_lines
    assert header.split("\n")[0] == f"[{trace.id}] " + trace.raw.split("\n")[0]
    assert "continuation lines omitted" in header
