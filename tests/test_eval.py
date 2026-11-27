import json
from pathlib import Path

from refuterl.claim import load_claim
from refuterl.eval import run_episode, run_eval, summarize, system_prompt, user_prompt
from refuterl.tools import PROFILES
from tests.fakes import FakeProvider, FakeSandbox, fake_sandbox_factory, turn

ROOT = Path(__file__).resolve().parent.parent
EULER = load_claim(ROOT / "claims" / "number_theory" / "euler_prime_polynomial.py")
QUICK = PROFILES["quick"]


def episode(turns, tools_enabled=True, box=None):
    box = box or FakeSandbox()
    provider = FakeProvider(turns)
    result = run_episode(EULER, provider, QUICK, tools_enabled, fake_sandbox_factory(box))
    return result, provider, box


def test_search_then_correct_submit():
    result, provider, box = episode(
        [
            turn(("run_python", {"code": "search()"}), text="Searching."),
            turn(("submit", {"candidate": "40"})),
        ]
    )
    assert result.end_reason == "submitted"
    assert result.verdict == "refuted"
    assert result.reward == 1.0
    assert result.tool_calls == 1
    assert result.turns == 2
    assert result.input_tokens == 20
    assert box.calls == [("search()", 10)]
    # The tool result went back to the model before its second turn.
    assert provider.conversation.tool_results[0][0].call_id == "call0"


def test_wrong_answer():
    result, *_ = episode([turn(("submit", {"candidate": "39"}))])
    assert (result.verdict, result.reward) == ("not_refuted", 0.0)


def test_badly_formatted_answer():
    result, *_ = episode([turn(("submit", {"candidate": "forty"}))])
    assert result.verdict == "invalid_format"


def test_none_found():
    result, *_ = episode([turn(("submit", {"candidate": "none"}))])
    assert (result.end_reason, result.verdict) == ("none_found", "no_answer")


def test_model_stops_without_submitting():
    result, *_ = episode([turn(text="I think it is 40.")])
    assert (result.end_reason, result.verdict) == ("no_submit", "no_answer")
    assert result.final_text == "I think it is 40."


def test_max_turns():
    result, *_ = episode([turn(("run_python", {"code": "1"})) for _ in range(20)])
    assert result.end_reason == "max_turns"
    assert result.turns == QUICK.max_turns
    # Only 3 calls reached the sandbox; the rest were refused by the budget.
    assert result.tool_calls == 3


def test_no_tools_mode_never_starts_a_sandbox():
    def no_sandbox(profile):
        raise AssertionError("sandbox should not start")

    provider = FakeProvider([turn(("submit", {"candidate": "40"}))])
    result = run_episode(EULER, provider, QUICK, tools_enabled=False, sandbox_factory=no_sandbox)
    assert result.verdict == "refuted"
    assert provider.started_with[2] == ["submit"]
    assert "cannot run code" in provider.started_with[0]


def test_provider_error_is_recorded_not_raised():
    class Broken(FakeProvider):
        def start(self, *args):
            raise ConnectionError("network down")

    result = run_episode(EULER, Broken([]), QUICK, True, fake_sandbox_factory(FakeSandbox()))
    assert result.end_reason == "error"
    assert result.verdict is None
    assert "network down" in result.detail


def test_prompts_mention_budget_and_format():
    assert "3 times" in system_prompt(QUICK, True)
    assert "10 seconds" in system_prompt(QUICK, True)
    assert "n^2 + n + 41" in user_prompt(EULER)
    assert "single integer" in user_prompt(EULER)


def test_run_eval_writes_jsonl_and_summary(tmp_path):
    out = tmp_path / "run.jsonl"
    provider = FakeProvider([turn(("submit", {"candidate": "40"}))])
    results = run_eval(
        [EULER], provider, QUICK, out, sandbox_factory=fake_sandbox_factory(FakeSandbox()),
        log=lambda _: None,
    )  # fmt: skip
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["claim_id"] == "euler_prime_polynomial"
    assert rows[0]["verdict"] == "refuted"
    assert summarize(results).startswith("Refuted 1 of 1 claims")
