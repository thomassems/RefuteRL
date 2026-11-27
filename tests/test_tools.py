from refuterl.tools import PROFILES, ToolRunner, tool_specs
from tests.fakes import FakeSandbox

QUICK = PROFILES["quick"]


def test_profiles_match_plan():
    standard = PROFILES["standard"]
    assert (standard.max_tool_calls, standard.seconds_per_call, standard.max_turns) == (10, 30, 15)
    assert (QUICK.max_tool_calls, QUICK.seconds_per_call) == (3, 10)
    assert standard.sandbox_limits().memory == "1g"


def test_tool_specs_by_mode():
    assert [t.name for t in tool_specs(True)] == ["run_python", "submit"]
    assert [t.name for t in tool_specs(False)] == ["submit"]


def test_run_python_uses_sandbox_and_budget():
    box = FakeSandbox(stdout="42\n")
    runner = ToolRunner(QUICK, box)
    outcome = runner.handle("run_python", {"code": "print(42)"})
    assert box.calls == [("print(42)", 10)]
    assert "42" in outcome.content
    assert "calls left: 2" in outcome.content
    assert not outcome.is_error


def test_budget_runs_out():
    box = FakeSandbox()
    runner = ToolRunner(QUICK, box)
    for _ in range(3):
        runner.handle("run_python", {"code": "1"})
    outcome = runner.handle("run_python", {"code": "1"})
    assert outcome.is_error
    assert "No run_python calls left" in outcome.content
    assert len(box.calls) == 3


def test_timeout_and_failure_are_reported():
    runner = ToolRunner(QUICK, FakeSandbox(stdout="", exit_code=124, timed_out=True))
    outcome = runner.handle("run_python", {"code": "while True: pass"})
    assert outcome.is_error
    assert "exceeded 10s" in outcome.content


def test_submit_returns_candidate():
    outcome = ToolRunner(QUICK, None).handle("submit", {"candidate": " 40 "})
    assert outcome.submitted == "40"


def test_bad_inputs_are_errors_not_crashes():
    runner = ToolRunner(QUICK, FakeSandbox())
    assert runner.handle("submit", {"candidate": 40}).is_error
    assert runner.handle("run_python", {}).is_error
    assert runner.handle("delete_files", {}).is_error
    assert runner.calls_used == 0


def test_run_python_unavailable_without_sandbox():
    assert ToolRunner(QUICK, None).handle("run_python", {"code": "1"}).is_error
