"""Test doubles: a sandbox that does not need Docker and a scripted model."""

from contextlib import contextmanager

from refuterl.providers.base import ModelTurn, ToolCall, ToolResult, Usage
from refuterl.sandbox import RunResult


class FakeSandbox:
    def __init__(self, stdout="ok\n", exit_code=0, timed_out=False):
        self.calls = []
        self.stdout = stdout
        self.exit_code = exit_code
        self.timed_out = timed_out

    def run_python(self, code, timeout):
        self.calls.append((code, timeout))
        return RunResult(self.stdout, "", self.exit_code, self.timed_out, 0.1)


def fake_sandbox_factory(box):
    @contextmanager
    def factory(profile):
        yield box

    return factory


def turn(*calls, text=""):
    """Build a model turn. Each call is (name, input)."""
    return ModelTurn(
        text=text,
        tool_calls=[ToolCall(f"call{i}", name, inp) for i, (name, inp) in enumerate(calls)],
        usage=Usage(10, 5),
    )


class FakeConversation:
    def __init__(self, turns):
        self.turns = list(turns)
        self.tool_results: list[list[ToolResult]] = []

    def send(self):
        if not self.turns:
            return turn(text="(script ran out)")
        return self.turns.pop(0)

    def add_tool_results(self, results):
        self.tool_results.append(results)


class FakeProvider:
    name = "fake"
    model = "fake-model"

    def __init__(self, turns):
        self.conversation = FakeConversation(turns)
        self.started_with = None

    def start(self, system, prompt, tools):
        self.started_with = (system, prompt, [t.name for t in tools])
        return self.conversation
