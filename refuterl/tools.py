"""Tools shown to the model, budget profiles, and the per-episode tool runner.

The model gets two tools:

    run_python(code)      runs code in the episode's Docker sandbox, uses up budget
    submit(candidate)     final answer: JSON text of a counterexample, or "none"

In no-tools mode only submit is offered.
"""

from dataclasses import dataclass
from typing import Any, Protocol

from refuterl.sandbox import RunResult, SandboxLimits

NONE_FOUND = "none"


@dataclass(frozen=True)
class BudgetProfile:
    name: str
    max_tool_calls: int
    seconds_per_call: float
    memory: str
    cpus: float
    max_turns: int

    def sandbox_limits(self) -> SandboxLimits:
        return SandboxLimits(memory=self.memory, cpus=self.cpus)


PROFILES = {
    "standard": BudgetProfile("standard", 10, 30, "1g", 1.0, 15),
    "quick": BudgetProfile("quick", 3, 10, "1g", 1.0, 15),
}


@dataclass(frozen=True)
class ToolSpec:
    """Provider-neutral tool definition. Adapters convert it to their own format."""

    name: str
    description: str
    input_schema: dict[str, Any]


RUN_PYTHON = ToolSpec(
    name="run_python",
    description=(
        "Run Python 3.12 code in an isolated sandbox and get back stdout, stderr and the exit "
        "code. Each call is a fresh Python process: variables do not carry over, but files you "
        "write to /tmp do, and /tmp is on the import path. sympy, numpy and gmpy2 are installed. "
        "There is no network. Print what you want to see. Calls are limited, so plan searches "
        "before running them."
    ),
    input_schema={
        "type": "object",
        "properties": {"code": {"type": "string", "description": "Python source to run."}},
        "required": ["code"],
    },
)

SUBMIT = ToolSpec(
    name="submit",
    description=(
        "Submit your final answer. This ends the attempt. Pass the counterexample as JSON text "
        'in the required format, for example "40" or "[3, 4]". If you did not find a '
        f'counterexample, pass "{NONE_FOUND}".'
    ),
    input_schema={
        "type": "object",
        "properties": {
            "candidate": {
                "type": "string",
                "description": f'JSON text of the counterexample, or "{NONE_FOUND}".',
            }
        },
        "required": ["candidate"],
    },
)


def tool_specs(tools_enabled: bool) -> list[ToolSpec]:
    return [RUN_PYTHON, SUBMIT] if tools_enabled else [SUBMIT]


class CodeRunner(Protocol):
    """What the tool runner needs from a sandbox. Sandbox satisfies it; tests use fakes."""

    def run_python(self, code: str, timeout: float) -> RunResult: ...


@dataclass(frozen=True)
class ToolOutcome:
    content: str  # text sent back to the model as the tool result
    is_error: bool = False
    submitted: str | None = None  # set when the model called submit


class ToolRunner:
    """Executes tool calls for one episode and tracks the run_python budget."""

    def __init__(self, profile: BudgetProfile, sandbox: CodeRunner | None):
        self.profile = profile
        self.sandbox = sandbox
        self.calls_used = 0
        self.seconds_used = 0.0

    @property
    def calls_left(self) -> int:
        return self.profile.max_tool_calls - self.calls_used

    def handle(self, name: str, tool_input: dict[str, Any]) -> ToolOutcome:
        if name == SUBMIT.name:
            candidate = tool_input.get("candidate")
            if not isinstance(candidate, str):
                return ToolOutcome("submit needs a string field named candidate.", is_error=True)
            return ToolOutcome("Answer received.", submitted=candidate.strip())

        if name == RUN_PYTHON.name:
            return self._run_python(tool_input)

        return ToolOutcome(f"Unknown tool: {name}", is_error=True)

    def _run_python(self, tool_input: dict[str, Any]) -> ToolOutcome:
        if self.sandbox is None:
            return ToolOutcome("run_python is not available in this mode.", is_error=True)
        code = tool_input.get("code")
        if not isinstance(code, str):
            return ToolOutcome("run_python needs a string field named code.", is_error=True)
        if self.calls_left <= 0:
            return ToolOutcome(
                "No run_python calls left. Call submit with your best answer now.", is_error=True
            )

        self.calls_used += 1
        result = self.sandbox.run_python(code, timeout=self.profile.seconds_per_call)
        self.seconds_used += result.duration
        return ToolOutcome(self._format(result), is_error=result.exit_code != 0)

    def _format(self, result: RunResult) -> str:
        parts = []
        if result.timed_out:
            parts.append(f"[stopped: exceeded {self.profile.seconds_per_call:g}s time limit]")
        parts.append(f"exit code: {result.exit_code}")
        if result.stdout:
            parts.append(f"stdout:\n{result.stdout}")
        if result.stderr:
            parts.append(f"stderr:\n{result.stderr}")
        if not result.stdout and not result.stderr:
            parts.append("(no output)")
        parts.append(f"run_python calls left: {self.calls_left}")
        return "\n".join(parts)
