"""Provider interface shared by all adapters.

The eval loop only talks to these types. Each adapter keeps the conversation
history in its own native format, because providers have different rules about
what must be sent back (for example DeepSeek needs thinking blocks returned).

    conversation = provider.start(system, prompt, tools)
    turn = conversation.send()
    while turn.tool_calls:
        conversation.add_tool_results([...])
        turn = conversation.send()
"""

from dataclasses import dataclass, field
from typing import Any, Protocol

from refuterl.tools import ToolSpec


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]


@dataclass(frozen=True)
class ToolResult:
    call_id: str
    content: str
    is_error: bool = False


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True)
class ModelTurn:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str | None = None
    usage: Usage = Usage()


class Conversation(Protocol):
    def send(self) -> ModelTurn:
        """Ask the model for its next turn, given everything so far."""
        ...

    def add_tool_results(self, results: list[ToolResult]) -> None:
        """Answer every tool call from the last turn."""
        ...


class Provider(Protocol):
    name: str
    model: str

    def start(self, system: str, prompt: str, tools: list[ToolSpec]) -> Conversation: ...
