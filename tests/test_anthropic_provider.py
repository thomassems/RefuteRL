"""The Anthropic adapter, tested against a fake client (no network, no API key)."""

from types import SimpleNamespace

import pytest

from refuterl.providers import anthropic as adapter
from refuterl.providers.anthropic import DEEPSEEK_BASE_URL, AnthropicProvider
from refuterl.providers.base import ToolResult
from refuterl.tools import tool_specs


class FakeMessages:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def create(self, **kwargs):
        # Copy the message list, since the adapter keeps appending to the same one.
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        return self.responses.pop(0)


def block(type_, **fields):
    return SimpleNamespace(type=type_, **fields)


def response(*content, stop_reason="tool_use"):
    return SimpleNamespace(
        content=list(content),
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=100, output_tokens=20),
    )


def make_provider(responses, thinking=None):
    messages = FakeMessages(responses)
    client = SimpleNamespace(messages=messages)
    return AnthropicProvider("deepseek-flash", client=client, thinking=thinking), messages


def test_tool_loop_sends_full_assistant_content_back():
    thinking = block("thinking", thinking="try small n", signature="sig")
    call = block("tool_use", id="tu1", name="run_python", input={"code": "print(1)"})
    provider, messages = make_provider(
        [
            response(thinking, block("text", text="Let me search."), call),
            response(block("text", text="Done."), stop_reason="end_turn"),
        ],
        thinking={"type": "enabled"},
    )

    conversation = provider.start("system text", "claim text", tool_specs(True))
    first = conversation.send()
    assert first.text == "Let me search."
    assert [(c.id, c.name, c.input) for c in first.tool_calls] == [
        ("tu1", "run_python", {"code": "print(1)"})
    ]
    assert first.usage.input_tokens == 100

    conversation.add_tool_results([ToolResult("tu1", "1\n")])
    second = conversation.send()
    assert second.tool_calls == []
    assert second.stop_reason == "end_turn"

    request = messages.requests[1]
    assistant, tool_turn = request["messages"][1], request["messages"][2]
    # The thinking block goes back unchanged, as DeepSeek requires.
    assert assistant["content"][0] is thinking
    assert tool_turn["content"][0]["type"] == "tool_result"
    assert tool_turn["content"][0]["tool_use_id"] == "tu1"


def test_request_shape():
    provider, messages = make_provider(
        [response(block("text", text="hi"), stop_reason="end_turn")], thinking={"type": "enabled"}
    )
    provider.start("sys", "claim", tool_specs(False)).send()
    request = messages.requests[0]
    assert request["system"] == "sys"
    assert request["messages"] == [{"role": "user", "content": "claim"}]
    assert [t["name"] for t in request["tools"]] == ["submit"]
    assert request["thinking"] == {"type": "enabled"}
    assert "tool_choice" not in request  # rejected by DeepSeek in thinking mode


def test_cached_tokens_count_as_input():
    cached = SimpleNamespace(
        content=[block("text", text="hi")],
        stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=50, cache_read_input_tokens=400, output_tokens=10),
    )
    provider, _ = make_provider([cached])
    assert provider.start("sys", "claim", tool_specs(False)).send().usage.input_tokens == 450


def test_thinking_omitted_when_not_set():
    provider, messages = make_provider([response(block("text", text="hi"))])
    provider.start("sys", "claim", tool_specs(False)).send()
    assert "thinking" not in messages.requests[0]


def test_deepseek_factory(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    provider = adapter.deepseek()
    assert provider.name == "deepseek"
    assert provider.model == "deepseek-flash"
    assert provider.thinking == {"type": "enabled"}
    assert str(provider.client.base_url).rstrip("/") == DEEPSEEK_BASE_URL
    assert adapter.deepseek(thinking=False).thinking == {"type": "disabled"}


def test_deepseek_factory_needs_key(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
        adapter.deepseek()
