"""Anthropic SDK adapter. Used for DeepSeek (Anthropic-compatible endpoint) and Claude.

DeepSeek notes, from its Anthropic API guide:
- Base URL is https://api.deepseek.com/anthropic. Models: deepseek-flash, deepseek-v4-pro.
- With thinking on, the full assistant content (thinking and tool_use blocks) must be
  sent back with the tool results. We always send the whole content back.
- tool_choice "any" and "tool" are rejected in thinking mode, so we never set tool_choice.
"""

import os
from typing import Any

import anthropic

from refuterl.providers.base import ModelTurn, ToolCall, ToolResult, Usage
from refuterl.tools import ToolSpec

DEEPSEEK_BASE_URL = "https://api.deepseek.com/anthropic"
DEFAULT_MAX_TOKENS = 16_000


class AnthropicConversation:
    def __init__(
        self,
        client: Any,
        model: str,
        system: str,
        prompt: str,
        tools: list[ToolSpec],
        max_tokens: int,
        thinking: dict[str, Any] | None,
    ):
        self.client = client
        self.model = model
        self.system = system
        self.tools = [
            {"name": t.name, "description": t.description, "input_schema": t.input_schema}
            for t in tools
        ]
        self.max_tokens = max_tokens
        self.thinking = thinking
        self.messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]

    def send(self) -> ModelTurn:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": self.system,
            "messages": self.messages,
            "tools": self.tools,
        }
        if self.thinking is not None:
            kwargs["thinking"] = self.thinking
        response = self.client.messages.create(**kwargs)

        # Keep the whole content, thinking blocks included, for the next request.
        self.messages.append({"role": "assistant", "content": response.content})

        text_parts = []
        tool_calls = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(id=block.id, name=block.name, input=block.input or {}))

        usage = getattr(response, "usage", None)
        return ModelTurn(
            text="\n".join(text_parts),
            tool_calls=tool_calls,
            stop_reason=response.stop_reason,
            usage=Usage(
                # Cached prompt tokens are reported separately; count them as input too.
                input_tokens=sum(
                    getattr(usage, field, 0) or 0
                    for field in (
                        "input_tokens",
                        "cache_read_input_tokens",
                        "cache_creation_input_tokens",
                    )
                ),
                output_tokens=getattr(usage, "output_tokens", 0) or 0,
            ),
        )

    def add_tool_results(self, results: list[ToolResult]) -> None:
        self.messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": r.call_id,
                        "content": r.content,
                        "is_error": r.is_error,
                    }
                    for r in results
                ],
            }
        )


class AnthropicProvider:
    def __init__(
        self,
        model: str,
        name: str = "anthropic",
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        thinking: dict[str, Any] | None = None,
        client: Any = None,
    ):
        self.name = name
        self.model = model
        self.max_tokens = max_tokens
        self.thinking = thinking
        self.client = client or anthropic.Anthropic(api_key=api_key, base_url=base_url)

    def start(self, system: str, prompt: str, tools: list[ToolSpec]) -> AnthropicConversation:
        return AnthropicConversation(
            self.client, self.model, system, prompt, tools, self.max_tokens, self.thinking
        )


def deepseek(
    model: str = "deepseek-flash", thinking: bool = True, **kwargs: Any
) -> AnthropicProvider:
    """DeepSeek through its Anthropic-compatible endpoint. Reads DEEPSEEK_API_KEY."""
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set. Add it to .env or your environment.")
    return AnthropicProvider(
        model,
        name="deepseek",
        api_key=api_key,
        base_url=DEEPSEEK_BASE_URL,
        thinking={"type": "enabled" if thinking else "disabled"},
        **kwargs,
    )


def claude(model: str, **kwargs: Any) -> AnthropicProvider:
    """Claude through the Anthropic API. Reads ANTHROPIC_API_KEY.

    Thinking is off unless a thinking config is passed, since Claude models differ
    in which thinking settings they accept.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is not set. Add it to .env or your environment.")
    return AnthropicProvider(model, name="claude", **kwargs)
