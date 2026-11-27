"""Run a model against the claim set and write results.

One episode = one model attempting one claim. The model sees the claim, can
call run_python within its budget, and ends by calling submit. The submitted
answer is verified and the episode is written as one JSON line.

    uv run refuterl-eval --profile quick --limit 1
    uv run refuterl-eval --model deepseek-v4-pro --no-tools
"""

import argparse
import json
import os
import re
import time
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from refuterl.claim import Claim, load_claims
from refuterl.providers.base import Provider, ToolResult
from refuterl.sandbox import Sandbox
from refuterl.tools import NONE_FOUND, PROFILES, BudgetProfile, CodeRunner, ToolRunner, tool_specs
from refuterl.verify import Verdict, reward_for, verify

ROOT = Path(__file__).resolve().parent.parent
FINAL_TEXT_LIMIT = 2_000

SandboxFactory = Callable[[BudgetProfile], AbstractContextManager[CodeRunner]]


@dataclass
class EpisodeResult:
    claim_id: str
    source: str
    difficulty: str
    label: str
    provider: str
    model: str
    profile: str
    tools_enabled: bool
    end_reason: str  # submitted, none_found, no_submit, max_turns, error
    verdict: str | None  # None only when end_reason is error
    reward: float
    submitted: str | None
    detail: str
    tool_calls: int
    turns: int
    input_tokens: int
    output_tokens: int
    seconds: float
    final_text: str


# Prompts


def describe_type(candidate_type: str) -> str:
    if candidate_type == "int":
        return 'a single integer, for example "40"'
    if candidate_type == "list[int]":
        return 'a JSON list of integers, for example "[3, 4, 5]"'
    if candidate_type.startswith("tuple["):
        size = candidate_type.count("int")
        example = "[" + ", ".join(str(i + 2) for i in range(size)) + "]"
        return f'a JSON list of exactly {size} integers, for example "{example}"'
    return candidate_type


def system_prompt(profile: BudgetProfile, tools_enabled: bool) -> str:
    lines = [
        "You are given a mathematical claim. Your job is to find a counterexample: a specific "
        "input that makes the claim false.",
        "",
        "Only a counterexample that a checker confirms earns credit. Guessing does not help.",
    ]
    if tools_enabled:
        lines += [
            "",
            f"You can call run_python up to {profile.max_tool_calls} times, and each call is "
            f"stopped after {profile.seconds_per_call:g} seconds. Naive brute force is often too "
            "slow, so think about where a counterexample is likely to be before searching.",
        ]
    else:
        lines += ["", "You cannot run code. Reason it out."]
    lines += [
        "",
        f'When you are done, call submit with your counterexample, or with "{NONE_FOUND}" if you '
        "did not find one. Calling submit ends the attempt.",
    ]
    return "\n".join(lines)


def user_prompt(claim: Claim) -> str:
    return (
        f"Claim: {claim.statement}\n\n"
        f"Submit the counterexample as {describe_type(claim.candidate_type)}."
    )


# Episodes


@contextmanager
def docker_sandbox(profile: BudgetProfile) -> Iterator[CodeRunner]:
    with Sandbox(limits=profile.sandbox_limits()) as box:
        yield box


def run_episode(
    claim: Claim,
    provider: Provider,
    profile: BudgetProfile,
    tools_enabled: bool = True,
    sandbox_factory: SandboxFactory = docker_sandbox,
    check_timeout: float = 10.0,
) -> EpisodeResult:
    start = time.monotonic()
    turns = 0
    input_tokens = output_tokens = 0
    submitted: str | None = None
    final_text = ""
    end_reason = "max_turns"
    runner: ToolRunner | None = None

    try:
        sandbox_ctx = sandbox_factory(profile) if tools_enabled else nullcontext(None)
        with sandbox_ctx as box:
            runner = ToolRunner(profile, box)
            conversation = provider.start(
                system_prompt(profile, tools_enabled),
                user_prompt(claim),
                tool_specs(tools_enabled),
            )
            for _ in range(profile.max_turns):
                turn = conversation.send()
                turns += 1
                input_tokens += turn.usage.input_tokens
                output_tokens += turn.usage.output_tokens
                if turn.text:
                    final_text = turn.text

                if not turn.tool_calls:
                    end_reason = "no_submit"
                    break

                results = []
                for call in turn.tool_calls:
                    outcome = runner.handle(call.name, call.input)
                    if outcome.submitted is not None:
                        submitted = outcome.submitted
                        break
                    results.append(ToolResult(call.id, outcome.content, outcome.is_error))
                if submitted is not None:
                    end_reason = "submitted"
                    break
                conversation.add_tool_results(results)
    except Exception as e:  # one bad episode must not stop the run
        return _result(
            claim, provider, profile, tools_enabled, "error", None, submitted,
            f"{type(e).__name__}: {e}", runner, turns, input_tokens, output_tokens,
            start, final_text,
        )  # fmt: skip

    if submitted is not None and submitted.lower() == NONE_FOUND:
        end_reason = "none_found"
    if end_reason != "submitted":
        verdict, detail = Verdict.NO_ANSWER, ""
    else:
        check = verify(claim, submitted, timeout=check_timeout)
        verdict, detail = check.verdict, check.detail

    return _result(
        claim, provider, profile, tools_enabled, end_reason, verdict, submitted, detail,
        runner, turns, input_tokens, output_tokens, start, final_text,
    )  # fmt: skip


def _result(
    claim: Claim,
    provider: Provider,
    profile: BudgetProfile,
    tools_enabled: bool,
    end_reason: str,
    verdict: Verdict | None,
    submitted: str | None,
    detail: str,
    runner: ToolRunner | None,
    turns: int,
    input_tokens: int,
    output_tokens: int,
    start: float,
    final_text: str,
) -> EpisodeResult:
    return EpisodeResult(
        claim_id=claim.id,
        source=claim.source,
        difficulty=claim.difficulty,
        label=claim.label,
        provider=provider.name,
        model=provider.model,
        profile=profile.name,
        tools_enabled=tools_enabled,
        end_reason=end_reason,
        verdict=verdict.value if verdict else None,
        reward=reward_for(verdict) if verdict else 0.0,
        submitted=submitted,
        detail=detail,
        tool_calls=runner.calls_used if runner else 0,
        turns=turns,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        seconds=round(time.monotonic() - start, 2),
        final_text=final_text[:FINAL_TEXT_LIMIT],
    )


# Runs


def run_eval(
    claims: list[Claim],
    provider: Provider,
    profile: BudgetProfile,
    out_path: Path,
    tools_enabled: bool = True,
    sandbox_factory: SandboxFactory = docker_sandbox,
    log: Callable[[str], None] = print,
) -> list[EpisodeResult]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = []
    with out_path.open("w") as f:
        for i, claim in enumerate(claims, 1):
            result = run_episode(claim, provider, profile, tools_enabled, sandbox_factory)
            results.append(result)
            f.write(json.dumps(asdict(result)) + "\n")
            f.flush()
            log(
                f"[{i}/{len(claims)}] {claim.id}: {result.verdict or 'error'} "
                f"({result.end_reason}, {result.tool_calls} tool calls, {result.seconds}s)"
            )
    return results


def summarize(results: list[EpisodeResult]) -> str:
    scored = [r for r in results if r.end_reason != "error"]
    errors = len(results) - len(scored)
    refuted = sum(r.reward for r in scored)
    header = f"Refuted {refuted:g} of {len(scored)} claims"
    lines = [header + (f" ({errors} errors)" if errors else "")]
    for key in ("source", "difficulty"):
        totals = Counter(getattr(r, key) for r in scored)
        wins = Counter(getattr(r, key) for r in scored if r.reward)
        parts = [f"{k} {wins[k]}/{n}" for k, n in sorted(totals.items())]
        lines.append(f"  by {key}: " + ", ".join(parts))
    lines.append(f"  verdicts: {dict(Counter(r.verdict for r in scored))}")
    tokens_in = sum(r.input_tokens for r in results)
    tokens_out = sum(r.output_tokens for r in results)
    lines.append(f"  tokens: {tokens_in} in, {tokens_out} out")
    return "\n".join(lines)


# Command line


def load_dotenv(path: Path) -> None:
    """Set variables from a .env file without overriding ones already set."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def make_provider(args: argparse.Namespace) -> Provider:
    from refuterl.providers.anthropic import claude, deepseek

    if args.provider == "deepseek":
        return deepseek(args.model or "deepseek-flash", thinking=not args.no_thinking)
    if not args.model:
        raise SystemExit("--model is required for --provider claude")
    return claude(args.model)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a model against RefuteRL claims.")
    parser.add_argument("--provider", choices=["deepseek", "claude"], default="deepseek")
    parser.add_argument("--model", help="model name (default: deepseek-flash for deepseek)")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="standard")
    parser.add_argument("--no-tools", action="store_true", help="offer only the submit tool")
    parser.add_argument("--no-thinking", action="store_true", help="turn off DeepSeek thinking")
    parser.add_argument("--claims", type=Path, default=ROOT / "claims" / "number_theory")
    parser.add_argument("--claim", action="append", help="only run this claim id (repeatable)")
    parser.add_argument("--limit", type=int, help="run at most this many claims")
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    args = parser.parse_args(argv)

    load_dotenv(ROOT / ".env")
    provider = make_provider(args)
    profile = PROFILES[args.profile]

    # Scoring option A: only claims with a known counterexample are scored.
    claims = [c for c in load_claims(args.claims) if c.label == "false"]
    if args.claim:
        claims = [c for c in claims if c.id in args.claim]
    if args.limit:
        claims = claims[: args.limit]
    if not claims:
        raise SystemExit("No claims matched.")

    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    model_slug = re.sub(r"[^A-Za-z0-9.-]+", "-", provider.model)
    mode = "" if not args.no_tools else "-notools"
    out_path = args.out / f"{stamp}-{provider.name}-{model_slug}-{profile.name}{mode}.jsonl"

    print(f"Running {len(claims)} claims with {provider.name}/{provider.model}, "
          f"profile {profile.name}, tools {'off' if args.no_tools else 'on'}")  # fmt: skip
    results = run_eval(claims, provider, profile, out_path, not args.no_tools)
    print(summarize(results))
    print(f"Results: {out_path}")


if __name__ == "__main__":
    main()
