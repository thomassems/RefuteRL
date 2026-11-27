"""Validate model candidates and run claim checkers with timeouts.

The flow is: parse the raw answer as JSON, validate it against the claim's
candidate_type, then run the claim's check() in a separate Python process that
is killed if it runs too long. The result is a Verdict, not a reward;
reward_for() maps verdicts to rewards.

Checkers run in a subprocess, not in this process, because a single large
integer operation cannot be interrupted by an in-process timeout.
"""

import json
import re
import subprocess
import sys
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from refuterl.claim import Claim

MAX_RAW_CHARS = 100_000
MAX_INT_DIGITS = 10_000
MAX_LIST_LEN = 1_000
DEFAULT_CHECK_TIMEOUT = 10.0

# Python refuses int <-> str conversions above 4300 digits by default.
# Raise the limit so candidates up to MAX_INT_DIGITS can be parsed and passed on.
_limit = sys.get_int_max_str_digits()
if _limit and _limit < MAX_INT_DIGITS + 10:
    sys.set_int_max_str_digits(MAX_INT_DIGITS + 10)

_TUPLE_TYPE = re.compile(r"tuple\[int(?:, int)+\]")


class Verdict(StrEnum):
    REFUTED = "refuted"
    NOT_REFUTED = "not_refuted"
    INVALID_FORMAT = "invalid_format"
    CHECKER_TIMEOUT = "checker_timeout"
    CHECKER_ERROR = "checker_error"
    # Set by the eval loop, not by verify(): the model said "none", never
    # submitted, or ran out of turns.
    NO_ANSWER = "no_answer"


@dataclass(frozen=True)
class VerifyResult:
    verdict: Verdict
    detail: str = ""


class InvalidCandidate(ValueError):
    """The model's answer does not have the shape the claim expects."""


def reward_for(verdict: Verdict) -> float:
    """Scoring option A: only a real counterexample earns reward."""
    return 1.0 if verdict == Verdict.REFUTED else 0.0


def parse_candidate(raw: str) -> Any:
    """Parse a raw answer as JSON, allowing only integers, lists and nothing else numeric."""
    if len(raw) > MAX_RAW_CHARS:
        raise InvalidCandidate(f"answer is longer than {MAX_RAW_CHARS} characters")

    def parse_int(text: str) -> int:
        if len(text.lstrip("-")) > MAX_INT_DIGITS:
            raise InvalidCandidate(f"integer has more than {MAX_INT_DIGITS} digits")
        return int(text)

    def reject_float(text: str) -> None:
        raise InvalidCandidate(f"decimals are not allowed: {text[:40]}")

    def reject_constant(text: str) -> None:
        raise InvalidCandidate(f"{text} is not allowed")

    try:
        return json.loads(
            raw,
            parse_int=parse_int,
            parse_float=reject_float,
            parse_constant=reject_constant,
        )
    except json.JSONDecodeError as e:
        raise InvalidCandidate(f"not valid JSON: {e.msg}") from None


def _is_int(value: Any) -> bool:
    # bool is a subclass of int in Python, so True would otherwise pass as 1.
    return type(value) is int


def validate_candidate(value: Any, candidate_type: str) -> Any:
    """Check the value matches candidate_type. Returns the value, as a tuple for tuple types."""
    if candidate_type == "int":
        if not _is_int(value):
            raise InvalidCandidate(f"expected an integer, got {type(value).__name__}")
        return value

    if candidate_type == "list[int]":
        if not isinstance(value, list):
            raise InvalidCandidate(f"expected a list of integers, got {type(value).__name__}")
        if not 1 <= len(value) <= MAX_LIST_LEN:
            raise InvalidCandidate(f"list must have 1 to {MAX_LIST_LEN} items, got {len(value)}")
        if not all(_is_int(v) for v in value):
            raise InvalidCandidate("every list item must be an integer")
        return value

    if _TUPLE_TYPE.fullmatch(candidate_type):
        size = candidate_type.count("int")
        if not isinstance(value, list | tuple) or len(value) != size:
            raise InvalidCandidate(f"expected exactly {size} integers")
        if not all(_is_int(v) for v in value):
            raise InvalidCandidate("every item must be an integer")
        return tuple(value)

    raise ValueError(f"unknown candidate_type: {candidate_type!r}")


# Runs inside the checker subprocess. Reads {"path", "candidate", "as_tuple"} on stdin
# and writes one JSON line to stdout. Anything check() prints goes to stderr instead.
_WORKER = """
import importlib.util, json, sys
sys.set_int_max_str_digits(0)
out = sys.stdout
sys.stdout = sys.stderr
payload = json.loads(sys.stdin.read())
def reply(**kw):
    out.write(json.dumps(kw))
    out.flush()
try:
    spec = importlib.util.spec_from_file_location("claim_under_check", payload["path"])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    candidate = payload["candidate"]
    if payload["as_tuple"]:
        candidate = tuple(candidate)
    result = module.check(candidate)
except BaseException as e:
    reply(error=f"{type(e).__name__}: {e}")
else:
    if type(result) is bool:
        reply(result=result)
    else:
        reply(error=f"check() returned {type(result).__name__}, not bool")
"""


def run_check(claim: Claim, candidate: Any, timeout: float = DEFAULT_CHECK_TIMEOUT) -> VerifyResult:
    """Run the claim's check() on an already-validated candidate in a subprocess."""
    payload = json.dumps(
        {
            "path": str(claim.path),
            "candidate": candidate,
            "as_tuple": isinstance(candidate, tuple),
        }
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", _WORKER],
            input=payload,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return VerifyResult(Verdict.CHECKER_TIMEOUT, f"check() ran longer than {timeout}s")

    try:
        reply = json.loads(proc.stdout)
    except json.JSONDecodeError:
        detail = f"checker crashed (exit {proc.returncode}) {proc.stderr.strip()[-500:]}"
        return VerifyResult(Verdict.CHECKER_ERROR, detail)

    if "error" in reply:
        return VerifyResult(Verdict.CHECKER_ERROR, reply["error"])
    return VerifyResult(Verdict.REFUTED if reply["result"] else Verdict.NOT_REFUTED)


def verify(claim: Claim, raw: str, timeout: float = DEFAULT_CHECK_TIMEOUT) -> VerifyResult:
    """Full pipeline for a raw model answer: parse, validate, run check()."""
    try:
        candidate = validate_candidate(parse_candidate(raw), claim.candidate_type)
    except InvalidCandidate as e:
        return VerifyResult(Verdict.INVALID_FORMAT, str(e))
    return run_check(claim, candidate, timeout)
