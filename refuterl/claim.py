"""Claim schema and loader.

Each claim is one Python file. The file name (without .py) is the claim id.
A claim file defines these module-level names:

    STATEMENT = "n^2 + n + 41 is prime for every n >= 0"
    CANDIDATE_TYPE = "int"
    LABEL = "false"
    DIFFICULTY = "easy"
    SOURCE = "famous"
    KNOWN_COUNTEREXAMPLE = 40   # optional, required when LABEL is "false"

    def check(candidate):
        # Return True if the candidate breaks the claim.
        ...

Loading a claim runs its file in this process, so only load reviewed claims
here. Files in claims/pending/ are untrusted and must not go through load_claim.
"""

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LABELS = ("false", "true", "open")
DIFFICULTIES = ("easy", "medium", "hard")
SOURCES = ("famous", "perturbed", "templated", "llm")

_MISSING = object()


class ClaimError(Exception):
    """A claim file is missing fields or has invalid values."""


@dataclass(frozen=True)
class Claim:
    id: str
    path: Path
    statement: str
    candidate_type: str
    label: str
    difficulty: str
    source: str
    known_counterexample: Any = None


def load_claim(path: str | Path) -> Claim:
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(f"refuterl_claim_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ClaimError(f"{path}: cannot load as a Python file")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def field(name: str, choices: tuple[str, ...] | None = None) -> Any:
        value = getattr(module, name, _MISSING)
        if value is _MISSING:
            raise ClaimError(f"{path}: missing {name}")
        if choices is not None and value not in choices:
            raise ClaimError(f"{path}: {name} must be one of {choices}, got {value!r}")
        return value

    if not callable(getattr(module, "check", None)):
        raise ClaimError(f"{path}: missing check() function")

    label = field("LABEL", LABELS)
    known = getattr(module, "KNOWN_COUNTEREXAMPLE", None)
    if label == "false" and known is None:
        raise ClaimError(f"{path}: false claims need KNOWN_COUNTEREXAMPLE")

    return Claim(
        id=path.stem,
        path=path,
        statement=field("STATEMENT"),
        candidate_type=field("CANDIDATE_TYPE"),
        label=label,
        difficulty=field("DIFFICULTY", DIFFICULTIES),
        source=field("SOURCE", SOURCES),
        known_counterexample=known,
    )


def load_claims(directory: str | Path) -> list[Claim]:
    """Load every claim file under a directory, sorted by id."""
    paths = sorted(p for p in Path(directory).rglob("*.py") if not p.name.startswith("_"))
    return [load_claim(p) for p in paths]
