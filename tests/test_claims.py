"""Every reviewed claim must load, and every false claim's known counterexample must verify."""

from pathlib import Path

import pytest

from refuterl.claim import load_claims
from refuterl.verify import Verdict, run_check, validate_candidate

CLAIMS_DIR = Path(__file__).resolve().parent.parent / "claims" / "number_theory"
CLAIMS = load_claims(CLAIMS_DIR)


def test_there_are_claims():
    assert CLAIMS


@pytest.mark.parametrize("claim", [c for c in CLAIMS if c.label == "false"], ids=lambda c: c.id)
def test_known_counterexample_refutes(claim):
    candidate = validate_candidate(claim.known_counterexample, claim.candidate_type)
    assert run_check(claim, candidate).verdict == Verdict.REFUTED
