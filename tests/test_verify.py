from pathlib import Path

import pytest

from refuterl.claim import load_claim
from refuterl.verify import (
    MAX_INT_DIGITS,
    InvalidCandidate,
    Verdict,
    parse_candidate,
    reward_for,
    validate_candidate,
    verify,
)

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).parent / "fixtures" / "claims"
EULER = load_claim(ROOT / "claims" / "number_theory" / "euler_prime_polynomial.py")


def fixture(name):
    return load_claim(FIXTURES / f"{name}.py")


# Parsing and validation


@pytest.mark.parametrize("raw", ["40.0", "1e3", "NaN", "Infinity", "not json", "[1, 2"])
def test_parse_rejects_non_integers_and_bad_json(raw):
    with pytest.raises(InvalidCandidate):
        parse_candidate(raw)


def test_parse_rejects_too_many_digits():
    with pytest.raises(InvalidCandidate, match="digits"):
        parse_candidate("9" * (MAX_INT_DIGITS + 1))


def test_parse_accepts_large_int_within_cap():
    assert parse_candidate("9" * 5000) == int("9" * 5000)


@pytest.mark.parametrize("value", [True, False, "40", [40], None])
def test_int_type_rejects_wrong_shapes(value):
    with pytest.raises(InvalidCandidate):
        validate_candidate(value, "int")


def test_list_type():
    assert validate_candidate([1, 2, 3], "list[int]") == [1, 2, 3]
    for bad in ([], [1, True], [1, "2"], 5):
        with pytest.raises(InvalidCandidate):
            validate_candidate(bad, "list[int]")


def test_tuple_type():
    assert validate_candidate([7, 3], "tuple[int, int]") == (7, 3)
    for bad in ([7], [7, 3, 1], [7, False]):
        with pytest.raises(InvalidCandidate):
            validate_candidate(bad, "tuple[int, int]")


def test_unknown_type_is_a_claim_bug_not_a_candidate_error():
    with pytest.raises(ValueError, match="unknown candidate_type"):
        validate_candidate(1, "float")


# Full pipeline


def test_known_counterexample_refutes():
    assert verify(EULER, "40").verdict == Verdict.REFUTED


def test_non_counterexample():
    assert verify(EULER, "39").verdict == Verdict.NOT_REFUTED


def test_bool_loophole_is_closed():
    assert verify(EULER, "true").verdict == Verdict.INVALID_FORMAT


def test_tuple_reaches_checker_as_tuple():
    claim = fixture("pair_sum")
    assert verify(claim, "[7, 3]").verdict == Verdict.REFUTED
    assert verify(claim, "[3, 7]").verdict == Verdict.NOT_REFUTED


def test_huge_candidate_times_out_instead_of_hanging():
    result = verify(EULER, "1" + "0" * 60, timeout=1)
    assert result.verdict == Verdict.CHECKER_TIMEOUT


def test_hanging_checker_times_out():
    assert verify(fixture("hangs"), "1", timeout=1).verdict == Verdict.CHECKER_TIMEOUT


def test_non_bool_return_is_checker_error():
    result = verify(fixture("returns_int"), "1")
    assert result.verdict == Verdict.CHECKER_ERROR
    assert "not bool" in result.detail


def test_exception_is_checker_error():
    result = verify(fixture("raises"), "1")
    assert result.verdict == Verdict.CHECKER_ERROR
    assert "ZeroDivisionError" in result.detail


def test_checker_output_does_not_break_result():
    assert verify(fixture("prints"), "7").verdict == Verdict.REFUTED


def test_reward_only_for_refuted():
    assert reward_for(Verdict.REFUTED) == 1.0
    assert all(reward_for(v) == 0.0 for v in Verdict if v != Verdict.REFUTED)
