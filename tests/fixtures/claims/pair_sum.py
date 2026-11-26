STATEMENT = "No pair (a, b) has a + b == 10 and a > b"
CANDIDATE_TYPE = "tuple[int, int]"
LABEL = "open"
DIFFICULTY = "easy"
SOURCE = "templated"


def check(pair):
    assert isinstance(pair, tuple)
    a, b = pair
    return a + b == 10 and a > b
