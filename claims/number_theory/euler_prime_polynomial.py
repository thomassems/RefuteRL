STATEMENT = "n^2 + n + 41 is prime for every integer n >= 0"
CANDIDATE_TYPE = "int"
LABEL = "false"
DIFFICULTY = "easy"
SOURCE = "famous"
KNOWN_COUNTEREXAMPLE = 40


def is_prime(m):
    if m < 2:
        return False
    if m % 2 == 0:
        return m == 2
    d = 3
    while d * d <= m:
        if m % d == 0:
            return False
        d += 2
    return True


def check(n):
    return n >= 0 and not is_prime(n * n + n + 41)
