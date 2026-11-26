# RefuteRL

RefuteRL is a benchmark and RL environment for finding counterexamples to math claims.

A model gets a claim, such as "every n with property P also has property Q", and tries to break it. It can run Python in a sandbox to search, then submits a candidate. A checker decides whether the candidate really breaks the claim, and that decides the reward.

The project is at an early stage. The layout is in place, but most modules are still empty.

## How it works

- Each claim is a Python file with a statement, a `check(candidate)` function, and some metadata.
- The model has two tools: `run_python` to run code in a Docker sandbox, and `submit` to give its answer.
- Tool use has a budget, so a good claim is one where naive search will not find the counterexample in time.
- Claims come from famous false conjectures, changed versions of them, templated families, and LLM-proposed claims that a person reviews.

v1 covers number theory. Graph theory comes later.

## Setup

You need [uv](https://docs.astral.sh/uv/) and Docker.

```bash
uv sync
docker build -t refuterl-sandbox sandbox/
```

Run the tests and linter:

```bash
uv run pytest
uv run ruff check .
```

## Layout

```
claims/            claim files (number_theory/, pending/ for review)
refuterl/          the package: claims, verification, sandbox, providers, generator
sandbox/           Dockerfile for the code sandbox
results/           eval outputs
tests/
```

## Provider

v1 uses DeepSeek through its Anthropic-compatible API, so the Anthropic SDK works with a different base URL. The same code can run Claude.
