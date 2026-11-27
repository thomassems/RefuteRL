# RefuteRL

RefuteRL is a benchmark and RL environment for finding counterexamples to math claims.

A model gets a claim, such as "every n with property P also has property Q", and tries to break it. It can run Python in a sandbox to search, then submits a candidate. A checker decides whether the candidate really breaks the claim, and that decides the reward.

The project is at an early stage. You can run a model against the claims and get scored results. The claim set is still tiny and the claim generators are not written yet.

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

The sandbox tests are skipped if Docker is not running or the image is not built.

## Running an eval

Copy `.env.example` to `.env` and add your DeepSeek API key. Then:

```bash
uv run refuterl-eval --profile quick --limit 1
```

Useful options:

- `--model deepseek-v4-pro` to use the larger model (default is `deepseek-flash`)
- `--profile standard` for the full budget (10 code runs, 30 seconds each)
- `--no-tools` to test reasoning alone, without code
- `--claim <id>` to run one claim
- `--provider claude --model <name>` to run Claude instead (needs `ANTHROPIC_API_KEY`)

Each run writes one JSON line per claim to `results/` and prints a summary.

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
