# Image for running model-written code (run_python) and unreviewed checkers.
# Network, CPU and memory limits are set when the container is started, not here.
FROM python:3.12-slim

RUN pip install --no-cache-dir sympy numpy gmpy2

RUN useradd --create-home --uid 1000 runner
USER runner
WORKDIR /home/runner

CMD ["python"]
