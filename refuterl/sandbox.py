"""Docker sandbox for model-written code and unreviewed checkers.

One container per episode. Each run_python() call starts a new Python process
inside that container, so files written to /tmp persist between calls but
variables do not. The container is removed when the episode ends.

    with Sandbox() as box:
        result = box.run_python("print(2 ** 10)", timeout=30)

Build the image first: docker build -t refuterl-sandbox sandbox/
"""

import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from typing import IO

DEFAULT_IMAGE = "refuterl-sandbox"
DEFAULT_OUTPUT_LIMIT = 10_000
TIMEOUT_EXIT_CODE = 124  # what coreutils `timeout` returns when it stops a command
BACKSTOP_SECONDS = 5  # extra host-side wait before the whole container is killed


class SandboxError(Exception):
    """The sandbox could not be started or is no longer usable."""


@dataclass(frozen=True)
class SandboxLimits:
    memory: str = "1g"
    cpus: float = 1.0
    pids: int = 64
    tmp_size: str = "64m"


@dataclass(frozen=True)
class RunResult:
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool
    duration: float


def _read_capped(stream: IO[bytes], limit: int, out: list[str]) -> None:
    """Keep the first `limit` bytes of a stream, count and discard the rest."""
    kept = stream.read(limit)
    dropped = 0
    while chunk := stream.read(65536):
        dropped += len(chunk)
    text = kept.decode("utf-8", errors="replace")
    if dropped:
        text += f"\n[output truncated: {dropped} more bytes]"
    out.append(text)


class Sandbox:
    def __init__(
        self,
        image: str = DEFAULT_IMAGE,
        limits: SandboxLimits = SandboxLimits(),
        output_limit: int = DEFAULT_OUTPUT_LIMIT,
    ):
        self.image = image
        self.limits = limits
        self.output_limit = output_limit
        self.name: str | None = None

    def start(self) -> None:
        if self.name is not None:
            raise SandboxError("sandbox already started")
        name = f"refuterl-{uuid.uuid4().hex[:12]}"
        lim = self.limits
        cmd = [
            "docker", "run", "--detach", "--rm",
            "--name", name,
            "--network", "none",
            "--memory", lim.memory,
            "--memory-swap", lim.memory,  # same as memory, so no swap
            "--cpus", str(lim.cpus),
            "--pids-limit", str(lim.pids),
            "--read-only",
            "--tmpfs", f"/tmp:rw,size={lim.tmp_size}",
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--user", "1000:1000",
            "--workdir", "/tmp",
            "--env", "PYTHONPATH=/tmp",
            "--env", "PYTHONDONTWRITEBYTECODE=1",
            self.image,
            "sleep", "infinity",
        ]  # fmt: skip
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise SandboxError(f"could not start container: {proc.stderr.strip()}")
        self.name = name

    def stop(self) -> None:
        if self.name is None:
            return
        subprocess.run(["docker", "rm", "--force", self.name], capture_output=True)
        self.name = None

    def run_python(self, code: str, timeout: float) -> RunResult:
        """Run code with python in the container. The code is passed on stdin."""
        if self.name is None:
            raise SandboxError("sandbox is not running")

        cmd = [
            "docker", "exec", "--interactive", self.name,
            # TERM at the deadline, KILL 2 seconds later if it is still alive.
            "timeout", "--kill-after=2", f"{timeout}s",
            "python", "-",
        ]  # fmt: skip
        start = time.monotonic()
        proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout: list[str] = []
        stderr: list[str] = []
        readers = [
            threading.Thread(target=_read_capped, args=(proc.stdout, self.output_limit, stdout)),
            threading.Thread(target=_read_capped, args=(proc.stderr, self.output_limit, stderr)),
        ]
        for t in readers:
            t.start()
        try:
            proc.stdin.write(code.encode())
            proc.stdin.close()
        except BrokenPipeError:
            pass

        try:
            exit_code = proc.wait(timeout=timeout + BACKSTOP_SECONDS)
        except subprocess.TimeoutExpired:
            # The in-container timeout did not work. Kill the whole container;
            # the episode cannot use this sandbox any more.
            self.stop()
            proc.kill()
            proc.wait()
            for t in readers:
                t.join()
            return RunResult(
                stdout="".join(stdout),
                stderr="".join(stderr) + "\n[sandbox killed after backstop timeout]",
                exit_code=-1,
                timed_out=True,
                duration=time.monotonic() - start,
            )

        for t in readers:
            t.join()
        duration = time.monotonic() - start
        # 137 means KILL, which is either timeout's kill-after or running out of memory.
        timed_out = exit_code == TIMEOUT_EXIT_CODE or (exit_code == 137 and duration >= timeout)
        return RunResult(
            stdout="".join(stdout),
            stderr="".join(stderr),
            exit_code=exit_code,
            timed_out=timed_out,
            duration=duration,
        )

    def __enter__(self) -> "Sandbox":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()
