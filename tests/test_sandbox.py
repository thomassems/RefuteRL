"""Sandbox tests. They need Docker running and the refuterl-sandbox image built."""

import subprocess

import pytest

from refuterl.sandbox import DEFAULT_IMAGE, Sandbox, SandboxError


def _image_available() -> bool:
    try:
        proc = subprocess.run(
            ["docker", "image", "inspect", DEFAULT_IMAGE], capture_output=True, timeout=10
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0


pytestmark = pytest.mark.skipif(
    not _image_available(),
    reason="needs Docker and the image: docker build -t refuterl-sandbox sandbox/",
)


@pytest.fixture(scope="module")
def box():
    with Sandbox() as sandbox:
        yield sandbox


def test_runs_code(box):
    result = box.run_python("print(2 ** 100)", timeout=10)
    assert result.exit_code == 0
    assert result.stdout.strip() == str(2**100)
    assert not result.timed_out


def test_files_persist_but_variables_do_not(box):
    box.run_python(
        "x = 5\nopen('/tmp/helpers.py', 'w').write('def double(n):\\n    return 2 * n\\n')",
        timeout=10,
    )
    result = box.run_python("from helpers import double\nprint(double(21))", timeout=10)
    assert result.stdout.strip() == "42"
    result = box.run_python("print(x)", timeout=10)
    assert "NameError" in result.stderr


def test_errors_are_reported(box):
    result = box.run_python("1 / 0", timeout=10)
    assert result.exit_code != 0
    assert "ZeroDivisionError" in result.stderr


def test_timeout_stops_the_process(box):
    result = box.run_python("while True:\n    pass", timeout=2)
    assert result.timed_out
    assert result.duration < 8
    # The container is still usable afterwards.
    assert box.run_python("print('ok')", timeout=10).stdout.strip() == "ok"


def test_output_is_capped(box):
    result = box.run_python("print('a' * 1_000_000)", timeout=10)
    assert len(result.stdout) < 11_000
    assert "output truncated" in result.stdout


def test_no_network(box):
    code = "import socket\nsocket.create_connection(('1.1.1.1', 53), timeout=2)"
    result = box.run_python(code, timeout=10)
    assert result.exit_code != 0


def test_root_filesystem_is_read_only(box):
    result = box.run_python("open('/home/runner/x', 'w').write('x')", timeout=10)
    assert "Read-only file system" in result.stderr


def test_math_libraries_available(box):
    result = box.run_python("import sympy, numpy, gmpy2\nprint(sympy.isprime(97))", timeout=30)
    assert result.stdout.strip() == "True"


def test_stopped_sandbox_refuses_to_run():
    sandbox = Sandbox()
    with pytest.raises(SandboxError):
        sandbox.run_python("print(1)", timeout=5)
