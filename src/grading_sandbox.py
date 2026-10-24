"""Build isolated code-grading subprocesses with explicit filesystem mounts."""

from pathlib import Path
import shutil
import subprocess
import sys


def command(argv, *, readonly=(), writable=(), environment=None):
    executable = shutil.which("bwrap")
    if executable is None:
        raise RuntimeError(
            "Code grading requires bubblewrap (bwrap) and Linux user namespaces"
        )
    args = [
        executable,
        "--unshare-all",
        "--die-with-parent",
        "--new-session",
        "--cap-drop",
        "ALL",
        "--clearenv",
    ]
    roots = [
        Path(value)
        for value in ("/usr", "/bin", "/lib", "/lib64")
        if Path(value).exists()
    ]
    for prefix in (Path(sys.prefix), Path(sys.base_prefix)):
        if not any(prefix == root or root in prefix.parents for root in roots):
            roots.append(prefix)
    for path in [*roots, *map(Path, readonly)]:
        args += ["--ro-bind", str(path), str(path)]
    if Path("/etc/ld.so.cache").exists():
        args += ["--ro-bind", "/etc/ld.so.cache", "/etc/ld.so.cache"]
    args += [
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
        "--dir",
        "/work",
        "--chdir",
        "/work",
    ]
    for path in map(Path, writable):
        args += ["--bind", str(path), str(path)]
    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": "/work",
        "PYTHONHASHSEED": "0",
        "PYTHONUNBUFFERED": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        **(environment or {}),
    }
    for key, value in env.items():
        args += ["--setenv", key, str(value)]
    return [*args, "--", *map(str, argv)]


def preflight():
    result = subprocess.run(
        command([sys.executable, "-I", "-c", 'print("sandbox_ready")']),
        capture_output=True,
        text=True,
        timeout=15,
    )
    if result.returncode != 0 or result.stdout.strip() != "sandbox_ready":
        raise RuntimeError("Code sandbox could not start: " + result.stderr[-1000:])
