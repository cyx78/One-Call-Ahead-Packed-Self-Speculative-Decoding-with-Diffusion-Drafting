"Execute HumanEval base tests with resource limits."

from __future__ import annotations
import builtins
import json
import resource
import sys
import traceback
from pathlib import Path
from typing import Any

ALLOWED_IMPORTS = {
    "bisect",
    "collections",
    "copy",
    "dataclasses",
    "decimal",
    "fractions",
    "functools",
    "hashlib",
    "heapq",
    "itertools",
    "math",
    "numpy",
    "operator",
    "random",
    "re",
    "statistics",
    "string",
    "sys",
    "typing",
}


class _SafeSys:
    """Expose the supported sys functions to generated code."""

    getsizeof = staticmethod(sys.getsizeof)


SAFE_SYS = _SafeSys()


def _limits() -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
    resource.setrlimit(resource.RLIMIT_NPROC, (16, 16))


def _safe_import(
    name: str,
    globals: dict[str, Any] | None = None,
    locals: dict[str, Any] | None = None,
    fromlist: tuple[str, ...] = (),
    level: int = 0,
) -> Any:
    del globals, locals
    top_level = name.split(".", 1)[0]
    if level != 0 or top_level not in ALLOWED_IMPORTS:
        raise ImportError(f"import blocked by benchmark sandbox: {name}")
    if top_level == "sys":
        return SAFE_SYS
    return builtins.__import__(name, fromlist=fromlist, level=level)


def _safe_builtins() -> dict[str, Any]:
    blocked = {"breakpoint", "exit", "help", "input", "open", "quit"}
    result = {
        name: value for (name, value) in vars(builtins).items() if name not in blocked
    }
    result["__import__"] = _safe_import
    return result


def main() -> None:
    payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    _limits()
    namespace: dict[str, Any] = {
        "__builtins__": _safe_builtins(),
        "__name__": "__candidate__",
    }
    source = str(payload["candidate"]) + "\n\n" + str(payload["tests"])
    if payload["kind"] == "humaneval":
        source += f"\ncheck({payload['entry_point']})\n"
    try:
        compiled = compile(source, "<candidate>", "exec")
        exec(compiled, namespace, namespace)
    except BaseException as error:
        result = {
            "passed": False,
            "outcome": "assertion_failure"
            if isinstance(error, AssertionError)
            else "runtime_error",
            "error_type": type(error).__name__,
            "error_message": str(error)[:500],
            "traceback_tail": traceback.format_exc(limit=2)[-1200:],
        }
    else:
        result = {
            "passed": True,
            "outcome": "passed",
            "error_type": None,
            "error_message": None,
            "traceback_tail": None,
        }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
