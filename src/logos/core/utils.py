import ast
import json
import multiprocessing as mp
from pathlib import Path
from typing import Any, Iterable

import yaml

SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "pow": pow,
    "print": print,
    "range": range,
    "reversed": reversed,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
    "__import__": __import__,
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")
            count += 1
    return count





def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if isinstance(loaded, dict):
        return loaded
    return {}


def parse_maybe_literal(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return stripped
        try:
            return ast.literal_eval(stripped)
        except (SyntaxError, ValueError):
            return value
    return value


def _discover_callable(namespace: dict[str, Any]) -> Any:
    user_symbols = {
        key: value
        for key, value in namespace.items()
        if not key.startswith("__") and callable(value)
    }
    if not user_symbols:
        return None

    preferred = [
        "solve",
        "solution",
        "main",
        "min_operations_to_equal",
        "max_subarray_sum",
        "factorial",
    ]
    for name in preferred:
        if name in user_symbols:
            return user_symbols[name]

    first_key = sorted(user_symbols.keys())[0]
    return user_symbols[first_key]


def _run_single_test(code: str, test: dict[str, Any]) -> tuple[bool, str]:
    namespace: dict[str, Any] = {"__builtins__": SAFE_BUILTINS.copy()}
    exec(code, namespace, namespace)

    fn = _discover_callable(namespace)
    if fn is None:
        return False, "no callable found"

    raw_input = parse_maybe_literal(test.get("input"))
    expected = parse_maybe_literal(test.get("expected"))

    try:
        if isinstance(raw_input, dict):
            actual = fn(**raw_input)
        elif isinstance(raw_input, tuple):
            actual = fn(*raw_input)
        elif isinstance(raw_input, list):
            try:
                actual = fn(raw_input)
            except TypeError:
                actual = fn(*raw_input)
        else:
            actual = fn(raw_input)
    except TypeError:
        actual = fn()

    if isinstance(expected, float):
        passed = abs(float(actual) - expected) < 1e-6
    else:
        passed = actual == expected
    return passed, "ok" if passed else f"expected {expected!r}, got {actual!r}"


def _worker(code: str, tests: list[dict[str, Any]], queue: mp.Queue) -> None:
    passed = 0
    details: list[dict[str, Any]] = []
    for index, test in enumerate(tests, start=1):
        try:
            ok, message = _run_single_test(code, test)
            passed += int(ok)
            details.append({"index": index, "passed": ok, "message": message})
        except Exception as exc:  # noqa: BLE001
            details.append({"index": index, "passed": False, "message": str(exc)})
    queue.put({"passed": passed, "total": len(tests), "details": details})


def run_code_against_tests(
    code: str,
    tests: list[dict[str, Any]],
    timeout_seconds: float = 2.0,
) -> dict[str, Any]:
    if not code.strip():
        return {"passed": 0, "total": len(tests), "details": [{"message": "empty code"}]}

    if not tests:
        return {"passed": 0, "total": 0, "details": []}

    queue: mp.Queue = mp.Queue()
    process = mp.Process(target=_worker, args=(code, tests, queue))
    process.start()
    process.join(timeout_seconds)

    if process.is_alive():
        process.terminate()
        process.join()
        return {
            "passed": 0,
            "total": len(tests),
            "details": [{"message": f"timeout after {timeout_seconds} seconds"}],
        }

    if queue.empty():
        return {
            "passed": 0,
            "total": len(tests),
            "details": [{"message": "no result from worker"}],
        }

    return queue.get()
