import __future__

import ast
import copy
import faulthandler
import io
import json
import multiprocessing
import re
import sys
import time
import traceback
from collections.abc import MutableMapping
from typing import Optional

import numpy as np

INCORRECT_FORMAT = "Incorrect format"
TIMEOUT = "Time out"
ERROR_PREFIX = "Error: "
OUTER_ERROR_PREFIX = "ERROR: "
MAX_ADDITIONAL_MEMORY_BYTES = 1024 * 1024 * 1024
DEFAULT_TIMEOUT = 1
TIMEOUT_SCALER = 1.0
FORMAT_PENALTY = False

FILENAME = "Solution.py"
TESTS_FILENAME = "Tests.py"
CONTEXT_FILENAME = "Context.py"
DEBUG_BUFFER_NAME = "__debug_buffer__"
DEBUG_PRINT_NAME = "debug_print"

PROTECTED_GLOBAL_NAMES = {DEBUG_PRINT_NAME, DEBUG_BUFFER_NAME}

try:
    import resource as _resource
except Exception:
    _resource = None


def set_memory_limits(maximum_memory_bytes: Optional[int]) -> None:
    if maximum_memory_bytes is None or maximum_memory_bytes <= 0 or _resource is None:
        return

    try:
        if hasattr(_resource, "RLIMIT_AS"):
            _resource.setrlimit(_resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
        for limit_name in ("RLIMIT_DATA", "RLIMIT_RSS"):
            if hasattr(_resource, limit_name):
                limit_const = getattr(_resource, limit_name)
                try:
                    _resource.setrlimit(limit_const, (maximum_memory_bytes, maximum_memory_bytes))
                except Exception:
                    print("Failed to set memory limits")
                    pass
    except Exception:
        print("Failed to set memory limits")
        pass


def _build_restricted_builtins():
    import builtins as _builtins

    allowed_names = {
        "abs": _builtins.abs,
        "all": _builtins.all,
        "any": _builtins.any,
        "bool": _builtins.bool,
        "bytes": _builtins.bytes,
        "callable": _builtins.callable,
        "chr": _builtins.chr,
        "dict": _builtins.dict,
        "enumerate": _builtins.enumerate,
        "filter": _builtins.filter,
        "float": _builtins.float,
        "format": _builtins.format,
        "frozenset": _builtins.frozenset,
        "hash": _builtins.hash,
        "hex": _builtins.hex,
        "int": _builtins.int,
        "isinstance": _builtins.isinstance,
        "issubclass": _builtins.issubclass,
        "iter": _builtins.iter,
        "len": _builtins.len,
        "list": _builtins.list,
        "map": _builtins.map,
        "max": _builtins.max,
        "min": _builtins.min,
        "next": _builtins.next,
        "object": _builtins.object,
        "ord": _builtins.ord,
        "pow": _builtins.pow,
        "print": _builtins.print,
        "range": _builtins.range,
        "repr": _builtins.repr,
        "reversed": _builtins.reversed,
        "round": _builtins.round,
        "set": _builtins.set,
        "slice": _builtins.slice,
        "sorted": _builtins.sorted,
        "str": _builtins.str,
        "sum": _builtins.sum,
        "tuple": _builtins.tuple,
        "zip": _builtins.zip,
        "input": _builtins.input,
    }

    allowed_modules = {
        "math",
        "cmath",
        "itertools",
        "functools",
        "operator",
        "statistics",
        "random",
        "collections",
        "heapq",
        "bisect",
        "array",
        "string",
        "re",
        "typing",
        "json",
        "io",
        "fractions",
        "decimal",
        "dataclasses",
        "datetime",
        "time",
        "sys",
        "sortedcontainers",
        "numpy",
    }

    real_import = _builtins.__import__

    def restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
        root = name.split(".")[0]
        if root not in allowed_modules:
            raise ImportError(f"Import of module '{name}' is not allowed")
        return real_import(name, globals, locals, fromlist, level)

    allowed_names["open"] = None
    allowed_names["__import__"] = restricted_import

    allowed_names["__build_class__"] = _builtins.__build_class__

    return allowed_names


def _capture_stderr(namespace):
    old_stderr = sys.stderr
    sys.stderr = namespace[DEBUG_BUFFER_NAME]
    return old_stderr


def _create_sandbox_namespace(extra_globals=None):
    ns = {"__builtins__": _build_restricted_builtins()}

    _debug_buffer = io.StringIO()

    def debug_print(*args, **kwargs):
        sep = kwargs.get("sep", " ")
        end = kwargs.get("end", "\n")
        try:
            text = sep.join(str(a) for a in args) + end
        except Exception:
            text = " ".join(["<unprintable>"] * len(args)) + end
        _debug_buffer.write(text)

    ns[DEBUG_PRINT_NAME] = debug_print
    ns[DEBUG_BUFFER_NAME] = _debug_buffer

    try:
        import typing as _typing
        _common_typing_names = (
            "Any",
            "Optional",
            "Union",
            "List",
            "Dict",
            "Set",
            "Tuple",
            "Callable",
            "Iterable",
            "Iterator",
            "Sequence",
            "Mapping",
            "MutableMapping",
            "MutableSequence",
            "MutableSet",
            "DefaultDict",
            "Deque",
            "FrozenSet",
            "Type",
            "TypeVar",
            "Generic",
            "Literal",
            "TypedDict",
            "NoReturn",
            "overload",
        )
        for _name in _common_typing_names:
            if hasattr(_typing, _name):
                ns[_name] = getattr(_typing, _name)
    except Exception:
        pass
    if "__name__" not in ns:
        ns["__name__"] = "__not_main__"
    if extra_globals:
        ns.update(extra_globals)
    return ns


def _exec_with_isolated_locals(code_obj, globals_ns):
    class _GuardedLocals(MutableMapping):
        def __init__(self, backing):
            self._b = backing
        def __getitem__(self, key):
            return self._b[key]
        def __setitem__(self, key, value):
            if key in PROTECTED_GLOBAL_NAMES:
                return
            self._b[key] = value
        def __delitem__(self, key):
            if key in PROTECTED_GLOBAL_NAMES:
                return
            del self._b[key]
        def __iter__(self):
            return iter(self._b)
        def __len__(self):
            return len(self._b)

    exec(code_obj, globals_ns, _GuardedLocals(globals_ns))


def _to_safe_jsonable(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_to_safe_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_safe_jsonable(v) for k, v in value.items()}
    raise TypeError(f"Non-serializable result type: {type(value).__name__}")


def reliability_guard():

    faulthandler.disable()

    try:
        import warnings as _warnings
        _warnings.filterwarnings("ignore", category=SyntaxWarning)
    except Exception:
        pass

    import builtins

    set_memory_limits(MAX_ADDITIONAL_MEMORY_BYTES)

    builtins.exit = None
    builtins.quit = None
    builtins.open = None

    import os

    os.environ["OMP_NUM_THREADS"] = "1"

    os.kill = None
    os.system = None
    os.putenv = None
    os.remove = None
    os.removedirs = None
    os.rmdir = None
    os.fchdir = None
    os.setuid = None
    os.fork = None
    os.forkpty = None
    os.killpg = None
    os.rename = None
    os.renames = None
    os.truncate = None
    os.replace = None
    os.unlink = None
    os.fchmod = None
    os.fchown = None
    os.chmod = None
    os.chown = None
    os.chroot = None
    os.fchdir = None
    os.lchflags = None
    os.lchmod = None
    os.lchown = None
    os.getcwd = None
    os.chdir = None

    import shutil

    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None

    import subprocess

    subprocess.Popen = None

    import sys

    sys.modules["ipdb"] = None
    sys.modules["joblib"] = None
    sys.modules["resource"] = None
    sys.modules["psutil"] = None
    sys.modules["tkinter"] = None
    sys.modules["inspect"] = None
    sys.modules["ctypes"] = None
    sys.modules["threading"] = None
    sys.modules["multiprocessing"] = None
    sys.modules["socket"] = None
    sys.modules["ssl"] = None
    sys.modules["urllib"] = None
    sys.modules["requests"] = None

    try:
        sys._getframe = None
    except Exception:
        pass


def _short_trace(e, limit=3):
    frames = traceback.extract_tb(e.__traceback__)
    solution_frames = [
        f for f in frames
        if isinstance(getattr(f, "filename", None), str)
        and f.filename == FILENAME
    ]
    tail = solution_frames[-limit:] if solution_frames else []
    lines = [f"{type(e).__name__}: {e}"]
    for f in tail:
        if f.line:
            lines.append(f"  {f.line}")
        lines.append(f"Line {f.lineno} in {f.name} ({FILENAME})")
    return "\n".join(lines)


def run_test_func(completion, test_input, test_output, fn_name, namespace=None):
    namespace = _create_sandbox_namespace() if namespace is None else namespace
    code_obj = compile(
        completion,
        FILENAME,
        "exec",
        flags=__future__.annotations.compiler_flag,
        dont_inherit=True,
    )
    _exec_with_isolated_locals(code_obj, namespace)

    def _infer_func_name(src, explicit_name):
        try:
            tree = ast.parse(src)
            if explicit_name:
                for node in tree.body:
                    if isinstance(node, ast.FunctionDef) and node.name == explicit_name:
                        return explicit_name
            for node in tree.body:
                if isinstance(node, ast.FunctionDef):
                    return node.name
        except Exception:
            pass
        return None

    if fn_name != "" and fn_name in namespace and callable(namespace[fn_name]):
        func_name = fn_name
    else:
        func_name = _infer_func_name(completion, fn_name)
        if func_name not in namespace or not callable(namespace.get(func_name, None)):
            func_name = completion.split("(")[0].split()[-1]

    output = io.StringIO()
    sys.stdout = output
    old_stderr = _capture_stderr(namespace)

    try:
        if isinstance(test_input, dict):
            result_output = namespace[func_name](**test_input)
        else:
            test_input_args = [json.loads(x) for x in test_input.split()]
            result_output = namespace[func_name](*test_input_args)

        try:
            lhs = _to_safe_jsonable(result_output)
            rhs = _to_safe_jsonable(json.loads(test_output))
            lhs_dump = json.dumps(lhs, sort_keys=True, separators=(",", ":"))
            rhs_dump = json.dumps(rhs, sort_keys=True, separators=(",", ":"))
            if lhs_dump != rhs_dump:
                return False, result_output
            return True, result_output
        except Exception as ser_err:
            error_msg = f"{ERROR_PREFIX}{ser_err}"
            return False, error_msg

    except BaseException as e:
        error_msg = f"{ERROR_PREFIX}{_short_trace(e)}"
        return False, error_msg

    finally:
        sys.stdout = sys.__stdout__
        sys.stderr = old_stderr


def run_test_std(completion, test_input, test_output, namespace=None):
    namespace = _create_sandbox_namespace() if namespace is None else namespace
    output = io.StringIO()
    old_stdout, old_stdin = sys.stdout, sys.stdin
    old_stderr = _capture_stderr(namespace)
    try:
        sys.stdout = output
        sys.stdin = io.StringIO(test_input)
        code_obj = compile('__name__ = "__main__"\n' + completion, FILENAME, "exec")
        _exec_with_isolated_locals(code_obj, namespace)
        out = output.getvalue().strip().replace("\n", " ").replace("\r", "")
        expected = test_output.strip().replace("\n", " ").replace("\r", "")
        return out == expected, output.getvalue().strip()
    except BaseException as e:
        return False, f"{ERROR_PREFIX}{_short_trace(e)}"
    finally:
        sys.stdout = old_stdout
        sys.stdin = old_stdin
        sys.stderr = old_stderr


def run_test_code(completion, test_input, namespace=None):
    namespace = _create_sandbox_namespace() if namespace is None else namespace
    namespace["__name__"] = "__main__"
    code_obj = compile(completion, FILENAME, "exec")
    _exec_with_isolated_locals(code_obj, namespace)

    old_stderr = _capture_stderr(namespace)
    try:
        test_code_obj = compile(test_input, TESTS_FILENAME, "exec")
        _exec_with_isolated_locals(test_code_obj, namespace)
        return True, "All tests pass"
    except BaseException as e:
        return False, f"{ERROR_PREFIX}{_short_trace(e)}"
    finally:
        sys.stderr = old_stderr


def run_tests_for_one_example(test_cases, completion, send_conn, sparse_rewards, test_idx):

    test_type = test_cases["testtype"]
    fn_name = test_cases["fn_name"]

    context = test_cases.get("context", "")
    globals = _create_sandbox_namespace()
    if context.strip() != "":
        code_obj = compile(context, CONTEXT_FILENAME, "exec")
        _exec_with_isolated_locals(code_obj, globals)

    reliability_guard()
    old_stderr = _capture_stderr(globals)

    time_elapsed = float("inf")
    test_input = test_cases["inputs"][test_idx]
    test_output = test_cases["outputs"][test_idx]
    output_value = ""
    test_debug = ""

    try:
        time_start = time.time()
        try:
            if test_type == "functional":
                passed, output_value = run_test_func(
                    completion, copy.deepcopy(test_input), copy.deepcopy(test_output), fn_name, globals
                )
            elif test_type == "stdin":
                test_output = test_output.strip()
                if test_output.endswith("-"):
                    test_output = test_output[: test_output.rfind("-")].rstrip()
                passed, output_value = run_test_std(
                    completion, copy.deepcopy(test_input), copy.deepcopy(test_output), globals
                )
            elif test_type == "code":
                passed, output_value = run_test_code(completion, copy.deepcopy(test_input), globals)
            else:
                raise ValueError(f"Invalid test type: {test_type}")

        except BaseException as e:
            passed = False
            output_value = f"{ERROR_PREFIX}{_short_trace(e)}"

        finally:
            time_elapsed = time.time() - time_start

        if DEBUG_BUFFER_NAME in globals:
            test_debug = globals[DEBUG_BUFFER_NAME].getvalue()

        record = {
            "test_idx": test_idx,
            "input": test_input,
            "expected": test_output,
            "actual": output_value,
            "passed": passed,
            "debug": test_debug,
            "time": time_elapsed,
        }

        try:
            send_conn.send(record)
        except Exception:
            print(f"[{test_idx}] SEND ERROR: {type(e).__name__}: {e}\n")


    except BaseException as outer_e:
        print(f"[{test_idx}] OUTER EXCEPTION: {type(outer_e).__name__}: {outer_e}\n")

        record = {
            "test_idx": test_idx,
            "input": test_input,
            "expected": test_output,
            "actual": f"{OUTER_ERROR_PREFIX}{_short_trace(outer_e)}",
            "passed": False,
            "debug": "",
            "time": float("inf"),
        }
        send_conn.send(record)

    finally:
        sys.stderr = old_stderr
        send_conn.close()


def extract_code(response):
    blocks = re.findall(r"```(\w*)\n(.*?)```", response, re.DOTALL)
    if not blocks:
        return None
    return max((code for _, code in blocks), key=len)


def format_test_feedback(
    records,
    was_truncated=False,
    max_tests_to_show=2,
    sort_test_cases_by_length=True,
    max_length=2000,
    max_input_chars=250,
    max_input_lines=8,
    max_expected_chars=250,
    max_actual_chars=250,
    max_debug_lines=10,
    max_debug_line_chars=300,
):
    if not records:
        return "No test execution information available."

    def _truncate_str(value, max_chars):
        if not isinstance(value, str):
            value = str(value)
        if max_chars is not None and len(value) > max_chars:
            return value[:max_chars] + "..."
        return value

    failing = [r for r in records if not r["passed"]]

    def _first(predicate):
        for rec in records:
            try:
                if predicate(rec):
                    return rec
            except Exception:
                continue
        return None
    selected = (
        _first(lambda rec: isinstance(rec.get("actual"), str) and str(rec.get("actual")).startswith(ERROR_PREFIX))
        or _first(lambda rec: rec.get("actual") == TIMEOUT)
        or _first(lambda rec: rec.get("actual") == INCORRECT_FORMAT)
    )

    if selected is not None:
        failing = [selected]
    else:
        if sort_test_cases_by_length:
            failing = sorted(failing, key=lambda x: len(str(x["input"])) + len(str(x["actual"])))

        if max_tests_to_show is not None:
            failing = failing[: int(max_tests_to_show)]

    if not failing:
        return ""

    parts = []

    def _render_input_block(title, inp):
        parts.append(title)
        if inp is None:
            return
        if isinstance(inp, dict):
            for k, v in inp.items():
                parts.append(f"{k} = {_truncate_str(v, max_input_chars)}")
        else:
            text = str(inp)
            lines = text.splitlines()
            shown = lines[:max_input_lines]
            for line in shown:
                parts.append(_truncate_str(line, max_input_chars))
            if len(lines) > max_input_lines:
                parts.append(f"... ({len(lines) - max_input_lines} more lines)")

    def _render_debug_block(dbg_text):
        dbg = (dbg_text or "").strip()
        if not dbg:
            return
        parts.append("")
        parts.append("Debug Output")
        dbg_lines = dbg.split("\n")
        limit = int(max_debug_lines) if max_debug_lines is not None else None
        for line in dbg_lines[:limit]:
            parts.append(_truncate_str(line, max_debug_line_chars))
        if max_debug_lines is not None and len(dbg_lines) > int(max_debug_lines):
            parts.append(f"... ({len(dbg_lines) - int(max_debug_lines)} more lines)")

    for r in failing:
        test_idx = r["test_idx"] + 1
        actual = r["actual"]
        expected = r["expected"]
        stdin = r["input"]
        debug_text = r["debug"] or ""

        is_error = isinstance(actual, str) and actual.startswith(ERROR_PREFIX)
        is_timeout = actual == TIMEOUT
        is_incorrect_format = actual == INCORRECT_FORMAT

        if is_error:
            parts.append("Runtime Error")
            parts.append(actual[len(ERROR_PREFIX):])
            parts.append("")
            _render_input_block("Last Executed Input", stdin)
            _render_debug_block(debug_text)
        elif is_timeout:
            parts.append("Time Limit Exceeded")
            parts.append("")
            _render_input_block("Last Executed Input", stdin)
            _render_debug_block(debug_text)
        elif is_incorrect_format:
            if was_truncated:
                parts.append("Truncated Attempt: Your previous response was too long and truncated because it reached the maximum response length. Try again with a shorter response.")
            else:
                parts.append("Incorrect Format: Put your code inside a ```python ... ``` block.")
        else:
            parts.append(f"Test Case {test_idx}: Wrong Answer")
            parts.append("")
            _render_input_block("Input", stdin)
            parts.append("")
            parts.append("Output")
            parts.append(_truncate_str(actual, max_actual_chars))
            if expected is not None:
                parts.append("")
                parts.append("Expected")
                parts.append(_truncate_str(expected, max_expected_chars))
            _render_debug_block(debug_text)

        parts.append("")

    result = "\n".join(parts).rstrip()
    if len(result) > max_length:
        result = result[:max_length]
    return result


def run_tests(test_cases: dict, solution, sparse_rewards, max_test_cases):
    completion = extract_code(solution)
    if completion is None:
        return [{
            "test_idx": 0,
            "input": None,
            "expected": None,
            "actual": INCORRECT_FORMAT,
            "passed": False,
            "debug": "",
            "time": float("inf"),
        }]

    num_test_cases = min(max_test_cases, len(test_cases["inputs"])) if max_test_cases else len(test_cases["inputs"])
    timeout_per_test_case = float(test_cases["time_limit"]) if test_cases["time_limit"] is not None else DEFAULT_TIMEOUT

    records = []
    process_data = []

    for test_idx in range(num_test_cases):
        parent_conn, child_conn = multiprocessing.Pipe(duplex=False)
        p = multiprocessing.Process(target=run_tests_for_one_example, args=(test_cases, completion, child_conn, sparse_rewards, test_idx))
        p.start()
        child_conn.close()
        process_data.append({"process": p, "parent_conn": parent_conn})

    start_time = time.time()
    for test_idx, data in enumerate(process_data):
        p = data["process"]
        parent_conn = data["parent_conn"]

        timeout_this_test = max(0, timeout_per_test_case * TIMEOUT_SCALER + 1 - (time.time() - start_time))
        if parent_conn.poll(timeout_this_test):
            try:
                result = parent_conn.recv()
            except Exception as e:
                result = {
                    "test_idx": test_idx,
                    "input": test_cases["inputs"][test_idx],
                    "expected": test_cases["outputs"][test_idx],
                    "actual": f"Process Error: {_short_trace(e)}",
                    "passed": False,
                    "debug": "",
                    "time": float("inf"),
                }
        else:
            result = {
                "test_idx": test_idx,
                "input": test_cases["inputs"][test_idx],
                "expected": test_cases["outputs"][test_idx],
                "actual": TIMEOUT,
                "passed": False,
                "debug": "",
                "time": float("inf"),
            }

        records.append(result)

        p.join(timeout=0)
        if p.is_alive():
            p.kill()
            p.join()

    assert len(records) == num_test_cases
    return records


def compute_score(solution: str, ground_truth: str, extra_info = None, sparse_rewards=False, max_test_cases=None):
    split = extra_info["split"]
    was_truncated = extra_info.get("truncated", False)

    if split == "test":
        sparse_rewards = True

    try:
        test_cases = json.loads(ground_truth)
    except Exception:
        print("Error when reading tests: " + ground_truth[:1000])
        return {
            "score": 0.0,
            "acc": 0.0,
            "pred": "",
            "incorrect_format": 0,
            "error_in_test_cases": 1,
            "timed_out": 0,
            "truncated": 1 if was_truncated else 0,
            "truncated_and_missing_answer": 1 if was_truncated else 0,
            "feedback": "Failed to parse ground truth test cases.",
        }

    records = run_tests(test_cases=test_cases, solution=solution, sparse_rewards=sparse_rewards, max_test_cases=max_test_cases if split != "test" else None)

    correct_answers = [1.0 if r["passed"] else 0.0 for r in records]
    predictions = str([r["actual"] for r in records])[-5000:]
    accuracy = np.mean(correct_answers)

    if sparse_rewards:
        reward = 1.0 if accuracy == 1.0 else 0.0
    else:
        reward = accuracy

    incorrect_format = (len(records) == 1) and (not records[0]["passed"]) and (records[0]["actual"] == INCORRECT_FORMAT)
    error_in_test_cases = any([((not r["passed"]) and isinstance(r["actual"], str) and ERROR_PREFIX in r["actual"]) for r in records])
    timed_out = np.mean([1.0 if (not r["passed"]) and (r["actual"] == TIMEOUT) else 0.0 for r in records])
    if FORMAT_PENALTY and split == "train" and incorrect_format and (not was_truncated):
        reward -= 0.5

    return {
        "score": reward,
        "acc": accuracy,
        "pred": predictions,
        "incorrect_format": 1 if incorrect_format else 0,
        "error_in_test_cases": 1 if error_in_test_cases else 0,
        "timed_out": 1 if timed_out else 0,
        "truncated": 1 if was_truncated else 0,
        "truncated_and_missing_answer": 1 if incorrect_format and was_truncated else 0,
        "feedback": format_test_feedback(records, was_truncated=was_truncated),
    }
