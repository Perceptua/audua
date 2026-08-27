#!/usr/bin/env python3
"""Run the test suite with pytest if available, else with a built-in shim.

Normally you want ``uv run pytest``. This script exists as a fallback for a
machine with no PyPI access, where pytest cannot be installed at all. The shim
implements only the pytest surface these tests use: ``approx``, ``raises``,
``fixture``, ``mark.parametrize``, ``skip``, and the ``tmp_path`` fixture.

    python tools/run_tests.py
"""

from __future__ import annotations

import importlib.util
import inspect
import shutil
import sys
import tempfile
import traceback
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
TESTS = ROOT / "tests"


# --------------------------------------------------------------------------
# minimal pytest stand-in
# --------------------------------------------------------------------------

class Skipped(Exception):
    pass


class _Approx:
    def __init__(self, expected, abs_=None, rel=None):
        self.expected = expected
        self.abs = abs_
        self.rel = rel if rel is not None else 1e-6

    def _close(self, actual, expected) -> bool:
        if self.abs is not None:
            return abs(actual - expected) <= self.abs
        return abs(actual - expected) <= max(abs(expected), abs(actual)) * self.rel + 1e-12

    def __eq__(self, other):
        if isinstance(self.expected, (list, tuple)):
            if not isinstance(other, (list, tuple)) or len(other) != len(self.expected):
                return False
            return all(self._close(a, b) for a, b in zip(other, self.expected))
        return self._close(other, self.expected)

    def __repr__(self):
        return f"approx({self.expected!r})"


class _Raises:
    def __init__(self, expected):
        self.expected = expected

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            raise AssertionError(f"Expected {self.expected} to be raised, nothing was.")
        return issubclass(exc_type, self.expected)


def _fixture(func=None, **_kwargs):
    def wrap(f):
        f._is_fixture = True
        return f
    return wrap(func) if func is not None else wrap


class _Mark:
    @staticmethod
    def parametrize(argnames, argvalues):
        names = [n.strip() for n in argnames.split(",")]

        def decorator(func):
            cases = []
            for value in argvalues:
                if len(names) == 1:
                    cases.append({names[0]: value})
                else:
                    cases.append(dict(zip(names, value)))
            func._parametrize = cases
            return func
        return decorator

    @staticmethod
    def usefixtures(*names):
        def decorator(obj):
            return obj
        return decorator

    def __getattr__(self, item):  # tolerate unknown marks
        def decorator(*args, **kwargs):
            def inner(func):
                return func
            return inner if args and callable(args[0]) is False else (lambda f: f)
        return decorator


def _install_shim() -> None:
    module = types.ModuleType("pytest")
    module.approx = lambda expected, abs=None, rel=None: _Approx(expected, abs, rel)
    module.raises = _Raises
    module.fixture = _fixture
    module.mark = _Mark()
    module.skip = lambda reason="": (_ for _ in ()).throw(Skipped(reason))
    module.Skipped = Skipped
    sys.modules["pytest"] = module


# --------------------------------------------------------------------------
# collection and execution
# --------------------------------------------------------------------------

def _load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def _resolve(name: str, module, cache: dict, tmp_root: Path):
    if name in cache:
        return cache[name]

    if name == "tmp_path":
        path = Path(tempfile.mkdtemp(dir=tmp_root))
        cache[name] = path
        return path

    fixture = getattr(module, name, None)
    if fixture is None or not getattr(fixture, "_is_fixture", False):
        raise LookupError(f"No fixture named {name!r}")

    kwargs = {
        param: _resolve(param, module, cache, tmp_root)
        for param in inspect.signature(fixture).parameters
    }
    value = fixture(**kwargs)
    if inspect.isgenerator(value):
        value = next(value)
    cache[name] = value
    return value


def main() -> int:
    if importlib.util.find_spec("pytest") is None:
        _install_shim()
        print("pytest not installed - using built-in shim\n")
    else:  # pragma: no cover
        import pytest
        return pytest.main([str(TESTS), "-q"])

    # src/ layout: the package is not importable from the repo root.
    sys.path.insert(0, str(SRC))
    tmp_root = Path(tempfile.mkdtemp(prefix="audua-tests-"))

    passed = failed = skipped = 0
    failures: list[tuple[str, str]] = []

    for test_file in sorted(TESTS.glob("test_*.py")):
        module = _load(test_file)
        names = [n for n in dir(module) if n.startswith("test_")]
        print(f"{test_file.name}")

        for name in sorted(names):
            func = getattr(module, name)
            if not callable(func):
                continue
            cases = getattr(func, "_parametrize", [{}])

            for case in cases:
                label = f"  {name}" + (f"[{list(case.values())}]" if case else "")
                cache: dict = dict(case)
                try:
                    kwargs = {
                        param: cache[param] if param in cache
                        else _resolve(param, module, cache, tmp_root)
                        for param in inspect.signature(func).parameters
                    }
                    func(**kwargs)
                except Skipped as exc:
                    skipped += 1
                    print(f"{label} SKIP ({exc})")
                except Exception:
                    failed += 1
                    failures.append((label.strip(), traceback.format_exc()))
                    print(f"{label} FAIL")
                else:
                    passed += 1

    print()
    for label, tb in failures:
        print("=" * 70)
        print(label)
        print(tb)

    print("=" * 70)
    print(f"{passed} passed, {failed} failed, {skipped} skipped")
    shutil.rmtree(tmp_root, ignore_errors=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
