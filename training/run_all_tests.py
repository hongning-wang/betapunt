import importlib.util
import inspect
import os
import sys
from pathlib import Path


def _load_module_from_path(path: Path):
    module_name = f"_local_test_{path.stem}_{abs(hash(str(path)))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _find_test_files(root: Path):
    patterns = ["test_*.py", "*_test.py"]
    files = []
    for pattern in patterns:
        files.extend(root.rglob(pattern))
    return sorted({p for p in files if p.is_file()})


def _run_tests_in_module(module, module_path: Path):
    tests = [
        (name, fn)
        for name, fn in inspect.getmembers(module, inspect.isfunction)
        if name.startswith("test_") and fn.__module__ == module.__name__
    ]

    if not tests:
        print(f"{module_path}: no test_ functions found")
        return 0, 0

    passed = 0
    failed = 0

    for name, fn in tests:
        print(f"{module_path}::{name} ... ", end="")
        try:
            fn()
            print("PASSED")
            passed += 1
        except Exception as exc:
            print("FAILED")
            print(f"  Error: {exc}")
            failed += 1

    return passed, failed


def main():
    project_root = Path(__file__).resolve().parent
    sys.path.append(str(project_root))

    test_files = _find_test_files(project_root)
    if not test_files:
        print("No test files found.")
        return

    total_passed = 0
    total_failed = 0

    for test_path in test_files:
        print(f"\nLoading {test_path} ...")
        try:
            module = _load_module_from_path(test_path)
        except Exception as exc:
            print(f"FAILED to import: {exc}")
            total_failed += 1
            continue

        passed, failed = _run_tests_in_module(module, test_path)
        total_passed += passed
        total_failed += failed

    print("\nSummary:")
    print(f"  Passed: {total_passed}")
    print(f"  Failed: {total_failed}")


if __name__ == "__main__":
    main()
