"""A 'smoke test': the cheapest possible check that the package imports and is
wired up correctly. If this fails, something fundamental is broken. Add real
tests next to it as you implement each function.

Run all tests with:  pytest
"""

import deepssf


def test_package_imports():
    assert deepssf is not None


def test_version_is_set():
    assert isinstance(deepssf.__version__, str)
    assert deepssf.__version__  # non-empty
