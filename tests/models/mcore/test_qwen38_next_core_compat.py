# SPDX-License-Identifier: Apache-2.0
"""Opt-in contract for the isolated Core optional-FA4 import patch.

Execute the installed source's actual import guard without importing the rest
of Core. This must fail the AttributeError case on the affected unpatched
version. Real package imports and CUDA/model tests remain separate gates.
"""

import ast
import builtins
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_QWEN38_CORE_COMPAT_TESTS") != "1", reason="explicit isolated Core patch opt-in required"
)


@pytest.mark.parametrize("failure", [None, ImportError, AttributeError, RuntimeError])
def test_optional_fa4_guard_keeps_available_kernel_and_propagates_other_failures(failure):
    spec = importlib.util.find_spec("megatron.core")
    assert spec is not None and spec.submodule_search_locations
    source = Path(next(iter(spec.submodule_search_locations))) / "transformer/attention.py"
    tree = ast.parse(source.read_text())
    guards = [
        node
        for node in tree.body
        if isinstance(node, ast.Try)
        and any(isinstance(item, ast.ImportFrom) and item.module == "flash_attn.cute" for item in node.body)
    ]
    assert len(guards) == 1, "Re-audit the patch if the upstream import structure changes"
    kernel = object()

    def import_fa4(name, globals=None, locals=None, fromlist=(), level=0):
        assert name == "flash_attn.cute" and tuple(fromlist) == ("flash_attn_varlen_func",) and level == 0
        if failure is not None:
            raise failure("optional FA4 probe")
        return SimpleNamespace(flash_attn_varlen_func=kernel)

    namespace = {"__builtins__": {**vars(builtins), "__import__": import_fa4}}
    code = compile(ast.Module(body=guards, type_ignores=[]), str(source), "exec")
    if failure is RuntimeError:
        with pytest.raises(RuntimeError, match="optional FA4 probe"):
            exec(code, namespace)
        assert "HAVE_FA4" not in namespace
    else:
        exec(code, namespace)
        assert namespace["HAVE_FA4"] is (failure is None)
        if failure is None:
            assert namespace["flash_attn4_varlen_func"] is kernel
        else:
            assert "flash_attn4_varlen_func" not in namespace
