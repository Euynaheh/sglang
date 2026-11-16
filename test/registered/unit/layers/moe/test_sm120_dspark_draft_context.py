"""Exercise DSpark's backend context without loading a model or GPU."""

import ast
import contextlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest


def _draft_context():
    # Compile the real method in isolation; the worker imports CUDA runtime
    # components unrelated to the backend-context lifetime under test.
    import sglang

    source = Path(sglang.__file__).parent / "srt/speculative/dspark_components/dspark_worker_v2.py"
    tree = ast.parse(source.read_text())
    method = next(
        node
        for cls in tree.body
        if isinstance(cls, ast.ClassDef)
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "_draft_context"
    )
    events = []

    @contextlib.contextmanager
    def scope(name):
        events.append((name, "enter"))
        try:
            yield
        finally:
            events.append((name, "exit"))

    namespace = {"draft_tp_context": lambda layout: scope("tp")}
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    exec(compile(module, str(source), "exec"), namespace)
    return namespace["_draft_context"], events, scope


@pytest.mark.parametrize("raise_inside", [False, True])
def test_dspark_draft_restores_all_backend_contexts(raise_inside):
    from sglang.srt.layers.moe import utils

    method, events, scope = _draft_context()
    with patch.object(utils, "speculative_moe_backend_context", lambda: scope("runner")), patch.object(
        utils, "speculative_moe_a2a_backend_context", lambda: scope("a2a")
    ):
        try:
            with method(SimpleNamespace(draft_parallel_layout=object())):
                assert events == [("tp", "enter"), ("runner", "enter"), ("a2a", "enter")]
                if raise_inside:
                    raise RuntimeError("draft failed")
        except RuntimeError as exc:
            assert raise_inside and str(exc) == "draft failed"
    assert events[-3:] == [("a2a", "exit"), ("runner", "exit"), ("tp", "exit")]
