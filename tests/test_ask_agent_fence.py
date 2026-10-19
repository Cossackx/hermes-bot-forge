"""ask_agent must not hand a Kanban worker's identity to the Bot it asks (local Raz fork patch).

The asked Bot's environment comes from forge.safe_env, an allowlist that already drops every
HERMES_KANBAN_* variable and all credentials. Dropping TASK alone would make the asked Bot look
like an orchestrator on the worker's own board, so Hermes core's delegated-child write fence
(agent.delegation_context) must travel with it, and ask_agent refuses to spawn when that core
isolation cannot be imported.

This module needs Hermes core and is mandatory for this fork: it finds core through
HERMES_AGENT_SOURCE, PYTHONPATH, or the standard install location, and errors loudly (never skips)
when none is present. Example: PYTHONPATH=<hermes-agent> python -m unittest discover -s tests
No LLM or board is touched: subprocess.run is captured and board writes are made to explode.
"""
import builtins
import importlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from support import ROOT, make_root, tools


def _core_source():
    candidates = [os.environ.get("HERMES_AGENT_SOURCE", "")]
    base = os.environ.get("LOCALAPPDATA") if os.name == "nt" else ""
    candidates += [str(Path(base) / "hermes" / "hermes-agent") if base else "",
                   str(Path.home() / ".hermes" / "hermes-agent")]
    for c in candidates:
        if c and (Path(c) / "agent" / "delegation_context.py").is_file() and c not in sys.path:
            sys.path.append(c)  # appended: never shadows the plugin's own modules
            return


_core_source()
from agent import delegation_context as dc  # noqa: E402  (loud failure without Hermes core)

WORKER = {
    "HERMES_KANBAN_TASK": "t_parent", "HERMES_KANBAN_RUN_ID": "7", "HERMES_KANBAN_CLAIM_LOCK": "lock",
    "HERMES_KANBAN_GOAL_MODE": "1", "HERMES_KANBAN_GOAL_MAX_TURNS": "5",
}
MARKER = dc.DELEGATED_CHILD_ENV_MARKER


def _core(module):
    """A core module whose package name the plugin's tools.py may shadow (support caches it as
    ``tools``)."""
    cached = sys.modules.pop("tools", None)
    saved = list(sys.path)
    sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() not in (ROOT, ROOT.parent)]
    try:
        return importlib.import_module(module)
    finally:
        sys.path[:] = saved
        if module.startswith("tools."):
            for name in [m for m in sys.modules if m == "tools" or m.startswith("tools.")]:
                sys.modules.pop(name)
        if cached is not None:
            sys.modules["tools"] = cached


class AskAgentFence(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.tmp = Path(temporary.name)
        self.root = make_root(self.tmp / "hermes", profiles=("bob",))
        self.captured = []

        def fake_run(cmd, **kw):
            self.captured.append(dict(kw["env"]))
            return subprocess.CompletedProcess(cmd, 0, stdout="hello from bob", stderr="")

        base = {k: v for k, v in os.environ.items() if k != MARKER and not k.startswith("HERMES_KANBAN")}
        base.update({"HERMES_HOME": str(self.root / "profiles" / "worker"), "OPENAI_API_KEY": "sk-secret",
                     "HERMES_KANBAN_HOME": str(self.root), "HERMES_KANBAN_DB": str(self.root / "kanban.db")})
        self.base = base
        for p in (mock.patch.object(tools, "hermes_root", return_value=self.root),
                  mock.patch.object(tools.subprocess, "run", side_effect=fake_run)):
            p.start()
            self.addCleanup(p.stop)

    def ask(self, env):
        with mock.patch.dict(os.environ, env, clear=True):
            before = dict(os.environ)
            out = json.loads(tools.ask_agent({"name": "bob", "message": "hi"}))
            self.assertEqual(dict(os.environ), before, "parent env must not be mutated")
        self.assertTrue(out["ok"], out)
        child = self.captured[-1]
        self.assertEqual(sorted(k for k in child if k.startswith("HERMES_KANBAN")), [])
        self.assertNotIn("OPENAI_API_KEY", child)  # credentials never forwarded
        self.assertNotIn("sk-secret", json.dumps(out))
        return child

    def assert_board_fenced(self, child, *paths):
        """Core's own write-path guard, evaluated in the child's environment exactly as emitted."""
        kanban_db = _core("hermes_cli.kanban_db")
        with mock.patch.dict(os.environ, child, clear=True):
            self.assertEqual(dc.owned_kanban_task(), "")
            for path in paths:
                self.assertTrue(dc.kanban_path_is_fenced(path), path)
                with self.assertRaises(PermissionError):
                    kanban_db._assert_not_delegated_child_mutation(path)

    def test_worker_identity_scrubbed_and_fence_propagated(self):
        child = self.ask({**self.base, **WORKER})
        self.assertEqual(Path(child[MARKER]).resolve(), self.root.resolve())  # the lineage's board root
        self.assertEqual(child["HERMES_HOME"], str(self.root))                 # upstream contract kept
        self.assert_board_fenced(child, self.root / "kanban.db", self.root / "kanban" / "boards" / "x")

        kt = _core("tools.kanban_tools")

        def no_db(*a, **k):
            raise AssertionError("guarded mutation reached the board")

        with mock.patch.dict(os.environ, child, clear=True), mock.patch.object(kt, "_board", no_db):
            for handler, args in ((kt._handle_complete, {"task_id": "t_parent", "summary": "x"}),
                                  (kt._handle_block, {"task_id": "t_parent", "reason": "x"}),
                                  (kt._handle_heartbeat, {"task_id": "t_parent"})):
                res = json.loads(handler(args))
                self.assertIn("refused", res.get("error", ""), res)

        # The worker itself (outside the child's env) still owns its task.
        with mock.patch.dict(os.environ, {**self.base, **WORKER}, clear=True):
            self.assertEqual(dc.owned_kanban_task(), "t_parent")

    def test_distinct_kanban_home_is_the_fenced_root(self):
        board = self.tmp / "board-home"
        board.mkdir()
        env = {**self.base, **WORKER, "HERMES_KANBAN_HOME": str(board), "HERMES_KANBAN_DB": str(board / "kanban.db")}
        child = self.ask(env)
        self.assertEqual(Path(child[MARKER]).resolve(), board.resolve())
        self.assertEqual(child["HERMES_HOME"], str(self.root))
        self.assert_board_fenced(child, board / "kanban.db")

    def test_pinned_db_outside_the_fence_root_fences_every_board(self):
        elsewhere = self.tmp / "elsewhere"
        elsewhere.mkdir()
        child = self.ask({**self.base, **WORKER, "HERMES_KANBAN_DB": str(elsewhere / "worker.db")})
        self.assertEqual(child[MARKER], "1")
        self.assert_board_fenced(child, elsewhere / "worker.db", self.root / "kanban.db")

    def test_ordinary_caller_gets_no_fence(self):
        child = self.ask(dict(self.base))
        self.assertNotIn(MARKER, child)
        with mock.patch.dict(os.environ, self.base, clear=True):
            self.assertEqual(child, tools.forge.safe_env(self.root))  # exactly the upstream allowlist

    def test_windows_home_reaches_the_child(self):
        child = self.ask({**self.base, "USERPROFILE": str(self.tmp)})
        self.assertEqual(child["USERPROFILE"], str(self.tmp))

    def test_isolation_import_failure_prevents_spawn(self):
        original = builtins.__import__

        def fail_isolation(name, *args, **kwargs):
            if name == "agent.delegation_context":
                raise ImportError("isolation unavailable")
            return original(name, *args, **kwargs)

        for env in (dict(self.base), {**self.base, **WORKER}):
            with mock.patch.dict(os.environ, env, clear=True), \
                    mock.patch("builtins.__import__", side_effect=fail_isolation):
                result = json.loads(tools.ask_agent({"name": "bob", "message": "hi"}))
            self.assertFalse(result["ok"])
            self.assertIn("isolation is unavailable", result["error"])
        self.assertEqual(self.captured, [], "unsafe subprocess must not launch")

    def test_already_fenced_descendant_keeps_its_inherited_root(self):
        fenced = self.tmp / "fenced-root"
        fenced.mkdir()
        env = {k: v for k, v in self.base.items() if k != "HERMES_KANBAN_DB"}
        child = self.ask({**env, MARKER: str(fenced)})
        self.assertEqual(child[MARKER], str(fenced))
        self.assert_board_fenced(child, fenced / "kanban.db")

    def test_inherited_fence_everything_marker_is_not_narrowed(self):
        child = self.ask({**self.base, MARKER: "1"})
        self.assertEqual(child[MARKER], "1")
        self.assert_board_fenced(child, self.root / "kanban.db", self.tmp / "any" / "other.db")

    def test_in_process_delegate_child_is_fenced(self):
        with dc.delegated_child_context():
            child = self.ask(dict(self.base))
        self.assertEqual(Path(child[MARKER]).resolve(), self.root.resolve())
        self.assert_board_fenced(child, self.root / "kanban.db")


if __name__ == "__main__":
    unittest.main()
