"""Quiet CLI output loss must not rerun or misattribute a completed specialist turn."""
from contextlib import closing
import json
import sqlite3
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from support import make_root, tools


class AskAgentReply(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1])
        self.addCleanup(self.tmp.cleanup)
        self.root = make_root(Path(self.tmp.name), profiles=("bob",))
        self.db = self.root / "profiles" / "bob" / "state.db"
        with closing(sqlite3.connect(self.db)) as c, c:
            c.executescript("""
                create table sessions(id text, ended_at real, end_reason text);
                create table messages(id integer primary key, session_id text, role text,
                                      content text, finish_reason text, tool_calls text);
            """)
        self.saved("20261022_204553_4ec43b", "saved — café ✓")
        # Another request can finish later on the same profile, even with an identical prompt.
        self.saved("newer_concurrent", "NOT this invocation")

    def saved(self, sid, reply):
        with closing(sqlite3.connect(self.db)) as c, c:
            c.execute("insert into sessions values (?, 123, 'cli_close')", (sid,))
            c.execute("insert into messages values (null, ?, 'user', 'hi', null, null)", (sid,))
            c.execute("insert into messages values (null, ?, 'assistant', ?, 'stop', null)", (sid, reply))

    def ask(self, stdout="", stderr="session_id: 20261022_204553_4ec43b\n", rc=0, error=None, name="bob"):
        with patch.object(tools, "hermes_root", return_value=self.root), \
             patch.object(tools.subprocess, "run", side_effect=error,
                          return_value=subprocess.CompletedProcess([], rc, stdout, stderr)) as run:
            result = json.loads(tools.ask_agent({"name": name, "message": "hi"}))
        self.assertEqual(run.call_count, 1)  # recovery is a read, never another worker turn
        self.assertEqual(run.call_args.kwargs["encoding"], "utf-8")
        self.assertEqual(run.call_args.kwargs["cwd"], str(Path.home()))
        return result

    def test_session_id_only_recovers_exact_completed_session_not_newest(self):
        out = self.ask()
        self.assertTrue(out["ok"], out)
        self.assertEqual(out["reply"], "saved — café ✓")
        self.assertEqual(out["session_id"], "20261022_204553_4ec43b")
        self.assertEqual(out["reply_source"], "session")

    def test_stdout_fallback_is_not_filtered_or_truncated(self):
        # With no validated saved reply for this invocation, stdout is the reply, verbatim.
        reply = "hermes update is just quoted text\n" + "x" * 7000
        out = self.ask(stdout=reply + "\n", stderr="session_id: missing\n")
        self.assertEqual(out["reply"], reply)
        self.assertEqual(out["reply_source"], "stdout")

    def test_saved_final_reply_beats_stdout_tool_previews(self):
        # Quiet-mode stdout can be dominated by coloured patch previews; the saved final
        # assistant message for this exact session is the specialist's actual answer.
        preview = "\x1b[2m\u2502\x1b[0m @@ -1,3 +1,4 @@\n\x1b[32m+ added line\x1b[0m\n" * 400
        out = self.ask(stdout=preview + "saved — café ✓\n")
        self.assertTrue(out["ok"], out)
        self.assertEqual(out["reply"], "saved — café ✓")
        self.assertEqual(out["reply_source"], "session")

    def test_unvalidated_saved_reply_falls_back_to_stdout(self):
        with closing(sqlite3.connect(self.db)) as c, c:
            c.execute("update sessions set end_reason='interrupted'")
        out = self.ask(stdout="answer from stdout\n")
        self.assertEqual(out["reply"], "answer from stdout")
        self.assertEqual(out["reply_source"], "stdout")

    def test_real_worker_failure_does_not_promote_saved_text(self):
        out = self.ask(rc=1, stderr="provider failed\nsession_id: 20261022_204553_4ec43b\n")
        self.assertFalse(out["ok"])
        self.assertEqual(out["error_kind"], "worker_failure")
        self.assertNotIn("reply", out)

    def test_no_identity_missing_db_or_ambiguous_identity_stays_unknown(self):
        for stderr in ("", "session_id: missing", "session_id: a\nsession_id: b", "session_id: ../../state"):
            with self.subTest(stderr=stderr):
                out = self.ask(stderr=stderr)
                self.assertFalse(out["ok"])
                self.assertEqual(out["execution_status"], "unknown_effect")
        self.db.unlink()
        self.assertFalse(self.ask()["ok"])
        self.assertFalse(self.db.exists())  # read-only means no accidental database creation

    def test_incomplete_failed_empty_or_wrong_prompt_sessions_are_not_recovered(self):
        for sql in (
            "update sessions set ended_at=null",
            "update sessions set end_reason='interrupted'",
            "update messages set content='' where role='assistant'",
            "update messages set finish_reason='length' where role='assistant'",
            "update messages set role='tool' where role='assistant'",
            "update messages set tool_calls='[]' where role='assistant'",
            "update messages set content='different request' where role='user'",
        ):
            with self.subTest(sql=sql):
                with closing(sqlite3.connect(self.db)) as c, c:
                    c.execute(sql)
                out = self.ask()
                self.assertFalse(out["ok"], out)
                with closing(sqlite3.connect(self.db)) as c, c:
                    c.execute("delete from sessions")
                    c.execute("delete from messages")
                self.saved("20261022_204553_4ec43b", "saved — café ✓")

    def test_timeout_keeps_identity_and_unknown_effect_even_if_saved_reply_exists(self):
        out = self.ask(error=subprocess.TimeoutExpired("hermes", 900, stderr=b"session_id: 20261022_204553_4ec43b\n"))
        self.assertFalse(out["ok"])
        self.assertEqual(out["session_id"], "20261022_204553_4ec43b")
        self.assertEqual(out["execution_status"], "unknown_effect")
        self.assertEqual(out["error_kind"], "timeout")

    def test_default_profile_uses_root_database(self):
        self.db.replace(self.root / "state.db")
        self.assertEqual(self.ask(name="default")["reply"], "saved — café ✓")

    def test_database_error_does_not_escape(self):
        self.db.write_text("not sqlite")
        self.assertEqual(self.ask()["error_kind"], "output_failure")


if __name__ == "__main__":
    unittest.main()
