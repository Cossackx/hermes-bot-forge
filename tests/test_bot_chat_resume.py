"""Resuming a Bot Chat is policy-frozen in Hermes core: any creation flag that differs from the chat's
launch is refused. bot_chat must pass none on resume, and must survive a refusal it could not predict.
The real `hermes` is mocked at forge.run."""
import subprocess
import unittest
from pathlib import Path
from unittest import mock

from support import forge

REFUSAL = ("Error: Resume retains gateway session policy; creation overrides are unsupported on resume. "
           "--max-turns differs from this session's launch: resume without it, or start a new session to use it.")


def _runner(replies):
    """forge.run stand-in: pops one CompletedProcess per call, records argv."""
    calls = []

    def fake(root, *args, **kw):
        calls.append(args)
        rc, out, err = replies.pop(0)
        return subprocess.CompletedProcess(["hermes", *args], rc, out, err)

    return fake, calls


class ResumePassesNoCreationFlags(unittest.TestCase):
    def test_resume_carries_no_policy_flag(self):
        fake, calls = _runner([(0, "✅ done", "")])
        with mock.patch.object(forge, "run", fake), mock.patch.object(forge, "has_bot_chat", return_value=True):
            ok, reply = forge.bot_chat(Path("/tmp"), "nova", "hi", max_turns=30)
        self.assertTrue(ok)
        argv = calls[0]
        self.assertIn("--create-if-missing", argv)
        for flag in ("--max-turns", "-t", "-m", "--model", "--provider", "--reasoning"):
            self.assertNotIn(flag, argv, flag)

    def test_reading_only_on_an_existing_chat_uses_a_fresh_session(self):
        fake, calls = _runner([(0, "✅ looked", "")])
        detail = {}
        with mock.patch.object(forge, "run", fake), mock.patch.object(forge, "has_bot_chat", return_value=True):
            ok, _ = forge.bot_chat(Path("/tmp"), "nova", "hi", reading_only=True, detail=detail)
        self.assertTrue(ok)
        self.assertNotIn("--create-if-missing", calls[0])  # -t is a creation flag; never on a resume
        self.assertIn("-t", calls[0])
        self.assertEqual(detail["session"], "fresh")
        self.assertIsNone(detail["resume_refused"])


class RefusedResumeFallsBackOnce(unittest.TestCase):
    def test_refusal_reruns_in_a_fresh_session_and_reports_it(self):
        fake, calls = _runner([(2, "", REFUSAL), (0, "✅ done elsewhere", "")])
        detail = {}
        with mock.patch.object(forge, "run", fake), mock.patch.object(forge, "has_bot_chat", return_value=True):
            ok, reply = forge.bot_chat(Path("/tmp"), "nova", "hi", detail=detail)
        self.assertTrue(ok)
        self.assertEqual(reply, "✅ done elsewhere")
        self.assertEqual(len(calls), 2)
        self.assertIn("--create-if-missing", calls[0])
        self.assertNotIn("--create-if-missing", calls[1])
        self.assertEqual(detail["session"], "fresh")
        self.assertIn("Resume retains gateway session policy", detail["resume_refused"])

    def test_an_ordinary_failure_is_not_retried(self):
        fake, calls = _runner([(1, "", "boom: model not found")])
        detail = {}
        with mock.patch.object(forge, "run", fake), mock.patch.object(forge, "has_bot_chat", return_value=True):
            ok, reply = forge.bot_chat(Path("/tmp"), "nova", "hi", detail=detail)
        self.assertFalse(ok)
        self.assertEqual(len(calls), 1)
        self.assertEqual(detail["session"], "bot-chat")

    def test_first_turn_creates_the_chat_without_flags(self):
        fake, calls = _runner([(0, "✅ hello", ""), (0, "", "")])
        with mock.patch.object(forge, "run", fake), mock.patch.object(forge, "has_bot_chat", return_value=False), \
             mock.patch.object(forge, "bot_mode_install", return_value=True), \
             mock.patch.object(forge, "newest_session", return_value="sid"):
            ok, _ = forge.bot_chat(Path("/tmp"), "nova", "hi", max_turns=30)
        self.assertTrue(ok)
        self.assertNotIn("--max-turns", calls[0])
        self.assertIn("--source", calls[0])


if __name__ == "__main__":
    unittest.main()
