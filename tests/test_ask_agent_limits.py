"""Ask workers must honor profile turns and a finite configured wait."""
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from support import make_root, tools

class AskAgentLimits(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=make_root(Path(self.tmp.name),profiles=('bob',))

    def ask(self,settings=None,error=None):
        with patch.object(tools,'hermes_root',return_value=self.root), patch.object(tools.subprocess,'run',side_effect=error,return_value=subprocess.CompletedProcess([],0,'done','')) as run:
            out=json.loads(tools.ask_agent({'name':'bob','message':'bounded test'},settings=settings))
        return out,run

    def test_target_profile_turn_limit_is_not_overridden(self):
        out,run=self.ask()
        self.assertTrue(out['ok'],out)
        self.assertNotIn('--max-turns',run.call_args.args[0])

    def test_default_wait_is_finite_thirty_minutes(self):
        out,run=self.ask()
        self.assertTrue(out['ok'],out)
        self.assertEqual(run.call_args.kwargs['timeout'],1800)

    def test_operator_configuration_controls_wait(self):
        out,run=self.ask({'ask_timeout_seconds':1200})
        self.assertTrue(out['ok'],out)
        self.assertEqual(run.call_args.kwargs['timeout'],1200)

    def test_invalid_or_unbounded_configuration_cannot_spawn(self):
        for value in (0,-1,True,False,1.5,'1800',float('nan'),float('inf'),3601):
            with self.subTest(value=value):
                out,run=self.ask({'ask_timeout_seconds':value})
                self.assertFalse(out['ok'],out)
                self.assertEqual(out['error_kind'],'invalid_configuration')
                run.assert_not_called()

    def test_timeout_reports_actual_wait_without_retry(self):
        error=subprocess.TimeoutExpired('hermes',1200,stderr=b'session_id: exact_session\n')
        out,run=self.ask({'ask_timeout_seconds':1200},error)
        self.assertEqual(run.call_count,1)
        self.assertEqual(out['execution_status'],'unknown_effect')
        self.assertIn('1200',out['error'])

if __name__=='__main__':unittest.main()
