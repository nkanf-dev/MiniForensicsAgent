from __future__ import annotations

import unittest

from miniforensicsagent.loop import is_blocking_failure_for_final


class LoopTest(unittest.TestCase):
    def test_controller_guidance_failure_does_not_block_final(self) -> None:
        observation = {
            "ok": False,
            "controller_guidance": True,
            "error": "Read a different file.",
        }
        self.assertFalse(is_blocking_failure_for_final(observation))

    def test_real_tool_failure_blocks_final(self) -> None:
        observation = {
            "ok": False,
            "error": "Path does not exist",
        }
        self.assertTrue(is_blocking_failure_for_final(observation))
