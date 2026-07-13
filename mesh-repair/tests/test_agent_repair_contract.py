from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from agent_repair_contract import (  # noqa: E402
    GLOBAL_REGION_ID,
    advance_repair_state,
    decision_schema,
    initial_repair_state,
    validate_agent_response,
)
from agent_watertight_repair import combined_acceptance_gates  # noqa: E402


def packet() -> dict:
    return {
        "candidate_geometry_hash": "candidate-hash",
        "defect_signature": "defect-a",
        "views": [{"view_id": "view-1", "path": "/tmp/view.png"}],
        "review_items": [
            {
                "region_id": GLOBAL_REGION_ID,
                "kind": "global_post_audit",
                "allowed_actions": ["accept", "reject"],
            },
            {
                "region_id": "opening-1",
                "kind": "semantic_opening",
                "allowed_actions": ["cap", "ask"],
            },
        ],
    }


def response() -> dict:
    return {
        "candidate_geometry_hash": "candidate-hash",
        "decisions": [
            {
                "region_id": GLOBAL_REGION_ID,
                "action": "accept",
                "confidence": 0.9,
                "evidence_view_ids": ["view-1"],
                "rationale": "No visible semantic regression.",
            },
            {
                "region_id": "opening-1",
                "action": "cap",
                "confidence": 0.8,
                "evidence_view_ids": ["view-1"],
                "rationale": "The target requests a closed exterior shell.",
            },
        ],
    }


class AgentRepairContractTest(unittest.TestCase):
    def test_schema_binds_response_to_current_candidate_hash(self) -> None:
        schema = decision_schema(packet())
        self.assertEqual(
            schema["properties"]["candidate_geometry_hash"]["const"],
            "candidate-hash",
        )

    def test_valid_response_requires_one_decision_per_stable_region(self) -> None:
        decisions = validate_agent_response(packet(), response())
        self.assertEqual([row["region_id"] for row in decisions], [GLOBAL_REGION_ID, "opening-1"])

    def test_stale_hash_unknown_view_and_raw_coordinates_fail_closed(self) -> None:
        stale = response()
        stale["candidate_geometry_hash"] = "old-hash"
        with self.assertRaisesRegex(ValueError, "stale"):
            validate_agent_response(packet(), stale)

        unknown_view = response()
        unknown_view["decisions"][0]["evidence_view_ids"] = ["unknown"]
        with self.assertRaisesRegex(ValueError, "unknown evidence"):
            validate_agent_response(packet(), unknown_view)

        coordinates = response()
        coordinates["decisions"][0]["vertices"] = [[0.0, 0.0, 0.0]]
        with self.assertRaisesRegex(ValueError, "raw geometry"):
            validate_agent_response(packet(), coordinates)

    def test_state_stops_after_two_rounds_without_progress(self) -> None:
        state = initial_repair_state(packet(), max_rounds=5)
        state = advance_repair_state(
            state,
            candidate_geometry_hash="candidate-hash",
            defect_signature_value="defect-a",
            committed_transactions=0,
        )
        state = advance_repair_state(
            state,
            candidate_geometry_hash="candidate-hash",
            defect_signature_value="defect-a",
            committed_transactions=0,
        )
        self.assertEqual(state["status"], "stopped")
        self.assertIn(
            state["stop_reason"],
            {"two_rounds_without_progress", "candidate_or_defect_signature_oscillation"},
        )

    def test_independently_watertight_components_cannot_be_silently_merged(self) -> None:
        geometry_report = {
            "decision": {"status": "accepted"},
            "stages": {
                "source_projected_watertight_candidate": {
                    "metrics": {"topology": {"components": {"count": 1}}}
                }
            },
        }
        decisions = [
            {
                "region_id": GLOBAL_REGION_ID,
                "action": "accept",
                "confidence": 1.0,
                "evidence_view_ids": [],
                "rationale": "accepted",
            }
        ]
        gates = combined_acceptance_gates(
            geometry_report,
            {"status": "accepted"},
            decisions,
            source_independent_watertight_components=2,
        )
        self.assertFalse(gates["independent_watertight_components_not_merged"]["passed"])


if __name__ == "__main__":
    unittest.main()
